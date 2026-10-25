// v3 of the sm_75 disentangled attention kernel (same inputs, math and output as v1 / v2 / disentangled_flash).
//
// v3 over v2: the c2p/p2c blocks are stored rotated (row r starts at column offset 8 * (r & 7)) so quads and rows
// spread over all 32 banks on both the store and the lookup (v2's 192-byte p2c rows put a quad in one bank), and a
// thread resolves the 10 distinct relative distances its 32 scores need once per tile instead of 32 global lookups.
//
// v1 was bound by its skeleton, not its math: 64 KB of shared memory left one 4-warp CTA per SM, and global loads
// never overlapped compute. v2 keeps every relative term on chip as v1 does, but:
//   - 64 query rows x 32 keys per tile and 32 KB of shared memory, so two CTAs (8 warps) share an SM
//   - each warp's c2p window covers only its own 16 rows (<= 47 slots, warp-private, no CTA barrier)
//   - the CTA's p2c window covers 64 rows x 32 keys (<= 95 slots)
//   - the next K/V tile is fetched into registers while the current one is computed
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int D = 64;
constexpr int BM = 64;           // query rows per CTA (4 warps x 16)
constexpr int BN = 32;           // keys per tile
constexpr int CTA_WINDOW = 96;   // >= BM + BN - 1 slots
constexpr int WARP_WINDOW = 48;  // >= 16 + BN - 1 slots
constexpr int THREADS = 128;
constexpr int KV_CHUNKS = BN * 8 * 2 / THREADS;  // 16-byte chunks of K and V per thread per tile

__device__ __forceinline__ int swz(int row, int chunk) { return row * D + ((chunk ^ (row & 7)) << 3); }

// element (row, col) of a [rows][STRIDE] fp16 block, rotated per row; even columns stay pair-aligned
template <int STRIDE>
__device__ __forceinline__ int rot(int row, int col) { return row * STRIDE + (col + ((row & 7) << 3)) % STRIDE; }

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void load_tile(__half* tile, const __half* base, int64_t row_stride, int row0, int rows,
                                          int limit) {
  for (int index = threadIdx.x; index < rows * 8; index += THREADS) {
    int row = index >> 3, chunk = index & 7, source = row0 + row;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (source >= 0 && source < limit)
      value = __ldg(reinterpret_cast<const uint4*>(base + source * row_stride + chunk * 8));
    *reinterpret_cast<uint4*>(tile + swz(row, chunk)) = value;
  }
}

// K and V rows [j0, j0 + BN) into registers: chunk index < BN * 8 is K, the rest V
__device__ __forceinline__ void fetch_kv(uint4 (&regs)[KV_CHUNKS], const __half* k_base, int64_t k_sl,
                                         const __half* v_base, int64_t v_sl, int j0, int length) {
#pragma unroll
  for (int m = 0; m < KV_CHUNKS; ++m) {
    const int index = threadIdx.x + m * THREADS, is_v = index >= BN * 8, local = index & (BN * 8 - 1);
    const int row = j0 + (local >> 3), chunk = local & 7;
    regs[m] = make_uint4(0, 0, 0, 0);
    if (row < length)
      regs[m] = __ldg(reinterpret_cast<const uint4*>((is_v ? v_base + row * v_sl : k_base + row * k_sl) + chunk * 8));
  }
}

__device__ __forceinline__ void store_kv(const uint4 (&regs)[KV_CHUNKS], __half* k_tile, __half* v_tile) {
#pragma unroll
  for (int m = 0; m < KV_CHUNKS; ++m) {
    const int index = threadIdx.x + m * THREADS, is_v = index >= BN * 8, local = index & (BN * 8 - 1);
    *reinterpret_cast<uint4*>((is_v ? v_tile : k_tile) + swz(local >> 3, local & 7)) = regs[m];
  }
}

// acc (16 x 8*NT) = A (16 x 64 fragments) . B^T, B rows [row0, row0 + 8*NT) of a swizzled [rows][64] shared tile
template <int NT>
__device__ __forceinline__ void warp_gemm_abt(float (*acc)[4], const uint32_t (*a)[2], const __half* b, int row0,
                                              int lane) {
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ks += 4) {
      uint32_t b0, b1, b2, b3;
      ldmatrix_x4(b0, b1, b2, b3, b + swz(row0 + nt * 8 + (lane & 7), ks + (lane >> 3)));
      mma(acc[nt], a[ks][0], a[ks][1], b0);
      mma(acc[nt], a[ks + 1][0], a[ks + 1][1], b1);
      mma(acc[nt], a[ks + 2][0], a[ks + 2][1], b2);
      mma(acc[nt], a[ks + 3][0], a[ks + 3][1], b3);
    }
  }
}

__device__ __forceinline__ void load_a_fragments(uint32_t (*a)[2], const __half* tile, int row0, int lane) {
#pragma unroll
  for (int ks = 0; ks < 8; ks += 2)
    ldmatrix_x4(a[ks][0], a[ks][1], a[ks + 1][0], a[ks + 1][1], tile + swz(row0 + (lane & 15), ks + (lane >> 4)));
}

// a warp's 16 x 8*NT accumulators as fp16 into rows [0, 16) of a [16][stride] block, columns from col0
template <int NT, int STRIDE>
__device__ __forceinline__ void store_block(__half* block, int col0, const float (*acc)[4], int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int col = col0 + nt * 8 + 2 * t;
    *reinterpret_cast<__half2*>(block + rot<STRIDE>(g, col)) = __floats2half2_rn(acc[nt][0], acc[nt][1]);
    *reinterpret_cast<__half2*>(block + rot<STRIDE>(g + 8, col)) = __floats2half2_rn(acc[nt][2], acc[nt][3]);
  }
}

__global__ void __launch_bounds__(THREADS, 2) disentangled_attention_v3_kernel(
    const __half* __restrict__ query, const __half* __restrict__ key, const __half* __restrict__ value,
    int64_t q_sb, int64_t q_sh, int64_t q_sl, int64_t k_sb, int64_t k_sh, int64_t k_sl, int64_t v_sb, int64_t v_sh,
    int64_t v_sl, const __half* __restrict__ pos_key, const __half* __restrict__ pos_query, int pos_rows,
    const int* __restrict__ delta_to_local, int position_offset, const uint8_t* __restrict__ mask,
    __half* __restrict__ output, int heads, int length, float scale_log2) {
  __shared__ __align__(16) __half k_tile[BN * D];
  __shared__ __align__(16) __half v_tile[BN * D];
  __shared__ __align__(16) __half pos_tile[CTA_WINDOW * D];         // pos_key, then pos_query, window rows
  __shared__ __align__(16) __half cq[4][16 * WARP_WINDOW];          // per warp: Q_i . pos_key[slot]
  __shared__ __align__(16) __half pk[BN * CTA_WINDOW];              // K_j . pos_query[slot]

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int batch = blockIdx.y / heads, head = blockIdx.y % heads, i0 = blockIdx.x * BM;
  const __half* q_base = query + batch * q_sb + head * q_sh;
  const __half* k_base = key + batch * k_sb + head * k_sh;
  const __half* v_base = value + batch * v_sb + head * v_sh;
  const __half* pk_base = pos_key + static_cast<int64_t>(head) * pos_rows * D;
  const __half* pq_base = pos_query + static_cast<int64_t>(head) * pos_rows * D;
  const uint8_t* mask_row = mask ? mask + static_cast<int64_t>(batch) * length : nullptr;
  __half* my_cq = cq[warp];

  const int row_a = i0 + warp * 16 + g, row_b = row_a + 8;
  const bool padded_a = row_a < length && mask_row && !mask_row[row_a];
  const bool padded_b = row_b < length && mask_row && !mask_row[row_b];
  const int i_last = min(i0 + BM - 1, length - 1), warp_row0 = min(i0 + warp * 16, i_last);

  uint32_t q_frag[8][2];
  load_tile(pos_tile, q_base, q_sl, i0, BM, length);
  __syncthreads();
  load_a_fragments(q_frag, pos_tile, warp * 16, lane);

  uint4 kv[KV_CHUNKS];
  fetch_kv(kv, k_base, k_sl, v_base, v_sl, 0, length);

  float o[8][4];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
  float max_a = -INFINITY, max_b = -INFINITY, sum_a = 0.f, sum_b = 0.f;

  for (int j0 = 0; j0 < length; j0 += BN) {
    const int j_last = min(j0 + BN - 1, length - 1);
    // slots are non-decreasing in i - j: the CTA's window starts at slot(i0 - j_last), the warp's at slot(row0 - j_last)
    const int slot_lo = __ldg(delta_to_local + (i0 - j_last + position_offset));
    const int warp_lo = min(__ldg(delta_to_local + (warp_row0 - j_last + position_offset)) - slot_lo,
                            CTA_WINDOW - WARP_WINDOW);

    __syncthreads();  // last tile's shared reads are done
    store_kv(kv, k_tile, v_tile);
#ifndef SKIP_POS
    load_tile(pos_tile, pk_base, D, slot_lo, CTA_WINDOW, pos_rows);
#endif
    __syncthreads();
    if (j0 + BN < length) fetch_kv(kv, k_base, k_sl, v_base, v_sl, j0 + BN, length);  // lands while we compute

    float s[4][4];
    warp_gemm_abt<4>(s, q_frag, k_tile, 0, lane);
    {
#ifndef SKIP_CQ
      float c[WARP_WINDOW / 8][4];
      warp_gemm_abt<WARP_WINDOW / 8>(c, q_frag, pos_tile, warp_lo, lane);
      store_block<WARP_WINDOW / 8, WARP_WINDOW>(my_cq, 0, c, lane);
#endif
    }
#ifndef SKIP_POS
    __syncthreads();  // pos_tile free
    load_tile(pos_tile, pq_base, D, slot_lo, CTA_WINDOW, pos_rows);
    __syncthreads();
#endif
    {
      // p2c block [32 keys][96 slots]: warp w takes keys (w >> 1) * 16.. and slot half (w & 1)
#ifndef SKIP_PK
      uint32_t k_frag[8][2];
      load_a_fragments(k_frag, k_tile, (warp >> 1) * 16, lane);
      float c[CTA_WINDOW / 16][4];
      warp_gemm_abt<CTA_WINDOW / 16>(c, k_frag, pos_tile, (warp & 1) * (CTA_WINDOW / 2), lane);
      store_block<CTA_WINDOW / 16, CTA_WINDOW>(pk + (warp >> 1) * 16 * CTA_WINDOW, (warp & 1) * (CTA_WINDOW / 2), c,
                                               lane);
#endif
    }
    __syncthreads();

    // this thread's scores sit at i - j = base - 8 * m - e, m in [-1, 3] (row b is row a + 8), e in {0, 1}
    int slot_of[5][2];
    {
      const int base = row_a - j0 - 2 * t;
#pragma unroll
      for (int m = 0; m < 5; ++m)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int delta = base - 8 * (m - 1) - e;
          // clamp keeps the lookup inside the table for out-of-range rows/keys, which are never used
          const int clamped = min(max(delta, i0 - j_last), i_last - j0);
          slot_of[m][e] = __ldg(delta_to_local + (clamped + position_offset)) - slot_lo;
        }
    }
    float tile_max_a = -INFINITY, tile_max_b = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int local_row = g + (e >= 2 ? 8 : 0), col = nt * 8 + 2 * t + (e & 1);
        const int i = (e >= 2 ? row_b : row_a), j = j0 + col;
        float score;
        if (j >= length || i >= length) {
          score = -INFINITY;
        } else if (e >= 2 ? padded_b : padded_a) {
          score = 0.f;  // Hugging Face semantics: a padded query row attends uniformly
        } else if (mask_row && !mask_row[j]) {
          score = -INFINITY;
        } else {
#ifdef SKIP_GATHER
          score = s[nt][e] * scale_log2;
#else
          const int w = slot_of[nt + (e >= 2 ? 0 : 1)][e & 1];
          score = (s[nt][e] + __half2float(my_cq[rot<WARP_WINDOW>(local_row, w - warp_lo)]) +
                   __half2float(pk[rot<CTA_WINDOW>(col, w)])) * scale_log2;
#endif
        }
        s[nt][e] = score;
        if (e >= 2) tile_max_b = fmaxf(tile_max_b, score); else tile_max_a = fmaxf(tile_max_a, score);
      }
    }
    tile_max_a = fmaxf(tile_max_a, __shfl_xor_sync(0xffffffff, tile_max_a, 1));
    tile_max_a = fmaxf(tile_max_a, __shfl_xor_sync(0xffffffff, tile_max_a, 2));
    tile_max_b = fmaxf(tile_max_b, __shfl_xor_sync(0xffffffff, tile_max_b, 1));
    tile_max_b = fmaxf(tile_max_b, __shfl_xor_sync(0xffffffff, tile_max_b, 2));
    const float new_max_a = fmaxf(max_a, tile_max_a), new_max_b = fmaxf(max_b, tile_max_b);
    const float center_a = new_max_a == -INFINITY ? 0.f : new_max_a, center_b = new_max_b == -INFINITY ? 0.f : new_max_b;
    const float correction_a = exp2f(max_a - center_a), correction_b = exp2f(max_b - center_b);
    float tile_sum_a = 0.f, tile_sum_b = 0.f;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      s[nt][0] = exp2f(s[nt][0] - center_a);
      s[nt][1] = exp2f(s[nt][1] - center_a);
      s[nt][2] = exp2f(s[nt][2] - center_b);
      s[nt][3] = exp2f(s[nt][3] - center_b);
      tile_sum_a += s[nt][0] + s[nt][1];
      tile_sum_b += s[nt][2] + s[nt][3];
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      o[nt][0] *= correction_a;
      o[nt][1] *= correction_a;
      o[nt][2] *= correction_b;
      o[nt][3] *= correction_b;
    }
    sum_a = sum_a * correction_a + tile_sum_a;
    sum_b = sum_b * correction_b + tile_sum_b;
    max_a = new_max_a;
    max_b = new_max_b;

#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint32_t a0 = pack_half2(s[ks][0], s[ks][1]), a1 = pack_half2(s[ks][2], s[ks][3]);
#pragma unroll
      for (int nt = 0; nt < 8; nt += 4) {
        uint32_t b0, b1, b2, b3;
        ldmatrix_x4_trans(b0, b1, b2, b3, v_tile + swz(ks * 8 + (lane & 7), nt + (lane >> 3)));
        mma(o[nt], a0, a1, b0);
        mma(o[nt + 1], a0, a1, b1);
        mma(o[nt + 2], a0, a1, b2);
        mma(o[nt + 3], a0, a1, b3);
      }
    }
  }

  sum_a += __shfl_xor_sync(0xffffffff, sum_a, 1);
  sum_a += __shfl_xor_sync(0xffffffff, sum_a, 2);
  sum_b += __shfl_xor_sync(0xffffffff, sum_b, 1);
  sum_b += __shfl_xor_sync(0xffffffff, sum_b, 2);
  const int64_t out_row = static_cast<int64_t>(heads) * D;
  __half* out_base = output + static_cast<int64_t>(batch) * length * out_row + head * D;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    const int col = nt * 8 + 2 * t;
    if (row_a < length)
      *reinterpret_cast<__half2*>(out_base + row_a * out_row + col) = __floats2half2_rn(o[nt][0] / sum_a, o[nt][1] / sum_a);
    if (row_b < length)
      *reinterpret_cast<__half2*>(out_base + row_b * out_row + col) = __floats2half2_rn(o[nt][2] / sum_b, o[nt][3] / sum_b);
  }
}

}  // namespace

void launch_disentangled_attention_v3(const __half* query, const __half* key, const __half* value,
                                      const int64_t* q_strides, const int64_t* k_strides, const int64_t* v_strides,
                                      const __half* pos_key, const __half* pos_query, int pos_rows,
                                      const int* delta_to_local, int position_offset, const uint8_t* mask,
                                      __half* output, int batch, int heads, int length, float scale_log2,
                                      cudaStream_t stream) {
  dim3 grid((length + BM - 1) / BM, batch * heads);
  disentangled_attention_v3_kernel<<<grid, THREADS, 0, stream>>>(
      query, key, value, q_strides[0], q_strides[1], q_strides[2], k_strides[0], k_strides[1], k_strides[2],
      v_strides[0], v_strides[1], v_strides[2], pos_key, pos_query, pos_rows, delta_to_local, position_offset, mask,
      output, heads, length, scale_log2);
}
