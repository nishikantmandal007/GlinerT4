// v4 of the sm_75 disentangled attention kernel (same inputs, math and output as v1-v3 / disentangled_flash).
//
// Over v2 (v3's bank-conflict rotation did not pay and is dropped):
//   - templated on warps per CTA: 4 warps x 16 rows (32 KB smem, 2 CTAs/SM) or 8 warps x 16 rows (50 KB, 1 CTA/SM,
//     every p2c block and position window serves twice the rows)
//   - the next tile's K, V, pos_key window and pos_query window are fetched into registers while the current tile is
//     computed, so the shared-memory refills between barriers are register stores only
//   - packed (variable-length) mode: with cu_seqlens, sequences sit back to back with no padding at all, as in
//     GLiNER2's packed path; each CTA reads its sequence's start and length
//   - relative distances within +-radius (the un-bucketed region, |i - j| <= position_buckets / 2) map to slots
//     arithmetically, so only log-bucketed pairs read the slot table
//   - per-row padding: rows and keys past a sequence's last real token are skipped (work and loads), and those output
//     rows are written as zeros, as GLiNER2's packed path does; a padded row's output never reaches a real token,
//     since every real query masks padded keys. Interior mask zeros keep exact Hugging Face semantics.
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int D = 64;
constexpr int BN = 32;           // keys per tile
constexpr int WARP_WINDOW = 48;  // >= 16 + BN - 1 slots

template <int WARPS>
struct Shape {
  static constexpr int THREADS = WARPS * 32;
  static constexpr int BM = WARPS * 16;
  static constexpr int CTA_WINDOW = (BM + BN - 1 + 15) / 16 * 16;  // 96 or 160 slots
  static constexpr int KV_CHUNKS = BN * 8 * 2 / THREADS;           // 16-byte K/V chunks per thread
  static constexpr int POS_CHUNKS = CTA_WINDOW * 8 / THREADS;      // 16-byte position-window chunks per thread
  static constexpr int PK_PARTS = WARPS / 2;                       // slot parts of the p2c block per key block
  static constexpr int PK_COLS = CTA_WINDOW / PK_PARTS;            // 48 or 40
  static constexpr int SMEM = (2 * BN * D + CTA_WINDOW * D + WARPS * 16 * WARP_WINDOW + BN * CTA_WINDOW) * 2;
  static_assert(CTA_WINDOW * 8 % THREADS == 0 && PK_COLS % 8 == 0, "tile shape");
};

__device__ __forceinline__ int swz(int row, int chunk) { return row * D + ((chunk ^ (row & 7)) << 3); }

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

// rows [row0, row0 + rows) of a [n][64] fp16 global matrix into registers, chunk index = threadIdx.x + m * THREADS;
// rows outside [0, limit) are zero
template <int CHUNKS, int THREADS>
__device__ __forceinline__ void fetch_rows(uint4 (&regs)[CHUNKS], const __half* base, int64_t row_stride, int row0,
                                           int limit) {
#pragma unroll
  for (int m = 0; m < CHUNKS; ++m) {
    const int index = threadIdx.x + m * THREADS, row = row0 + (index >> 3);
    regs[m] = make_uint4(0, 0, 0, 0);
    if (row >= 0 && row < limit)
      regs[m] = __ldg(reinterpret_cast<const uint4*>(base + row * row_stride + (index & 7) * 8));
  }
}

template <int CHUNKS, int THREADS>
__device__ __forceinline__ void store_rows(const uint4 (&regs)[CHUNKS], __half* tile) {
#pragma unroll
  for (int m = 0; m < CHUNKS; ++m) {
    const int index = threadIdx.x + m * THREADS;
    *reinterpret_cast<uint4*>(tile + swz(index >> 3, index & 7)) = regs[m];
  }
}

template <int CHUNKS, int THREADS>
__device__ __forceinline__ void copy_rows_sync(__half* tile, const __half* base, int64_t stride, int row0, int limit) {
#pragma unroll
  for (int m = 0; m < CHUNKS; ++m) {
    const int index = threadIdx.x + m * THREADS, row = row0 + (index >> 3);
    uint4 value = make_uint4(0, 0, 0, 0);
    if (row >= 0 && row < limit)
      value = __ldg(reinterpret_cast<const uint4*>(base + row * stride + (index & 7) * 8));
    *reinterpret_cast<uint4*>(tile + swz(index >> 3, index & 7)) = value;
  }
}

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

template <int NT>
__device__ __forceinline__ void store_block(__half* block, int stride, int col0, const float (*acc)[4], int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int col = col0 + nt * 8 + 2 * t;
    *reinterpret_cast<__half2*>(block + g * stride + col) = __floats2half2_rn(acc[nt][0], acc[nt][1]);
    *reinterpret_cast<__half2*>(block + (g + 8) * stride + col) = __floats2half2_rn(acc[nt][2], acc[nt][3]);
  }
}

template <int WARPS>
__global__ void __launch_bounds__(Shape<WARPS>::THREADS, 8 / WARPS) disentangled_attention_v4_kernel(
    const __half* __restrict__ query, const __half* __restrict__ key, const __half* __restrict__ value,
    int64_t q_sb, int64_t q_sh, int64_t q_sl, int64_t k_sb, int64_t k_sh, int64_t k_sl, int64_t v_sb, int64_t v_sh,
    int64_t v_sl, const __half* __restrict__ pos_key, const __half* __restrict__ pos_query, int pos_rows,
    const int* __restrict__ delta_to_local, int position_offset, int radius, const uint8_t* __restrict__ mask,
    const int* __restrict__ lengths, const int* __restrict__ cu_seqlens, __half* __restrict__ output, int heads,
    int length, float scale_log2) {
  using S = Shape<WARPS>;
  extern __shared__ __align__(16) __half smem[];
  __half* k_tile = smem;
  __half* v_tile = k_tile + BN * D;
  __half* pos_tile = v_tile + BN * D;
  __half* cq = pos_tile + S::CTA_WINDOW * D;
  __half* pk = cq + WARPS * 16 * WARP_WINDOW;

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int batch = blockIdx.y / heads, head = blockIdx.y % heads, i0 = blockIdx.x * S::BM;
  const int64_t out_row = static_cast<int64_t>(heads) * D;
  const int row_a = i0 + warp * 16 + g, row_b = row_a + 8;
  // a sequence's real tokens are [0, real); later rows and keys are padding (padded layout) or the next sequence
  // (packed layout, where rows are only written up to real)
  const int64_t start = cu_seqlens ? cu_seqlens[batch] : 0;
  const int real = cu_seqlens ? cu_seqlens[batch + 1] - static_cast<int>(start) : lengths ? lengths[batch] : length;
  const int row_limit = cu_seqlens ? real : length;
  __half* out_base = output + (cu_seqlens ? start : static_cast<int64_t>(batch) * length) * out_row + head * D;

  if (i0 >= real) {  // a tile of padding rows only
    if (cu_seqlens) return;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int col = nt * 8 + 2 * t;
      if (row_a < length) *reinterpret_cast<__half2*>(out_base + row_a * out_row + col) = __floats2half2_rn(0.f, 0.f);
      if (row_b < length) *reinterpret_cast<__half2*>(out_base + row_b * out_row + col) = __floats2half2_rn(0.f, 0.f);
    }
    return;
  }

  const __half* q_base = query + (cu_seqlens ? start * q_sl : batch * q_sb) + head * q_sh;
  const __half* k_base = key + (cu_seqlens ? start * k_sl : batch * k_sb) + head * k_sh;
  const __half* v_base = value + (cu_seqlens ? start * v_sl : batch * v_sb) + head * v_sh;
  const __half* pk_base = pos_key + static_cast<int64_t>(head) * pos_rows * D;
  const __half* pq_base = pos_query + static_cast<int64_t>(head) * pos_rows * D;
  const uint8_t* mask_row = mask && !cu_seqlens ? mask + static_cast<int64_t>(batch) * length : nullptr;
  const int zero_slot = __ldg(delta_to_local + position_offset);  // slot of i - j = 0
  __half* my_cq = cq + warp * 16 * WARP_WINDOW;

  const bool padded_a = row_a < real && mask_row && !mask_row[row_a];
  const bool padded_b = row_b < real && mask_row && !mask_row[row_b];
  const int i_last = min(i0 + S::BM - 1, real - 1), warp_row0 = min(i0 + warp * 16, i_last);

  // Q through the position buffer once
  {
    for (int index = threadIdx.x; index < S::BM * 8; index += S::THREADS) {
      const int row = i0 + (index >> 3);
      uint4 value4 = make_uint4(0, 0, 0, 0);
      if (row < real) value4 = __ldg(reinterpret_cast<const uint4*>(q_base + row * q_sl + (index & 7) * 8));
      *reinterpret_cast<uint4*>(pos_tile + swz(index >> 3, index & 7)) = value4;
    }
  }
  __syncthreads();
  uint32_t q_frag[8][2];
  load_a_fragments(q_frag, pos_tile, warp * 16, lane);

  // tile 0's operands
  auto window_start = [&](int j0) {
    return __ldg(delta_to_local + (i0 - min(j0 + BN - 1, real - 1) + position_offset));
  };
  uint4 k_regs[S::KV_CHUNKS / 2], v_regs[S::KV_CHUNKS / 2];
  int slot_lo_next = window_start(0);
  fetch_rows<S::KV_CHUNKS / 2, S::THREADS>(k_regs, k_base, k_sl, 0, real);
  fetch_rows<S::KV_CHUNKS / 2, S::THREADS>(v_regs, v_base, v_sl, 0, real);

  float o[8][4];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
  float max_a = -INFINITY, max_b = -INFINITY, sum_a = 0.f, sum_b = 0.f;

  for (int j0 = 0; j0 < real; j0 += BN) {
    const int j_last = min(j0 + BN - 1, real - 1);
    const int slot_lo = slot_lo_next;
    const int warp_lo = min(__ldg(delta_to_local + (warp_row0 - j_last + position_offset)) - slot_lo,
                            S::CTA_WINDOW - WARP_WINDOW);
    const bool has_next = j0 + BN < real;

    __syncthreads();  // last tile's shared reads are done
    store_rows<S::KV_CHUNKS / 2, S::THREADS>(k_regs, k_tile);
    store_rows<S::KV_CHUNKS / 2, S::THREADS>(v_regs, v_tile);
    copy_rows_sync<S::POS_CHUNKS, S::THREADS>(pos_tile, pk_base, D, slot_lo, pos_rows);
    __syncthreads();
    if (has_next) {  // lands while this tile computes
      slot_lo_next = window_start(j0 + BN);
      fetch_rows<S::KV_CHUNKS / 2, S::THREADS>(k_regs, k_base, k_sl, j0 + BN, real);
      fetch_rows<S::KV_CHUNKS / 2, S::THREADS>(v_regs, v_base, v_sl, j0 + BN, real);
    }

    float s[4][4];
    warp_gemm_abt<4>(s, q_frag, k_tile, 0, lane);
    {
      float c[WARP_WINDOW / 8][4];
      warp_gemm_abt<WARP_WINDOW / 8>(c, q_frag, pos_tile, warp_lo, lane);
      store_block<WARP_WINDOW / 8>(my_cq, WARP_WINDOW, 0, c, lane);
    }
    __syncthreads();  // pos_tile free
    copy_rows_sync<S::POS_CHUNKS, S::THREADS>(pos_tile, pq_base, D, slot_lo, pos_rows);
    __syncthreads();
    {
      // p2c block [32 keys][CTA_WINDOW slots]: warp w takes key block w % 2 and slot part w / 2
      uint32_t k_frag[8][2];
      load_a_fragments(k_frag, k_tile, (warp & 1) * 16, lane);
      float c[S::PK_COLS / 8][4];
      warp_gemm_abt<S::PK_COLS / 8>(c, k_frag, pos_tile, (warp >> 1) * S::PK_COLS, lane);
      store_block<S::PK_COLS / 8>(pk + (warp & 1) * 16 * S::CTA_WINDOW, S::CTA_WINDOW, (warp >> 1) * S::PK_COLS, c,
                                  lane);
    }
    __syncthreads();

    float tile_max_a = -INFINITY, tile_max_b = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int local_row = g + (e >= 2 ? 8 : 0), col = nt * 8 + 2 * t + (e & 1);
        const int i = (e >= 2 ? row_b : row_a), j = j0 + col;
        float score;
        if (j >= real || i >= real) {
          score = -INFINITY;
        } else if (e >= 2 ? padded_b : padded_a) {
          score = 0.f;  // Hugging Face semantics: a padded query row attends uniformly
        } else if (mask_row && !mask_row[j]) {
          score = -INFINITY;
        } else {
          const int delta = i - j;
          const int w = (delta <= radius && delta >= -radius ? delta + zero_slot
                                                              : __ldg(delta_to_local + (delta + position_offset))) - slot_lo;
          score = (s[nt][e] + __half2float(my_cq[local_row * WARP_WINDOW + w - warp_lo]) +
                   __half2float(pk[col * S::CTA_WINDOW + w])) * scale_log2;
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
  // rows past the real length (padding) are written as zeros
  const float inv_a = row_a < real ? 1.f / sum_a : 0.f, inv_b = row_b < real ? 1.f / sum_b : 0.f;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    const int col = nt * 8 + 2 * t;
    if (row_a < row_limit)
      *reinterpret_cast<__half2*>(out_base + row_a * out_row + col) = __floats2half2_rn(o[nt][0] * inv_a, o[nt][1] * inv_a);
    if (row_b < row_limit)
      *reinterpret_cast<__half2*>(out_base + row_b * out_row + col) = __floats2half2_rn(o[nt][2] * inv_b, o[nt][3] * inv_b);
  }
}

template <int WARPS>
void launch(const __half* query, const __half* key, const __half* value, const int64_t* q_strides,
            const int64_t* k_strides, const int64_t* v_strides, const __half* pos_key, const __half* pos_query,
            int pos_rows, const int* delta_to_local, int position_offset, int radius, const uint8_t* mask,
            const int* lengths, const int* cu_seqlens, __half* output, int batch, int heads, int length,
            float scale_log2, cudaStream_t stream) {
  using S = Shape<WARPS>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(disentangled_attention_v4_kernel<WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, S::SMEM);
    configured = true;
  }
  dim3 grid((length + S::BM - 1) / S::BM, batch * heads);
  disentangled_attention_v4_kernel<WARPS><<<grid, S::THREADS, S::SMEM, stream>>>(
      query, key, value, q_strides[0], q_strides[1], q_strides[2], k_strides[0], k_strides[1], k_strides[2],
      v_strides[0], v_strides[1], v_strides[2], pos_key, pos_query, pos_rows, delta_to_local, position_offset, radius,
      mask, lengths, cu_seqlens, output, heads, length, scale_log2);
}

}  // namespace

void launch_disentangled_attention_v4(const __half* query, const __half* key, const __half* value,
                                      const int64_t* q_strides, const int64_t* k_strides, const int64_t* v_strides,
                                      const __half* pos_key, const __half* pos_query, int pos_rows,
                                      const int* delta_to_local, int position_offset, int radius,
                                      const uint8_t* mask, const int* lengths, const int* cu_seqlens, __half* output,
                                      int batch, int heads, int length, float scale_log2, int warps,
                                      cudaStream_t stream) {
  (warps == 8 ? launch<8> : launch<4>)(query, key, value, q_strides, k_strides, v_strides, pos_key, pos_query, pos_rows,
                                      delta_to_local, position_offset, radius, mask, lengths, cu_seqlens, output, batch,
                                      heads, length, scale_log2, stream);
}
