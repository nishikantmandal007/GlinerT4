// DeBERTa-v2/v3 disentangled self-attention (c2p + p2c), inference forward, fp16, head dim 64, for sm_75 (Tesla T4).
//
// Same math and inputs as disentangled_flash's Triton kernel, which on a T4 is bound by three scattered global loads
// per (query, key) pair (slot index, c2p, p2c from dense [B, H, L, R] tables). Here a 64x64 tile's relative distances
// i - j span at most 127 consecutive values, so the tile only touches a window of <= 127 consecutive position slots:
//   Cq = Q_tile . pos_key[window]^T  and  Pk = K_tile . pos_query[window]^T
// are computed on tensor cores into shared memory, and every pair reads its c2p / p2c term from there.
// No relative tables in global memory, no table GEMMs, no scattered loads.
//
// Layout: 4 warps x 16 query rows = 64 rows per CTA, key tiles of 64, mma.sync m16n8k8 (fp16 in, fp32 accumulate),
// ldmatrix (sm_75+), XOR-swizzled 128-byte shared rows, FlashAttention-2 style online softmax in registers.
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int D = 64;        // head dim
constexpr int BM = 64;       // query rows per CTA
constexpr int BN = 64;       // keys per tile
constexpr int WINDOW = 128;  // position slots per tile (>= BM + BN - 1)
constexpr int THREADS = 128;
constexpr int SMEM_BYTES = (BN * D + BN * D + WINDOW * D + BM * WINDOW + BN * WINDOW) * 2;  // 64 KB

// 16-byte chunk c of row r in a [rows][64] fp16 tile, swizzled so ldmatrix's 8 rows hit 8 different bank groups
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

// c += a (16x8, row) . b (8x8, col); fp16 inputs, fp32 accumulators
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

// rows [row0, row0 + rows) of a [n][64] fp16 global matrix (row stride in elements) into a swizzled shared tile;
// rows outside [0, limit) are zero
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

// out (16 rows of this warp x 8*NT cols) = A (16 x 64, fragments in a) . B^T, B = shared [8*NT][64] swizzled
template <int NT>
__device__ __forceinline__ void warp_gemm_abt(float (*out)[4], const uint32_t (*a)[2], const __half* b, int lane) {
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    out[nt][0] = out[nt][1] = out[nt][2] = out[nt][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ks += 4) {
      uint32_t b0, b1, b2, b3;
      ldmatrix_x4(b0, b1, b2, b3, b + swz(nt * 8 + (lane & 7), ks + (lane >> 3)));
      mma(out[nt], a[ks][0], a[ks][1], b0);
      mma(out[nt], a[ks + 1][0], a[ks + 1][1], b1);
      mma(out[nt], a[ks + 2][0], a[ks + 2][1], b2);
      mma(out[nt], a[ks + 3][0], a[ks + 3][1], b3);
    }
  }
}

// A fragments of 16 rows x 64 of a swizzled shared tile, rows [row0, row0 + 16)
__device__ __forceinline__ void load_a_fragments(uint32_t (*a)[2], const __half* tile, int row0, int lane) {
#pragma unroll
  for (int ks = 0; ks < 8; ks += 2) {
    int row = row0 + (lane & 15), chunk = ks + (lane >> 4);
    ldmatrix_x4(a[ks][0], a[ks][1], a[ks + 1][0], a[ks + 1][1], tile + swz(row, chunk));
  }
}

// store a warp's 16 x 128 fp32 result as fp16 into a [rows][WINDOW] shared block at rows [row0, row0 + 16)
__device__ __forceinline__ void store_window(__half* block, const float (*acc)[4], int row0, int lane) {
  int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int nt = 0; nt < WINDOW / 8; ++nt) {
    int col = nt * 8 + 2 * t;
    *reinterpret_cast<__half2*>(block + (row0 + g) * WINDOW + col) = __floats2half2_rn(acc[nt][0], acc[nt][1]);
    *reinterpret_cast<__half2*>(block + (row0 + g + 8) * WINDOW + col) = __floats2half2_rn(acc[nt][2], acc[nt][3]);
  }
}

__global__ void __launch_bounds__(THREADS) disentangled_attention_kernel(
    const __half* __restrict__ query, const __half* __restrict__ key, const __half* __restrict__ value,
    int64_t q_sb, int64_t q_sh, int64_t q_sl, int64_t k_sb, int64_t k_sh, int64_t k_sl, int64_t v_sb, int64_t v_sh,
    int64_t v_sl, const __half* __restrict__ pos_key, const __half* __restrict__ pos_query, int pos_rows,
    const int* __restrict__ delta_to_local, int position_offset, const uint8_t* __restrict__ mask,
    __half* __restrict__ output, int heads, int length, float scale_log2) {
  extern __shared__ __align__(16) __half smem[];
  __half* k_tile = smem;
  __half* v_tile = k_tile + BN * D;
  __half* pos_tile = v_tile + BN * D;
  __half* cq = pos_tile + WINDOW * D;   // [BM][WINDOW]  Q_i . pos_key[slot]
  __half* pk = cq + BM * WINDOW;        // [BN][WINDOW]  K_j . pos_query[slot]

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int batch = blockIdx.y / heads, head = blockIdx.y % heads, i0 = blockIdx.x * BM;
  const __half* q_base = query + batch * q_sb + head * q_sh;
  const __half* k_base = key + batch * k_sb + head * k_sh;
  const __half* v_base = value + batch * v_sb + head * v_sh;
  const __half* pk_base = pos_key + static_cast<int64_t>(head) * pos_rows * D;
  const __half* pq_base = pos_query + static_cast<int64_t>(head) * pos_rows * D;
  const uint8_t* mask_row = mask ? mask + static_cast<int64_t>(batch) * length : nullptr;

  // this thread's two query rows
  const int row_a = i0 + warp * 16 + g, row_b = row_a + 8;
  const bool query_kept_a = row_a < length && (!mask_row || mask_row[row_a]);
  const bool query_kept_b = row_b < length && (!mask_row || mask_row[row_b]);
  const bool query_padded_a = row_a < length && !query_kept_a, query_padded_b = row_b < length && !query_kept_b;

  uint32_t q_frag[8][2];
  load_tile(k_tile, q_base, q_sl, i0, BM, length);
  __syncthreads();
  load_a_fragments(q_frag, k_tile, warp * 16, lane);

  float o[8][4];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
  float max_a = -INFINITY, max_b = -INFINITY, sum_a = 0.f, sum_b = 0.f;
  const int i_last = min(i0 + BM - 1, length - 1);

  for (int j0 = 0; j0 < length; j0 += BN) {
    const int j_last = min(j0 + BN - 1, length - 1);
    // slots are non-decreasing in i - j, so the tile's slots are [slot(i0 - j_last), slot(i_last - j0)]
    const int slot_lo = __ldg(delta_to_local + (i0 - j_last + position_offset));

    __syncthreads();  // previous tile's shared reads are done
    load_tile(k_tile, k_base, k_sl, j0, BN, length);
    load_tile(v_tile, v_base, v_sl, j0, BN, length);
    load_tile(pos_tile, pk_base, D, slot_lo, WINDOW, pos_rows);
    __syncthreads();

    float s[8][4];
    warp_gemm_abt<8>(s, q_frag, k_tile, lane);
    {
#ifndef SKIP_CQ
      float c[WINDOW / 8][4];
      warp_gemm_abt<WINDOW / 8>(c, q_frag, pos_tile, lane);
      store_window(cq, c, warp * 16, lane);
#endif
    }
    __syncthreads();  // pos_tile free
    load_tile(pos_tile, pq_base, D, slot_lo, WINDOW, pos_rows);
    __syncthreads();
    {
#ifndef SKIP_PK
      uint32_t k_frag[8][2];
      load_a_fragments(k_frag, k_tile, warp * 16, lane);
      float c[WINDOW / 8][4];
      warp_gemm_abt<WINDOW / 8>(c, k_frag, pos_tile, lane);
      store_window(pk, c, warp * 16, lane);
#endif
    }
    __syncthreads();

    // add c2p + p2c, scale, mask
    float tile_max_a = -INFINITY, tile_max_b = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int local_row = warp * 16 + g + (e >= 2 ? 8 : 0), col = nt * 8 + 2 * t + (e & 1);
        const int i = i0 + local_row, j = j0 + col;
        const bool padded = e >= 2 ? query_padded_b : query_padded_a;
        float score;
        if (j >= length || i >= length) {
          score = -INFINITY;
        } else if (padded) {
          score = 0.f;  // Hugging Face semantics: a padded query row attends uniformly
        } else if (mask_row && !mask_row[j]) {
          score = -INFINITY;
        } else {
#ifdef SKIP_GATHER
          score = s[nt][e] * scale_log2;
#else
          const int w = __ldg(delta_to_local + (i - j + position_offset)) - slot_lo;
          score = (s[nt][e] + __half2float(cq[local_row * WINDOW + w]) + __half2float(pk[col * WINDOW + w])) * scale_log2;
#endif
        }
        s[nt][e] = score;
        if (e >= 2) tile_max_b = fmaxf(tile_max_b, score); else tile_max_a = fmaxf(tile_max_a, score);
      }
    }
    // the four threads of a quad share a row
    tile_max_a = fmaxf(tile_max_a, __shfl_xor_sync(0xffffffff, tile_max_a, 1));
    tile_max_a = fmaxf(tile_max_a, __shfl_xor_sync(0xffffffff, tile_max_a, 2));
    tile_max_b = fmaxf(tile_max_b, __shfl_xor_sync(0xffffffff, tile_max_b, 1));
    tile_max_b = fmaxf(tile_max_b, __shfl_xor_sync(0xffffffff, tile_max_b, 2));
    const float new_max_a = fmaxf(max_a, tile_max_a), new_max_b = fmaxf(max_b, tile_max_b);
    // a row can still have no scores (masked keys so far): keep its centre finite
    const float center_a = new_max_a == -INFINITY ? 0.f : new_max_a, center_b = new_max_b == -INFINITY ? 0.f : new_max_b;
    const float correction_a = exp2f(max_a - center_a), correction_b = exp2f(max_b - center_b);
    float tile_sum_a = 0.f, tile_sum_b = 0.f;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      s[nt][0] = exp2f(s[nt][0] - center_a);
      s[nt][1] = exp2f(s[nt][1] - center_a);
      s[nt][2] = exp2f(s[nt][2] - center_b);
      s[nt][3] = exp2f(s[nt][3] - center_b);
      tile_sum_a += s[nt][0] + s[nt][1];
      tile_sum_b += s[nt][2] + s[nt][3];
      o[nt][0] *= correction_a;
      o[nt][1] *= correction_a;
      o[nt][2] *= correction_b;
      o[nt][3] *= correction_b;
    }
    sum_a = sum_a * correction_a + tile_sum_a;
    sum_b = sum_b * correction_b + tile_sum_b;
    max_a = new_max_a;
    max_b = new_max_b;

    // O += P . V; a 16x8 slice of P in accumulator layout is exactly an A fragment for one 8-key k-step
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
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

void launch_disentangled_attention(const __half* query, const __half* key, const __half* value, const int64_t* q_strides,
                                   const int64_t* k_strides, const int64_t* v_strides, const __half* pos_key,
                                   const __half* pos_query, int pos_rows, const int* delta_to_local, int position_offset,
                                   const uint8_t* mask, __half* output, int batch, int heads, int length,
                                   float scale_log2, cudaStream_t stream) {
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(disentangled_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    configured = true;
  }
  dim3 grid((length + BM - 1) / BM, batch * heads);
  disentangled_attention_kernel<<<grid, THREADS, SMEM_BYTES, stream>>>(
      query, key, value, q_strides[0], q_strides[1], q_strides[2], k_strides[0], k_strides[1], k_strides[2],
      v_strides[0], v_strides[1], v_strides[2], pos_key, pos_query, pos_rows, delta_to_local, position_offset, mask,
      output, heads, length, scale_log2);
}
