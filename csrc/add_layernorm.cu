// LayerNorm(x + residual) for DeBERTa's two per-layer output blocks, fp16, hidden size 768, in one pass: PyTorch runs
// the add as its own kernel (a full write + read of the activations) before LayerNorm reads them again.
//
// Bit-identical to PyTorch 2.6.0's add + vectorized_layer_norm_kernel (aten/src/ATen/native/cuda/layer_norm_kernel.cu):
// the fp16 sum is rounded exactly as the add kernel stores it, then the same 32x4 thread layout, 4-element vectors,
// per-element Welford update, shuffle-down and shared-memory combine order, and output expression. This file is built
// with --use_fast_math, which PyTorch is not: every division PyTorch performs is spelled as __fdiv_rn (IEEE) here.
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int HIDDEN = 768;
constexpr int VEC = 4;                    // PyTorch's vec_size
constexpr int THREADS_X = 32, THREADS_Y = 4;  // PyTorch's num_threads() = 4 warps
constexpr int THREADS = THREADS_X * THREADS_Y;
constexpr int N_VEC = HIDDEN / VEC;      // 192 vectors: threads 0-63 read two, 64-127 one
constexpr int PER_THREAD = (N_VEC + THREADS - 1) / THREADS;

struct Welford {
  float mean, sigma2, count;
};

__device__ __forceinline__ Welford online_sum(float val, Welford curr) {
  const float delta = val - curr.mean;
  const float new_count = curr.count + 1.f;
  const float new_mean = curr.mean + delta * __fdiv_rn(1.f, new_count);
  return {new_mean, curr.sigma2 + delta * (val - new_mean), new_count};
}

// cuWelfordCombine(dataB, dataA), argument order kept
__device__ __forceinline__ Welford combine(Welford b, Welford a) {
  const float delta = b.mean - a.mean;
  const float count = a.count + b.count;
  if (count > 0.f) {
    const float coef = __fdiv_rn(1.f, count);
    const float na = a.count * coef, nb = b.count * coef;
    return {na * a.mean + nb * b.mean, a.sigma2 + b.sigma2 + delta * delta * a.count * nb, count};
  }
  return {0.f, 0.f, count};
}

__global__ void __launch_bounds__(THREADS) add_layernorm_kernel(const __half* __restrict__ x,
                                                                const __half* __restrict__ residual,
                                                                const __half* __restrict__ gamma,
                                                                const __half* __restrict__ beta,
                                                                __half* __restrict__ out, float eps) {
  __shared__ float buf[THREADS_Y * 3 / 2];
  const int64_t row = static_cast<int64_t>(blockIdx.x) * HIDDEN;
  const int thrx = threadIdx.x + threadIdx.y * THREADS_X;

  // the sum exactly as PyTorch's add stores it: float add, rounded to fp16
  float sum[PER_THREAD][VEC];
  Welford wd{0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < PER_THREAD; ++k) {
    const int i = thrx + k * THREADS;
    if (i < N_VEC) {
      const uint2 xv = *reinterpret_cast<const uint2*>(x + row + i * VEC);
      const uint2 rv = *reinterpret_cast<const uint2*>(residual + row + i * VEC);
      const __half* xh = reinterpret_cast<const __half*>(&xv);
      const __half* rh = reinterpret_cast<const __half*>(&rv);
#pragma unroll
      for (int ii = 0; ii < VEC; ++ii) {
        sum[k][ii] = __half2float(__float2half_rn(__half2float(xh[ii]) + __half2float(rh[ii])));
        wd = online_sum(sum[k][ii], wd);
      }
    }
  }
  // intra-warp
#pragma unroll
  for (int offset = THREADS_X >> 1; offset > 0; offset >>= 1) {
    const Welford other{__shfl_down_sync(0xffffffff, wd.mean, offset), __shfl_down_sync(0xffffffff, wd.sigma2, offset),
                        __shfl_down_sync(0xffffffff, wd.count, offset)};
    wd = combine(wd, other);
  }
  // inter-warp: upper half of the warps writes, lower half merges
  float* meansigma = buf;
  float* counts = buf + THREADS_Y;
  for (int offset = THREADS_Y / 2; offset > 0; offset /= 2) {
    if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
      const int y = threadIdx.y - offset;
      meansigma[2 * y] = wd.mean;
      meansigma[2 * y + 1] = wd.sigma2;
      counts[y] = wd.count;
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y < offset) {
      wd = combine(wd, Welford{meansigma[2 * threadIdx.y], meansigma[2 * threadIdx.y + 1], counts[threadIdx.y]});
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    meansigma[0] = wd.mean;
    meansigma[1] = __fdiv_rn(wd.sigma2, static_cast<float>(HIDDEN));
  }
  __syncthreads();
  const float mean = meansigma[0];
  const float rstd = rsqrtf(meansigma[1] + eps);

#pragma unroll
  for (int k = 0; k < PER_THREAD; ++k) {
    const int i = thrx + k * THREADS;
    if (i < N_VEC) {
      const uint2 gv = *reinterpret_cast<const uint2*>(gamma + i * VEC);
      const uint2 bv = *reinterpret_cast<const uint2*>(beta + i * VEC);
      const __half* gh = reinterpret_cast<const __half*>(&gv);
      const __half* bh = reinterpret_cast<const __half*>(&bv);
      uint2 ov;
      __half* oh = reinterpret_cast<__half*>(&ov);
#pragma unroll
      for (int ii = 0; ii < VEC; ++ii)
        oh[ii] = __float2half_rn(__half2float(gh[ii]) * (rstd * (sum[k][ii] - mean)) + __half2float(bh[ii]));
      *reinterpret_cast<uint2*>(out + row + i * VEC) = ov;
    }
  }
}

}  // namespace

void launch_add_layernorm(const __half* x, const __half* residual, const __half* gamma, const __half* beta,
                          __half* out, int64_t rows, float eps, cudaStream_t stream) {
  if (rows > 0)
    add_layernorm_kernel<<<static_cast<unsigned>(rows), dim3(THREADS_X, THREADS_Y), 0, stream>>>(x, residual, gamma,
                                                                                                beta, out, eps);
}
