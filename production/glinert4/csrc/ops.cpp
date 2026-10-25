#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <array>

void launch_disentangled_attention(const __half* query, const __half* key, const __half* value,
                                   const int64_t* q_strides, const int64_t* k_strides, const int64_t* v_strides,
                                   const __half* pos_key, const __half* pos_query, int pos_rows,
                                   const int* delta_to_local, int position_offset, int radius, const uint8_t* mask,
                                   const int* lengths, const int* cu_seqlens, __half* output, int batch, int heads,
                                   int length, float scale_log2, cudaStream_t stream);

void launch_add_layernorm(const __half* x, const __half* residual, const __half* gamma, const __half* beta, __half* out,
                          int64_t rows, float eps, cudaStream_t stream);

namespace {

const __half* half_ptr(const torch::Tensor& tensor) { return reinterpret_cast<const __half*>(tensor.data_ptr()); }

torch::Tensor add_layernorm(torch::Tensor x, torch::Tensor residual, torch::Tensor gamma, torch::Tensor beta,
                            double eps) {
  for (const auto& tensor : {x, residual, gamma, beta}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kHalf && tensor.is_contiguous(),
                "contiguous fp16 CUDA tensors expected");
  }
  TORCH_CHECK(x.sizes() == residual.sizes() && x.size(-1) == 768 && gamma.numel() == 768 && beta.numel() == 768);
  c10::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  launch_add_layernorm(half_ptr(x), half_ptr(residual), half_ptr(gamma), half_ptr(beta),
                       reinterpret_cast<__half*>(out.data_ptr()), x.numel() / 768, static_cast<float>(eps),
                       at::cuda::getCurrentCUDAStream());
  return out;
}

torch::Tensor disentangled_attention(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor pos_key,
                                     torch::Tensor pos_query, torch::Tensor delta_to_local, int64_t position_offset,
                                     std::optional<torch::Tensor> mask, double scale_log2,
                                     std::optional<torch::Tensor> lengths, int64_t radius,
                                     std::optional<torch::Tensor> cu_seqlens, int64_t max_seqlen) {
  for (const auto& tensor : {query, key, value, pos_key, pos_query}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kHalf, "fp16 CUDA tensors expected");
    TORCH_CHECK(tensor.size(-1) == 64 && tensor.stride(-1) == 1, "head dim 64, contiguous last dim expected");
  }
  TORCH_CHECK(pos_key.is_contiguous() && pos_query.is_contiguous() && pos_key.sizes() == pos_query.sizes());
  TORCH_CHECK(delta_to_local.scalar_type() == torch::kInt && delta_to_local.is_contiguous());
  const bool packed = cu_seqlens.has_value();
  TORCH_CHECK(!packed || (cu_seqlens->scalar_type() == torch::kInt && cu_seqlens->is_contiguous()),
              "int32 cu_seqlens expected");
  const int64_t batch = packed ? cu_seqlens->size(0) - 1 : query.size(0);
  const int64_t heads = packed ? query.size(0) : query.size(1);
  const int64_t length = packed ? max_seqlen : query.size(2);
  if (mask) {
    TORCH_CHECK(mask->scalar_type() == torch::kByte && mask->is_contiguous() && mask->size(0) == batch &&
                mask->size(1) == length);
  }
  if (lengths) {
    TORCH_CHECK(lengths->scalar_type() == torch::kInt && lengths->is_contiguous() && lengths->size(0) == batch);
  }
  c10::cuda::CUDAGuard guard(query.device());
  auto output = packed ? torch::empty({query.size(1), heads * 64}, query.options())
                       : torch::empty({batch, length, heads * 64}, query.options());
  auto strides = [packed](const torch::Tensor& t) {
    return packed ? std::array<int64_t, 3>{0, t.stride(0), t.stride(1)}
                  : std::array<int64_t, 3>{t.stride(0), t.stride(1), t.stride(2)};
  };
  const auto q_strides = strides(query), k_strides = strides(key), v_strides = strides(value);
  launch_disentangled_attention(
      half_ptr(query), half_ptr(key), half_ptr(value), q_strides.data(), k_strides.data(), v_strides.data(),
      half_ptr(pos_key), half_ptr(pos_query), static_cast<int>(pos_key.size(1)), delta_to_local.data_ptr<int>(),
      static_cast<int>(position_offset), static_cast<int>(radius), mask ? mask->data_ptr<uint8_t>() : nullptr,
      lengths ? lengths->data_ptr<int>() : nullptr, packed ? cu_seqlens->data_ptr<int>() : nullptr,
      reinterpret_cast<__half*>(output.data_ptr()), batch, heads, length, static_cast<float>(scale_log2),
      at::cuda::getCurrentCUDAStream());
  return output;
}

}

TORCH_LIBRARY(glinert4, ops) {
  ops.def(
      "disentangled_attention(Tensor query, Tensor key, Tensor value, Tensor pos_key, Tensor pos_query, "
      "Tensor delta_to_local, int position_offset, Tensor? mask, float scale_log2, Tensor? lengths=None, "
      "int radius=-1, Tensor? cu_seqlens=None, int max_seqlen=0) -> Tensor");
  ops.impl("disentangled_attention", torch::kCUDA, &disentangled_attention);
  ops.def("add_layernorm(Tensor x, Tensor residual, Tensor gamma, Tensor beta, float eps) -> Tensor");
  ops.impl("add_layernorm", torch::kCUDA, &add_layernorm);
}
