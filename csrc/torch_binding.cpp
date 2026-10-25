#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <array>

void launch_disentangled_attention(const __half* query, const __half* key, const __half* value, const int64_t* q_strides,
                                   const int64_t* k_strides, const int64_t* v_strides, const __half* pos_key,
                                   const __half* pos_query, int pos_rows, const int* delta_to_local, int position_offset,
                                   const uint8_t* mask, __half* output, int batch, int heads, int length,
                                   float scale_log2, cudaStream_t stream);
void launch_disentangled_attention_v2(const __half* query, const __half* key, const __half* value,
                                      const int64_t* q_strides, const int64_t* k_strides, const int64_t* v_strides,
                                      const __half* pos_key, const __half* pos_query, int pos_rows,
                                      const int* delta_to_local, int position_offset, const uint8_t* mask,
                                      __half* output, int batch, int heads, int length, float scale_log2,
                                      cudaStream_t stream);
void launch_disentangled_attention_v3(const __half* query, const __half* key, const __half* value,
                                      const int64_t* q_strides, const int64_t* k_strides, const int64_t* v_strides,
                                      const __half* pos_key, const __half* pos_query, int pos_rows,
                                      const int* delta_to_local, int position_offset, const uint8_t* mask,
                                      __half* output, int batch, int heads, int length, float scale_log2,
                                      cudaStream_t stream);

void launch_disentangled_attention_v4(const __half* query, const __half* key, const __half* value,
                                      const int64_t* q_strides, const int64_t* k_strides, const int64_t* v_strides,
                                      const __half* pos_key, const __half* pos_query, int pos_rows,
                                      const int* delta_to_local, int position_offset, int radius,
                                      const uint8_t* mask, const int* lengths, const int* cu_seqlens, __half* output,
                                      int batch, int heads, int length, float scale_log2, int warps,
                                      cudaStream_t stream);

void launch_add_layernorm(const __half* x, const __half* residual, const __half* gamma, const __half* beta,
                          __half* out, int64_t rows, float eps, cudaStream_t stream);

// LayerNorm(x + residual) over the last dimension (768), fp16; x and residual contiguous with the same shape
torch::Tensor add_layernorm(torch::Tensor x, torch::Tensor residual, torch::Tensor gamma, torch::Tensor beta, double eps) {
  for (const auto& tensor : {x, residual, gamma, beta}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kHalf && tensor.is_contiguous(),
                "contiguous fp16 CUDA tensors expected");
  }
  TORCH_CHECK(x.sizes() == residual.sizes() && x.size(-1) == 768 && gamma.numel() == 768 && beta.numel() == 768);
  c10::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  launch_add_layernorm(reinterpret_cast<const __half*>(x.data_ptr()), reinterpret_cast<const __half*>(residual.data_ptr()),
                       reinterpret_cast<const __half*>(gamma.data_ptr()), reinterpret_cast<const __half*>(beta.data_ptr()),
                       reinterpret_cast<__half*>(out.data_ptr()), x.numel() / 768, static_cast<float>(eps),
                       at::cuda::getCurrentCUDAStream());
  return out;
}

// query/key/value: [B, H, L, 64] fp16 (any strides, last dim contiguous); pos_key/pos_query: [H, R, 64];
// delta_to_local: int32 slot per (i - j + position_offset); mask: optional [B, L] uint8. Returns [B, L, H * 64].
// Packed (version >= 4 with cu_seqlens): query/key/value are [H, total_tokens, 64], sequences back to back as
// cu_seqlens [S + 1] int32 says, max_seqlen the longest; returns [total_tokens, H * 64].
torch::Tensor disentangled_attention(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor pos_key,
                                     torch::Tensor pos_query, torch::Tensor delta_to_local, int64_t position_offset,
                                     std::optional<torch::Tensor> mask, double scale_log2, int64_t version,
                                     std::optional<torch::Tensor> lengths, int64_t radius,
                                     std::optional<torch::Tensor> cu_seqlens, int64_t max_seqlen) {
  for (const auto& tensor : {query, key, value, pos_key, pos_query}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kHalf, "fp16 CUDA tensors expected");
    TORCH_CHECK(tensor.size(-1) == 64 && tensor.stride(-1) == 1, "head dim 64, contiguous last dim expected");
  }
  TORCH_CHECK(pos_key.is_contiguous() && pos_query.is_contiguous() && pos_key.sizes() == pos_query.sizes());
  TORCH_CHECK(delta_to_local.scalar_type() == torch::kInt && delta_to_local.is_contiguous());
  const bool packed = cu_seqlens.has_value();
  TORCH_CHECK(!packed || (version >= 4 && cu_seqlens->scalar_type() == torch::kInt && cu_seqlens->is_contiguous()),
              "packed layout needs version >= 4 and int32 cu_seqlens");
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
  // [batch, head, token] strides; packed tensors have no batch axis
  auto strides = [packed](const torch::Tensor& t) {
    return packed ? std::array<int64_t, 3>{0, t.stride(0), t.stride(1)}
                  : std::array<int64_t, 3>{t.stride(0), t.stride(1), t.stride(2)};
  };
  const auto q_s = strides(query), k_s = strides(key), v_s = strides(value);
  const int64_t* q_strides = q_s.data();
  const int64_t* k_strides = k_s.data();
  const int64_t* v_strides = v_s.data();
  if (version >= 4) {
    launch_disentangled_attention_v4(
        reinterpret_cast<const __half*>(query.data_ptr()), reinterpret_cast<const __half*>(key.data_ptr()),
        reinterpret_cast<const __half*>(value.data_ptr()), q_strides, k_strides, v_strides,
        reinterpret_cast<const __half*>(pos_key.data_ptr()), reinterpret_cast<const __half*>(pos_query.data_ptr()),
        static_cast<int>(pos_key.size(1)), delta_to_local.data_ptr<int>(), static_cast<int>(position_offset),
        static_cast<int>(radius), mask ? mask->data_ptr<uint8_t>() : nullptr,
        lengths ? lengths->data_ptr<int>() : nullptr, packed ? cu_seqlens->data_ptr<int>() : nullptr,
        reinterpret_cast<__half*>(output.data_ptr()), batch, heads, length, static_cast<float>(scale_log2),
        version == 5 ? 8 : 4, at::cuda::getCurrentCUDAStream());
    return output;
  }
  (version == 1 ? launch_disentangled_attention
   : version == 2 ? launch_disentangled_attention_v2 : launch_disentangled_attention_v3)(
      reinterpret_cast<const __half*>(query.data_ptr()), reinterpret_cast<const __half*>(key.data_ptr()),
      reinterpret_cast<const __half*>(value.data_ptr()), q_strides, k_strides, v_strides,
      reinterpret_cast<const __half*>(pos_key.data_ptr()), reinterpret_cast<const __half*>(pos_query.data_ptr()),
      static_cast<int>(pos_key.size(1)), delta_to_local.data_ptr<int>(), static_cast<int>(position_offset),
      mask ? mask->data_ptr<uint8_t>() : nullptr, reinterpret_cast<__half*>(output.data_ptr()), batch, heads, length,
      static_cast<float>(scale_log2), at::cuda::getCurrentCUDAStream());
  return output;
}

#ifndef GLINER_T4_NS
#define GLINER_T4_NS gliner_t4
#endif
// one level of indirection so the namespace macro expands before TORCH_LIBRARY stringizes it (kernel-builder's pattern)
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
TORCH_LIBRARY_EXPAND(GLINER_T4_NS, ops) {
  ops.def("disentangled_attention(Tensor query, Tensor key, Tensor value, Tensor pos_key, Tensor pos_query, "
          "Tensor delta_to_local, int position_offset, Tensor? mask, float scale_log2, int version=4, Tensor? lengths=None, int radius=-1, "
          "Tensor? cu_seqlens=None, int max_seqlen=0) -> Tensor");
  ops.impl("disentangled_attention", torch::kCUDA, &disentangled_attention);
  ops.def("add_layernorm(Tensor x, Tensor residual, Tensor gamma, Tensor beta, float eps) -> Tensor");
  ops.impl("add_layernorm", torch::kCUDA, &add_layernorm);
}
