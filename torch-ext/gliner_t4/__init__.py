# Dev loader for the sm_75 disentangled attention kernel, and a drop-in for disentangled_flash's inference attention.
# shortcut: JIT build with torch.utils.cpp_extension (kernel-builder/Nix is not on the VM); package with kernel-builder
# (build.toml + TORCH_LIBRARY_EXPAND) or an ahead-of-time .so before shipping.
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from ._build import CUDA_FLAGS, SOURCES

# a runtime image has no compiler: it ships the library built ahead of time (build.py) next to this file
_PREBUILT = Path(__file__).with_name("gliner_t4.so")
if _PREBUILT.exists():
    torch.ops.load_library(str(_PREBUILT))
else:
    load(name="gliner_t4", sources=SOURCES, extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"], is_python_module=False,
         verbose=bool(os.getenv("GLINER_T4_VERBOSE")))
ops = torch.ops.gliner_t4
VERSION = int(os.getenv("GLINER_T4_VERSION", "4"))
SCALE_LOG2 = (64 * 3) ** -0.5 * 1.4426950408889634  # 1 / sqrt(head_dim * (1 + c2p + p2c)), in log2 units


def attention(query, key, value, pos_key, pos_query, delta_to_local, position_offset, mask, scale_log2, version=None,
              lengths=None, radius=-1, cu_seqlens=None, max_seqlen=0):
    return ops.disentangled_attention(query, key, value, pos_key, pos_query, delta_to_local, position_offset,
                                      mask, scale_log2, version or VERSION, lengths, radius, cu_seqlens, max_seqlen)


# each row's real length (last kept token + 1), computed on the GPU so no host sync is needed
def real_lengths(mask):
    positions = torch.arange(1, mask.shape[1] + 1, device=mask.device, dtype=torch.int32)
    return (mask.to(torch.int32) * positions).amax(dim=1).to(torch.int32).contiguous()


_radius = {}


# the largest r with slot(d) == slot(0) + d for every |d| <= r, read from the plan's own table, so the kernel's
# arithmetic slots are exact by construction; one host copy per prepared plan
def linear_radius(plan):
    key = (plan.delta_to_local.data_ptr(), plan.position_offset)
    if key not in _radius:
        table, offset = plan.delta_to_local.cpu(), plan.position_offset
        radius = 0
        while (offset - radius - 1 >= 0 and offset + radius + 1 < table.numel()
               and int(table[offset + radius + 1]) - int(table[offset]) == radius + 1
               and int(table[offset]) - int(table[offset - radius - 1]) == radius + 1):
            radius += 1
        _radius[key] = radius
    return _radius[key]


def _ours(module, hidden_states, plan):
    has_relative = module.relative_attention and {"c2p", "p2c"} <= set(module.pos_att_type)
    return (module.backend != "torch" and hidden_states.dtype == torch.float16 and module.attention_head_size == 64
            and has_relative and plan.pos_key is not None and plan.pos_query is not None)


def install():
    """Route disentangled_flash's fp16 inference attention (head dim 64), padded and packed, through this kernel.

    Returns the original (forward_prepared, forward_packed) so a caller can restore them."""
    from disentangled_flash import kernel
    from disentangled_flash.packed import resolve_packed_info

    cls = kernel.InferenceDisentangledSelfAttention
    original_prepared, original_packed = cls.forward_prepared, cls.forward_packed

    def forward_prepared(self, hidden_states, attention_mask, plan):
        if not _ours(self, hidden_states, plan):
            return original_prepared(self, hidden_states, attention_mask, plan)
        batch_size, sequence_length = hidden_states.shape[:2]
        query, key, value = self._project_qkv(hidden_states)
        mask = None if self.assume_unpadded else attention_mask.to(torch.uint8).contiguous()
        lengths = real_lengths(mask) if mask is not None and VERSION >= 4 else None
        output = attention(self._reshape_heads(query, batch_size, sequence_length),
                           self._reshape_heads(key, batch_size, sequence_length),
                           self._reshape_heads(value, batch_size, sequence_length),
                           plan.pos_key, plan.pos_query, plan.delta_to_local, plan.position_offset, mask, SCALE_LOG2,
                           lengths=lengths, radius=linear_radius(plan) if VERSION >= 4 else -1)
        return output, None

    def forward_packed(self, hidden_states, cu_seqlens, max_seqlen=None, *, rel_embeddings=None, packed_info=None):
        if self.backend == "torch" or VERSION < 4:
            return original_packed(self, hidden_states, cu_seqlens, max_seqlen, rel_embeddings=rel_embeddings,
                                   packed_info=packed_info)
        info = resolve_packed_info(cu_seqlens, hidden_states.size(0), max_seqlen, packed_info)
        if self._cached_qkv_weight is None or (
                self._cached_pos_key is None and self._cached_pos_query is None
                and self._get_cached_shape_plan(info.max_seqlen, hidden_states.device) is None):
            self.prepare_for_inference(rel_embeddings)
        # plans are kept per length; packed runs vary in their longest sequence, so prepare the 32-token bucket above it
        # (a plan for a longer length maps every shorter distance to the same position slot) instead of one per length
        plan = self.prepare_shape(-(-info.max_seqlen // 32) * 32, hidden_states.device)
        if not _ours(self, hidden_states, plan):
            return original_packed(self, hidden_states, cu_seqlens, max_seqlen, rel_embeddings=rel_embeddings,
                                   packed_info=packed_info)
        total = hidden_states.size(0)
        query, key, value = (tensor.view(total, self.num_attention_heads, 64).permute(1, 0, 2)
                             for tensor in self._project_qkv(hidden_states))
        output = attention(query, key, value, plan.pos_key, plan.pos_query, plan.delta_to_local, plan.position_offset,
                           None, SCALE_LOG2, radius=linear_radius(plan),
                           cu_seqlens=cu_seqlens.to(torch.int32).contiguous(), max_seqlen=info.max_seqlen)
        return output, None

    cls.forward_prepared, cls.forward_packed = forward_prepared, forward_packed
    # bit-identical but no measured pipeline gain (117.7 vs 120.0 chunks/s), so off unless asked for
    if os.getenv("GLINER_T4_FUSE_LAYERNORM", "0") == "1":
        fuse_add_layernorm()
    return original_prepared, original_packed


# DeBERTa's attention and feed-forward outputs end in LayerNorm(dense(x) + residual); the add is fused into the
# LayerNorm for fp16 inference with hidden size 768 (dropout is the identity in eval mode)
def fuse_add_layernorm():
    from transformers.models.deberta_v2 import modeling_deberta_v2 as deberta

    for cls in (deberta.DebertaV2SelfOutput, deberta.DebertaV2Output):
        original = cls.forward

        def forward(self, hidden_states, input_tensor, _original=original):
            norm = self.LayerNorm
            if (self.training or hidden_states.dtype != torch.float16 or not hidden_states.is_cuda
                    or norm.normalized_shape != (768,) or norm.weight is None):
                return _original(self, hidden_states, input_tensor)
            return ops.add_layernorm(self.dense(hidden_states).contiguous(), input_tensor.contiguous(), norm.weight,
                                     norm.bias, norm.eps)

        cls.forward = forward
