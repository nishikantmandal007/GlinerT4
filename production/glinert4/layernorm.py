import torch

from .ops import ops

_installed = {}


def install():
    from transformers.models.deberta_v2 import modeling_deberta_v2 as deberta

    for cls in (deberta.DebertaV2SelfOutput, deberta.DebertaV2Output):
        if cls in _installed:
            continue
        stock = cls.forward

        def forward(self, hidden_states, input_tensor, _stock=stock):
            norm = self.LayerNorm
            if (self.training or hidden_states.dtype != torch.float16 or not hidden_states.is_cuda
                    or norm.normalized_shape != (768,) or norm.weight is None):
                return _stock(self, hidden_states, input_tensor)
            return ops().add_layernorm(self.dense(hidden_states).contiguous(), input_tensor.contiguous(), norm.weight,
                                       norm.bias, norm.eps)

        cls.forward = forward
        _installed[cls] = stock


def uninstall():
    for cls, stock in _installed.items():
        cls.forward = stock
    _installed.clear()
