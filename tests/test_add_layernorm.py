# The fused add + LayerNorm must be bit-identical to PyTorch's add followed by layer_norm (torch.equal, not allclose),
# on random rows and on the real inputs of every DeBERTa output block during an encoder run.
#   python tests/test_add_layernorm.py
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from gliner2 import AutoExtractor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
import gliner_t4  # noqa: E402

ops = gliner_t4.ops


def check(name, x, residual, weight, bias, eps):
    expected = F.layer_norm(x + residual, (768,), weight, bias, eps)
    got = ops.add_layernorm(x.contiguous(), residual.contiguous(), weight, bias, eps)
    differing = (expected != got).sum().item()
    max_diff = (expected.float() - got.float()).abs().max().item()
    print(f"{name:40} rows {x.numel() // 768:6}  differing values {differing}  max|diff| {max_diff}")
    return differing == 0


ok = True
torch.manual_seed(0)
weight, bias = torch.randn(768, device="cuda", dtype=torch.float16), torch.randn(768, device="cuda", dtype=torch.float16)
for scale in (0.1, 1.0, 10.0):
    x, r = (torch.randn(4096, 768, device="cuda", dtype=torch.float16) * scale for _ in range(2))
    ok &= check(f"random, scale {scale}", x, r, weight, bias, 1e-7)

# real inputs: capture (dense output, residual) at both output blocks of every layer, padded and packed runs
from transformers.models.deberta_v2 import modeling_deberta_v2 as deberta  # noqa: E402

captured = []
for cls in (deberta.DebertaV2SelfOutput, deberta.DebertaV2Output):
    original = cls.forward

    def capture(self, hidden_states, input_tensor, _original=original):
        captured.append((self.dense(hidden_states).detach(), input_tensor.detach(), self.LayerNorm))
        return _original(self, hidden_states, input_tensor)

    cls.forward = capture
model = AutoExtractor.from_pretrained("fastino/gliner2-multi-v1", map_location="cuda", quantize=True,
                                      attention_backend="disentangled_flash")
chunks = [c for d in json.load(open("/home/soumik/nishikant/detbench/corpus.json")) for c in d["chunks"]][:48]
with torch.inference_mode():
    model.batch_extract_entities(chunks, {"Person": "Person"}, batch_size=16)
for i, (dense, residual, norm) in enumerate(captured):
    ok &= check(f"real activation {i} ({tuple(dense.shape)})", dense, residual, norm.weight, norm.bias, norm.eps)
print("BIT-IDENTICAL" if ok else "NOT IDENTICAL")
