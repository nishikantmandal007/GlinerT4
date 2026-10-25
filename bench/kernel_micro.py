# Attention op alone at the encoder's shape (16 x 12 heads x L x 64), per build variant, to see where its time goes.
#   python bench/kernel_micro.py L variant[,variant...]   variants: base, SKIP_CQ, SKIP_PK, SKIP_GATHER (combine with +)
import sys
from pathlib import Path

import torch
from gliner2 import AutoExtractor
from torch.utils.cpp_extension import load

CSRC = Path(__file__).resolve().parents[1] / "csrc"
length = int(sys.argv[1])
variants = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base"]
versions = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2]

# a real prepared plan (slot map, compact position tables) for this length, from layer 0
model = AutoExtractor.from_pretrained("fastino/gliner2-multi-v1", map_location="cuda", quantize=True,
                                      attention_backend="disentangled_flash")
with torch.inference_mode():
    model.encoder(input_ids=torch.ones((1, length), dtype=torch.long, device="cuda"),
                  attention_mask=torch.ones((1, length), dtype=torch.long, device="cuda"))
plan = model.encoder.encoder._prepared_plans[length][0]
qkv = torch.randn(16, length, 3, 12, 64, device="cuda", dtype=torch.float16)
q, k, v = (qkv[:, :, index].transpose(1, 2) for index in range(3))  # [B, H, L, 64] views, like the fused projection
scale = (64 * 3) ** -0.5 * 1.4426950408889634

ops = {}
for variant in variants:
    flags = [f"-D{flag}" for flag in variant.split("+") if flag != "base"]
    namespace = "gliner_t4_" + (variant.replace("+", "_").lower() if flags else "base")
    load(name=namespace, sources=[str(CSRC / "torch_binding.cpp"), str(CSRC / "disentangled_attention.cu"),
                                   str(CSRC / "disentangled_attention_v2.cu"),
                                   str(CSRC / "disentangled_attention_v3.cu"),
                                   str(CSRC / "disentangled_attention_v4.cu")],
         extra_cuda_cflags=["-O3", "-gencode=arch=compute_75,code=sm_75", "--use_fast_math", f"-DGLINER_T4_NS={namespace}",
                            *flags],
         extra_cflags=["-O3", f"-DGLINER_T4_NS={namespace}"], is_python_module=False)
    for version in versions:
        ops[(variant, version)] = getattr(torch.ops, namespace).disentangled_attention

# rounds alternate between all configurations, so a T4's clock drift (it throttles with heat) hits each one equally
times = {key: [] for key in ops}
for _ in range(7):
    for (variant, version), op in ops.items():
        run = lambda: op(q, k, v, plan.pos_key, plan.pos_query, plan.delta_to_local, plan.position_offset, None, scale,
                         version)
        for _ in range(3):
            run()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            run()
        end.record()
        torch.cuda.synchronize()
        times[(variant, version)].append(start.elapsed_time(end) / 20)
for (variant, version), values in times.items():
    print(f"L={length} v{version} {variant:40} {sorted(values)[3]:.3f} ms per layer (min {min(values):.3f})", flush=True)
