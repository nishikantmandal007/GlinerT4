# Where the fused encoder's GPU time goes, per kernel family, for one 16 x L run.
#   python bench/profile_kernels.py [L ...]
import collections
import re
import sys

import os
from pathlib import Path

import torch
from gliner2 import AutoExtractor

if os.getenv("GLINER_T4"):  # profile with the sm_75 kernel installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
    import gliner_t4
    gliner_t4.install()

FAMILIES = [("attention", r"deberta|attention"), ("gemm", r"gemm|cutlass|s1688|h1688|turing|volta|sm75|xmma"),
            ("layernorm", r"layer_norm|LayerNorm"), ("gelu", r"gelu|GeluCUDA"), ("softmax", r"softmax"),
            ("copy/cast", r"copy|cast|to_copy"), ("elementwise", r"elementwise|vectorized|unrolled|add|mul"),
            ("reduce", r"reduce")]

model = AutoExtractor.from_pretrained("fastino/gliner2-multi-v1", map_location="cuda", quantize=True,
                                      attention_backend="disentangled_flash")
encoder = model.encoder
for length in [int(value) for value in sys.argv[1:]] or [512]:
    ids = torch.randint(5, 200000, (16, length), device="cuda")
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        for _ in range(3):
            encoder(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
            for _ in range(5):
                encoder(input_ids=ids, attention_mask=mask)
            torch.cuda.synchronize()
    by_family, by_kernel, launches = collections.Counter(), collections.Counter(), 0
    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        launches += 1
        family = next((name for name, pattern in FAMILIES if re.search(pattern, event.name, re.I)), "other")
        by_family[family] += event.device_time_total / 5
        by_kernel[event.name[:110]] += event.device_time_total / 5
    total = sum(by_family.values())
    print(f"16x{length}: GPU kernel time {total / 1000:.1f} ms per run, {launches // 5} kernels per run")
    for family, us in by_family.most_common():
        print(f"  {family:12} {us / 1000:7.2f} ms {us / total:6.1%}")
    for name, us in by_kernel.most_common(12):
        print(f"    {us / 1000:7.2f} ms  {name}")
