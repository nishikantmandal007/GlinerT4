# Per-layer attention cost at the encoder's shapes, to size the custom kernel's prize:
#   disentangled_flash attention kernel alone (from a profiled encoder run, / 12 layers)
#   vs plain fused attention with no relative terms (PyTorch SDPA, CUTLASS memory-efficient kernel, T4 tensor cores)
#   vs the c2p/p2c table GEMMs disentangled_flash runs before its kernel.
import re
import sys

import torch
import torch.nn.functional as F
from gliner2 import AutoExtractor

B, H, D = 16, 12, 64
model = AutoExtractor.from_pretrained("fastino/gliner2-multi-v1", map_location="cuda", quantize=True,
                                      attention_backend="disentangled_flash")


def gpu_ms(fn, reps=20):
    for _ in range(5):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


for L in [int(value) for value in sys.argv[1:]] or [256, 512]:
    q, k, v = (torch.randn(B, H, L, D, device="cuda", dtype=torch.float16) for _ in range(3))
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
        plain = gpu_ms(lambda: F.scaled_dot_product_attention(q, k, v))
    rows, table = torch.randn(H, B * L, D, device="cuda", dtype=torch.float16), torch.randn(H, 255, D, device="cuda", dtype=torch.float16)
    tables = gpu_ms(lambda: (torch.bmm(rows, table.transpose(1, 2)), torch.bmm(rows, table.transpose(1, 2))))
    ids = torch.randint(5, 200000, (B, L), device="cuda")
    with torch.inference_mode():
        for _ in range(3):
            model.encoder(input_ids=ids, attention_mask=torch.ones_like(ids))
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
            for _ in range(5):
                model.encoder(input_ids=ids, attention_mask=torch.ones_like(ids))
            torch.cuda.synchronize()
    fused = sum(e.device_time_total for e in profiler.events()
                if e.device_type == torch.autograd.DeviceType.CUDA and re.search("deberta_attention", e.name)) / 5 / 12 / 1000
    flops = 4 * B * H * L * L * D / 1e9
    print(f"L={L}: per layer  disentangled kernel {fused:.2f} ms ({flops / fused:.1f} TFLOP/s on QK+PV)   "
          f"plain SDPA {plain:.2f} ms ({flops / plain:.1f} TFLOP/s)   c2p+p2c table GEMMs {tables:.2f} ms", flush=True)
