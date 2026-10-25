# Host-side cost of one fused encoder run: syncs (item / device-to-host copies) and CPU time, vs GPU time.
import sys, time, torch, collections
from gliner2 import AutoExtractor
model = AutoExtractor.from_pretrained("fastino/gliner2-multi-v1", map_location="cuda", quantize=True, attention_backend="disentangled_flash")
encoder, length = model.encoder, int(sys.argv[1]) if len(sys.argv) > 1 else 512
ids = torch.randint(5, 200000, (16, length), device="cuda"); mask = torch.ones_like(ids)
with torch.inference_mode():
    for _ in range(3): encoder(input_ids=ids, attention_mask=mask)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_stack=True) as prof:
        encoder(input_ids=ids, attention_mask=mask); torch.cuda.synchronize()
counts = collections.Counter(e.name for e in prof.events() if e.name in ("aten::item", "aten::_local_scalar_dense", "cudaMemcpyAsync", "cudaStreamSynchronize", "cudaLaunchKernel", "cudaDeviceSynchronize"))
print("per run:", dict(counts))
stacks = collections.Counter()
for e in prof.events():
    if e.name == "aten::item":
        frames = [f for f in (e.stack or []) if "site-packages" in f and "torch/" not in f]
        stacks[frames[0] if frames else "?"] += 1
for frame, n in stacks.most_common(8): print(f"  {n:3d} item() at {frame}")
with torch.inference_mode():
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10): encoder(input_ids=ids, attention_mask=mask)
    torch.cuda.synchronize(); print(f"wall {((time.time()-t)/10)*1000:.1f} ms per run")
