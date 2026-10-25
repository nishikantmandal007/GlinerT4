# encoder-only time per run of 16 texts: fused DisentangledFlash vs TensorRT engine vs plain fp16 PyTorch
import os, sys, time, torch
sys.path.insert(0, os.getcwd())
from gliner2 import AutoExtractor
from app.models.trt_encoder import TensorRTEncoder
name = "fastino/gliner2-multi-v1"
fused = AutoExtractor.from_pretrained(name, map_location="cuda", quantize=True, attention_backend="disentangled_flash")
plain = AutoExtractor.from_pretrained(name, map_location="cuda", quantize=True)
engine = TensorRTEncoder(sys.argv[1], "cuda")
def timed(encoder, ids, mask, reps=20):
    with torch.inference_mode():
        for _ in range(3): encoder(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize(); start = time.time()
        for _ in range(reps): out = encoder(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize()
    return (time.time() - start) / reps * 1000, out.last_hidden_state
for length in (256, 384, 512):
    ids = torch.randint(5, 200000, (16, length), device="cuda"); mask = torch.ones_like(ids)
    tf, of = timed(fused.encoder, ids, mask); tp, op = timed(plain.encoder, ids, mask); te, oe = timed(engine, ids, mask)
    print(f"16x{length}: fused {tf:.1f}ms  plain fp16 {tp:.1f}ms  tensorrt {te:.1f}ms  max|trt-fused| {(oe.float()-of.float()).abs().max():.3f}")
