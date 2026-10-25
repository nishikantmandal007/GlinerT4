"""Profile the installed flash encoder on one visible GPU."""
import json
from pathlib import Path
import torch
from gliner2 import AutoExtractor

torch.set_num_threads(4)
torch.manual_seed(0)
model = AutoExtractor.from_pretrained('fastino/gliner2-multi-v1', map_location='cuda', quantize=True, attention_backend='disentangled_flash').eval()
print(type(model.encoder), model.encoder.config, flush=True)
ids = torch.randint(5, 200000, (16, 384), device='cuda')
mask = torch.ones_like(ids)
with torch.inference_mode():
    for _ in range(8):
        model.encoder(input_ids=ids, attention_mask=mask)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            model.encoder(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35))
prof.export_chrome_trace('artifacts/profile_baseline.json')
rows = [{'name': e.key, 'count': e.count, 'self_device_us': e.self_device_time_total, 'self_cpu_us': e.self_cpu_time_total} for e in prof.key_averages()]
Path('artifacts/profile_baseline_summary.json').write_text(json.dumps(rows, indent=2))
