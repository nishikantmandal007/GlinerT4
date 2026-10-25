# Encoder-only time per run of 16 texts for ready-made backends, and each one's distance to an fp32 reference.
#   python bench/encoder_variants.py default autotune flashdeberta
import json
import sys

import torch
from gliner2 import AutoExtractor

NAME = "fastino/gliner2-multi-v1"
VARIANTS = {
    "default": {"attention_backend": "disentangled_flash"},
    "autotune": {"attention_backend": "disentangled_flash", "disentangled_flash_tuning": "autotune"},
    "flashdeberta": {"attention_backend": "flashdeberta"},
    "standard": {},
}
LENGTHS = (256, 384, 512)


# median of 5 rounds: a T4 is passively cooled and its clock drifts with temperature
def timed(encoder, ids, mask, reps=10, rounds=5):
    rounds_ms = []
    with torch.inference_mode():
        for _ in range(5):
            encoder(input_ids=ids, attention_mask=mask)
        for _ in range(rounds):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):
                out = encoder(input_ids=ids, attention_mask=mask)
            end.record()
            torch.cuda.synchronize()
            rounds_ms.append(start.elapsed_time(end) / reps)
    return sorted(rounds_ms)[rounds // 2], out.last_hidden_state.float()


reference = AutoExtractor.from_pretrained(NAME, map_location="cuda")  # fp32, standard attention
# real text: the corpus chunks joined and cut into 16 rows of exactly L tokens
corpus = json.load(open("/home/soumik/nishikant/detbench/corpus.json"))
tokens = reference.processor.tokenizer(" ".join(chunk for doc in corpus for chunk in doc["chunks"]),
                                       add_special_tokens=False)["input_ids"]
inputs = {length: torch.tensor([tokens[row * length:(row + 1) * length] for row in range(16)], device="cuda")
          for length in LENGTHS}
with torch.inference_mode():
    expected = {length: reference.encoder(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state.float()
                for length, ids in inputs.items()}
del reference
torch.cuda.empty_cache()

results = {}
for variant in sys.argv[1:] or ["default"]:
    model = AutoExtractor.from_pretrained(NAME, map_location="cuda", quantize=True, **VARIANTS[variant])
    for length, ids in inputs.items():
        ms, out = timed(model.encoder, ids, torch.ones_like(ids))
        diff = (out - expected[length]).abs()
        results[f"{variant}/{length}"] = {"ms": round(ms, 2), "mean_diff": float(diff.mean()), "max_diff": float(diff.max())}
        print(f"{variant:13} 16x{length}: {ms:7.2f} ms  |diff vs fp32| mean {diff.mean():.5f} max {diff.max():.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
print(json.dumps(results))
