# Encoder with the sm_75 kernel vs disentangled_flash (and both vs an fp32 reference), on real text, then timing.
#   python tests/test_attention.py [lengths...]
import json
import sys
from pathlib import Path

import torch
from gliner2 import AutoExtractor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
import gliner_t4  # noqa: E402
from disentangled_flash import kernel  # noqa: E402

NAME = "fastino/gliner2-multi-v1"
LENGTHS = [int(value) for value in sys.argv[1:]] or [32, 96, 200, 256, 384, 512, 768, 1024]

fused = AutoExtractor.from_pretrained(NAME, map_location="cuda", quantize=True, attention_backend="disentangled_flash")
reference = AutoExtractor.from_pretrained(NAME, map_location="cuda")  # fp32, standard attention
corpus = json.load(open("/home/soumik/nishikant/detbench/corpus.json"))
tokens = fused.processor.tokenizer(" ".join(c for d in corpus for c in d["chunks"]), add_special_tokens=False)["input_ids"]
cls = kernel.InferenceDisentangledSelfAttention
originals = cls.forward_prepared, cls.forward_packed


def restore():
    cls.forward_prepared, cls.forward_packed = originals


def batch(length, padded):
    ids = torch.tensor([tokens[row * length:(row + 1) * length] for row in range(16)], device="cuda")
    mask = torch.ones_like(ids)
    if padded:  # rows end at different lengths, as a length-sorted run of real chunks does
        for row in range(16):
            mask[row, max(1, length - row * length // 20):] = 0
    return ids, mask


def run(model, ids, mask):
    with torch.inference_mode():
        return model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state.float()


def gpu_ms(model, ids, mask, reps=10, rounds=5):
    times = []
    with torch.inference_mode():
        for _ in range(3):
            model.encoder(input_ids=ids, attention_mask=mask)
        for _ in range(rounds):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):
                model.encoder(input_ids=ids, attention_mask=mask)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / reps)
    return sorted(times)[rounds // 2]


results = []
for length in LENGTHS:
    for padded in (False, True):
        ids, mask = batch(length, padded)
        keep = mask.bool()[..., None]
        expected = run(reference, ids, mask)
        restore()
        flash = run(fused, ids, mask)
        flash_ms = gpu_ms(fused, ids, mask)
        for plans in fused.encoder.encoder._prepared_plans.values():  # padded plans are keyed by L, packed by max length
            table = plans[0].delta_to_local
            assert bool(torch.all(table[1:] >= table[:-1])), "slot map not monotone"
        gliner_t4.install()
        before = dict(fused._disentangled_flash_batches)
        ours = run(fused, ids, mask)
        path = "packed" if fused._disentangled_flash_batches["packed"] > before["packed"] else "padded"
        ours_ms = gpu_ms(fused, ids, mask)
        restore()

        def dist(a, b):  # only real tokens matter downstream
            d = (a - b).abs().masked_select(keep)
            return round(float(d.mean()), 5), round(float(d.max()), 4)

        row = {"length": length, "padded": padded, "path": path, "flash_ms": round(flash_ms, 2), "ours_ms": round(ours_ms, 2),
               "speedup": round(flash_ms / ours_ms, 3), "ours_vs_flash": dist(ours, flash),
               "flash_vs_fp32": dist(flash, expected), "ours_vs_fp32": dist(ours, expected)}
        results.append(row)
        print(json.dumps(row), flush=True)
json.dump(results, open(Path(__file__).resolve().parents[1] / "artifacts" / "test_attention.json", "w"), indent=1)
