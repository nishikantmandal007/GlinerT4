# Same GLiNER2 benchmark on any GPU (T4 VM, Colab L4 / A100): encoder ms per backend, and end-to-end chunks/s with
# ai-utils' grouped passes (Organization | Person+Street+City | City alone), batch 16, length-sorted, fp16.
#   python gpu_compare.py [corpus.json]   (the Colab copy has the corpus embedded; see make_colab_script.py)
import json
import os
import subprocess
import sys
import time


def ensure_packages():
    try:
        import gliner2, disentangled_flash  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gliner2[local]==2.1.0",
                               "disentangled-flash==1.1.1"])


ensure_packages()
import torch  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402

NAME = "fastino/gliner2-multi-v1"
LABELS = {
    "Person": "First and last names of people. NOT company names, street addresses, cities, or identifier codes.",
    "Organization": "Company, corporation, institution, agency, or legal entity name. "
                    "NOT person names, street addresses, cities, or identifier codes.",
    "Street": "Street, road, avenue, boulevard, or lane name. NOT city names, building names, or postal codes.",
    "City": "City",
}
PASSES = [{"Organization": LABELS["Organization"]},
          {key: LABELS[key] for key in ("Person", "Street", "City")},
          {"City": "Name of a city, town or village."}]


def load_corpus():
    embedded = globals().get("EMBEDDED_CORPUS")
    docs = json.loads(embedded) if embedded else json.load(open(sys.argv[1]))
    return [chunk for doc in docs for chunk in doc["chunks"]]


def encoder_ms(model, ids, reps=10, rounds=5):
    mask = torch.ones_like(ids)
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
    return round(sorted(times)[rounds // 2], 2)


def end_to_end(model, chunks):
    ordered = sorted(chunks, key=len)
    with torch.inference_mode():
        for entity_types in PASSES:  # warm-up
            model.batch_extract_entities(ordered[:32], entity_types, batch_size=16, threshold=0.65)
        torch.cuda.synchronize()
        start = time.time()
        found = 0
        for entity_types in PASSES:
            results = model.batch_extract_entities(ordered, entity_types, batch_size=16, threshold=0.65)
            found += sum(len(values) for result in results for values in result.get("entities", {}).values())
        torch.cuda.synchronize()
    return round(len(chunks) / (time.time() - start), 1), found


chunks = load_corpus() * 2
report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
          "capability": ".".join(map(str, torch.cuda.get_device_capability())), "chunks": len(chunks)}
tokens = None
for backend, options in (("standard", {}), ("disentangled_flash", {"attention_backend": "disentangled_flash"})):
    model = AutoExtractor.from_pretrained(NAME, map_location="cuda", quantize=True, **options)
    if tokens is None:
        tokens = model.processor.tokenizer(" ".join(chunks), add_special_tokens=False)["input_ids"]
    for length in (256, 384, 512):
        ids = torch.tensor([tokens[row * length:(row + 1) * length] for row in range(16)], device="cuda")
        report[f"{backend}_encoder_16x{length}_ms"] = encoder_ms(model, ids)
    report[f"{backend}_chunks_per_s"], report[f"{backend}_entities_found"] = end_to_end(model, chunks)
    report[f"{backend}_peak_mb"] = torch.cuda.max_memory_allocated() // 2 ** 20
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
print("RESULT " + json.dumps(report))
