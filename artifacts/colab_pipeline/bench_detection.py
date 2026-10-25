# Detection throughput and accuracy on one corpus, in two serving shapes:
#   copies: N separate processes per GPU, each an fp32 GLiNER2 answering one text and one label per call
#           (today's dev code behind nginx replicas)
#   queue:  this branch's shared detection queue, batched runs with grouped label passes
# Usage:
#   python scripts/bench_detection.py run corpus.json --mode copies --copies 10,5 --out copies.json
#   python scripts/bench_detection.py run corpus.json --mode queue --out queue.json  (DETECTION_* env as in serving)
#   python scripts/bench_detection.py compare a.json b.json
import argparse
import json
import multiprocessing
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABELS = ["Person", "Organization", "Street", "City"]
# the thresholds dev's Gliner2NER uses per label
COPY_THRESHOLDS = {"Organization": 0.85}


def gpu_sampler(stop, samples):
    while not stop.is_set():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        samples.append((time.time(), [[int(value) for value in line.split(",")] for line in out.strip().splitlines()]))
        time.sleep(0.5)


def copy_worker(gpu, texts, loading, ready, start, results):
    import torch
    from gliner2 import GLiNER2
    from app.models.gliner2_ner import _labels_with_desc
    torch.cuda.set_device(gpu)
    # one copy loads at a time: fifteen at once contend for the Hugging Face cache lock and one can stall for good
    with loading:
        model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1", map_location=f"cuda:{gpu}")
    model.extract_entities(texts[0][1], _labels_with_desc(["Person"]), include_confidence=True, threshold=0.65)
    ready.wait(timeout=1200)
    start.wait(timeout=60)
    found = []
    for index, text in texts:
        values = {}
        for label in LABELS:
            result = model.extract_entities(text, _labels_with_desc([label]), include_confidence=True,
                                            threshold=COPY_THRESHOLDS.get(label, 0.65))
            kept = [entity["text"] for entity in result.get("entities", {}).get(label, [])]
            values[label] = [re.sub(r"\b\d+\b", "", value).strip() for value in kept] if label == "Street" else kept
        found.append((index, values))
    results.put((found, time.time(), torch.cuda.max_memory_allocated(gpu) // 2**20))


def run_copies(chunks, copies):
    context = multiprocessing.get_context("spawn")
    workers = [gpu for gpu, count in enumerate(copies) for _ in range(count)]
    loading, results = context.Lock(), context.Queue()
    ready, start = context.Barrier(len(workers) + 1), context.Barrier(len(workers) + 1)
    # round robin, as nginx spreads single-chunk requests over the replicas
    shares = [[(index, chunks[index]) for index in range(slot, len(chunks), len(workers))] for slot in range(len(workers))]
    processes = [context.Process(target=copy_worker, args=(gpu, share, loading, ready, start, results))
                 for gpu, share in zip(workers, shares)]
    for process in processes:
        process.start()
    ready.wait(timeout=1200)
    began = time.time()
    start.wait(timeout=60)
    outputs = [results.get() for _ in processes]
    for process in processes:
        process.join()
    found = [None] * len(chunks)
    for part, _, _ in outputs:
        for index, values in part:
            found[index] = values
    return found, max(end for _, end, _ in outputs) - began, {"peak_mb_per_copy": max(mb for _, _, mb in outputs)}


def run_queue(docs):
    from app.models.detection_queue import STREAMS, WARM_UP_TEXTS, detection_devices
    from app.models.gliner2_ner import detection_queue
    shared = detection_queue()
    for _ in range(len(detection_devices()) * STREAMS):
        # a serving thread that dies before its warm-up never releases: fail instead of waiting forever
        if not shared.ready.acquire(timeout=900):
            raise RuntimeError("a detection serving thread did not become ready; see its log")
    shared.detect(WARM_UP_TEXTS * 8, LABELS)
    # every document is one request and all arrive at once, as ai-safe sends a batch of files
    began = time.time()
    with ThreadPoolExecutor(len(docs)) as pool:
        per_doc = list(pool.map(lambda doc: shared.detect(doc["chunks"], LABELS), docs))
    seconds = time.time() - began
    return [found["values"] for doc in per_doc for found in doc], seconds, {}


def norm(value):
    return " ".join(re.sub(r"\b\d+\b", " ", value).lower().split())


def score(docs, found):
    totals, offset = {label: [0, 0, 0, 0] for label in LABELS}, 0  # truth hit, truth, detected hit, detected
    for doc in docs:
        values = {label: set() for label in LABELS}
        for chunk_values in found[offset:offset + len(doc["chunks"])]:
            for label in LABELS:
                values[label].update(value.strip() for value in chunk_values.get(label, []) if value.strip())
        offset += len(doc["chunks"])
        for label in LABELS:
            truth = {norm(value) for value in doc["truth"].get(label, [])}
            detected = {norm(value) for value in values[label]} - {""}
            # a street's truth is the whole address and ai-utils drops house numbers, so it matches when one holds the other
            same = (lambda a, b: a in b or b in a) if label == "Street" else str.__eq__
            totals[label][0] += sum(any(same(value, found) for found in detected) for value in truth)
            totals[label][1] += len(truth)
            totals[label][2] += sum(any(same(value, found) for found in truth) for value in detected)
            totals[label][3] += len(detected)
    return {label: {"recall": round(hit / max(1, n), 3), "precision": round(dhit / max(1, d), 3), "detected": d}
            for label, (hit, n, dhit, d) in totals.items()}


def run(args):
    docs = json.load(open(args.corpus)) * args.repeat
    chunks = [chunk for doc in docs for chunk in doc["chunks"]]
    stop, samples = threading.Event(), []
    sampler = threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True)
    sampler.start()
    if args.mode == "copies":
        found, seconds, extra = run_copies(chunks, [int(count) for count in args.copies.split(",")])
    else:
        found, seconds, extra = run_queue(docs)
    stop.set()
    serving = [sample for at, sample in samples if at >= time.time() - seconds - 1] or [sample for _, sample in samples]
    samples = [sample for _, sample in samples]
    busy = [sample for sample in samples if any(gpu[1] > 0 for gpu in sample)] or samples
    report = {
        "mode": args.mode, "copies": args.copies if args.mode == "copies" else None,
        "env": {key: value for key, value in os.environ.items() if key.startswith("DETECTION_")},
        "chunks": len(chunks), "seconds": round(seconds, 2), "chunks_per_s": round(len(chunks) / seconds, 1),
        "peak_gpu_mb": [max(sample[gpu][0] for sample in samples) for gpu in range(len(samples[0]))],
        # while serving only: the lifetime peak above includes model loading
        "serving_gpu_mb": [max(sample[gpu][0] for sample in serving) for gpu in range(len(serving[0]))],
        "mean_gpu_util": [round(sum(sample[gpu][1] for sample in busy) / len(busy)) for gpu in range(len(busy[0]))],
        **extra, "accuracy": score(docs, found), "found": found,
    }
    json.dump(report, open(args.out, "w"), ensure_ascii=False)
    print(json.dumps({key: value for key, value in report.items() if key != "found"}, indent=1))


# per chunk and label, how often two runs return the same set of values
def compare(args):
    first, second = (json.load(open(path)) for path in (args.first, args.second))
    for label in LABELS:
        pairs = list(zip(first["found"], second["found"]))
        same = sum(set(a.get(label, [])) == set(b.get(label, [])) for a, b in pairs)
        print(f"{label:13} identical {same}/{len(pairs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("corpus")
    runner.add_argument("--mode", choices=["copies", "queue"], required=True)
    runner.add_argument("--copies", default="10,5", help="processes per GPU, in GPU order")
    runner.add_argument("--repeat", type=int, default=1)
    runner.add_argument("--out", required=True)
    runner.set_defaults(handler=run)
    comparer = commands.add_parser("compare")
    comparer.add_argument("first")
    comparer.add_argument("second")
    comparer.set_defaults(handler=compare)
    arguments = parser.parse_args()
    arguments.handler(arguments)
