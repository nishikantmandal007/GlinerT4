# Where the host time goes in the production detection path at the shipping settings (two passes, exact count head,
# fused add + LayerNorm, one serving thread): Gliner2NER.detect_batch called as the detection queue calls it (each
# document cut into runs of RUN_SIZE chunks), timed and then under cProfile; plus the same chunks in one detect_batch
# call over the whole corpus, to size the per-run overhead. Prints the top functions by own time and by cumulative time.
#   python bench/queue_profile.py --corpus corpus.json --out results_dir [--standin]
import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--corpus", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--trials", type=int, default=2)
parser.add_argument("--standin", action="store_true", help="CPU smoke test of this script with a stand-in model")
parser.add_argument("--pipeline", action="store_true", help="with the one-thread CPU/GPU pipeline (DETECTION_PIPELINE)")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.update(DETECTION_FAST_COUNT="0", DETECTION_SPLIT_CACHE="1", DETECTION_SKIP_EVAL="1", DETECTION_STREAMS="1",
                  DETECTION_CITY_PASS="0", DETECTION_EXACT_COUNT="1", GLINER_T4_FUSE_LAYERNORM="1",
                  DETECTION_PIPELINE="1" if args.pipeline else "0")
if not args.standin:
    os.environ["DETECTION_ATTENTION_KERNEL"] = str(ROOT)
sys.path.insert(0, str(ROOT / "artifacts"))
sys.path.insert(0, str(ROOT / "serving"))
sys.path.insert(0, str(ROOT / "tests"))

import torch  # noqa: E402

from colab_pipeline import gliner2_ner as ner  # noqa: E402
from colab_pipeline.detection_queue import RUN_SIZE  # noqa: E402

DEVICE = "cpu" if args.standin else "cuda"
LABELS = ["Person", "Organization", "Street", "City"]
if args.standin:
    import gliner2_exact
    import gliner2_standin

    ner.loaded_checkpoints[("legacy", "cpu", None)] = gliner2_standin.build()
    gliner2_exact.install(count_step0=True, format_cache=False)
    if args.pipeline:
        import gliner2_pipelined

        ner.PIPELINE, ner.gliner2_pipelined = True, gliner2_pipelined

docs = json.loads(Path(args.corpus).read_text())
chunks = [chunk for doc in docs for chunk in doc["chunks"]]
detector = ner.Gliner2NER()


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def queue_runs():
    found = []
    for doc in docs:
        for start in range(0, len(doc["chunks"]), RUN_SIZE):
            found.extend(detector.detect_batch(doc["chunks"][start:start + RUN_SIZE], LABELS, device=DEVICE))
    return found


def whole_corpus():
    return detector.detect_batch(chunks, LABELS, device=DEVICE)


def timed(fn):
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    return time.perf_counter() - start, result


report = {"pipeline": args.pipeline, "run_size": RUN_SIZE, "chunks": len(chunks), "device": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu"}
for fn in (queue_runs, whole_corpus):  # warm-up: kernel plans for every length either shape produces
    fn()
for name, fn in (("queue_runs", queue_runs), ("whole_corpus", whole_corpus)):
    seconds = [timed(fn)[0] for _ in range(args.trials)]
    report[name] = {"seconds": seconds, "chunks_per_s": len(chunks) / min(seconds)}
    print("TIMING " + json.dumps({name: report[name]}), flush=True)

profile = cProfile.Profile()
sync()
profile.enable()
queue_runs()
sync()
profile.disable()
text = io.StringIO()
stats = pstats.Stats(profile, stream=text)
stats.sort_stats("tottime").print_stats(45)
stats.sort_stats("cumulative").print_stats(45)
(out / "queue_profile.txt").write_text(text.getvalue())
(out / "queue_profile.json").write_text(json.dumps(report, indent=2))
print(text.getvalue(), flush=True)
print("QUEUE_PROFILE_DONE", flush=True)
