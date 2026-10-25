# Label-pass layouts for GLiNER2 detection on one T4: speed and ground-truth accuracy of each layout, through the
# production detection code (artifacts/colab_pipeline/gliner2_ner.py: Gliner2NER.detect_batch with its thresholds,
# cross-pass City union, table-cell confirmation and span resolution) scored as bench_detection.py scores it.
#
# Each pass encodes the whole chunk again together with its labels, so the encoder work per chunk is proportional to
# the number of passes. Today's layout is three passes; a two-pass layout is about a third less work. A layout is
# accepted only if no label's recall or precision falls below today's layout on this corpus. The corpus is 10
# documents, so a difference of one value is visible; confirm an accepted layout on a held-out corpus (the 25-language
# set) before shipping it.
#   python bench/pass_bench.py --corpus corpus.json --out results_dir [--trials 2] [--standin]
import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--corpus", required=True, help="corpus.json: documents with 'chunks' and 'truth'")
parser.add_argument("--out", required=True)
parser.add_argument("--trials", type=int, default=2)
parser.add_argument("--layouts", default="", help="comma-separated layout names; default: all")
parser.add_argument("--standin", action="store_true", help="CPU smoke test of this script with a stand-in model")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# the production serving settings, without the non-exact fast count; the sm_75 kernel from this checkout
os.environ.update(DETECTION_FAST_COUNT="0", DETECTION_SPLIT_CACHE="1", DETECTION_SKIP_EVAL="1",
                  DETECTION_SHARE_WEIGHTS="1", DETECTION_STREAMS="1", DETECTION_LENGTH_BUCKET="32")
if not args.standin:
    os.environ["DETECTION_ATTENTION_KERNEL"] = str(ROOT)
sys.path.insert(0, str(ROOT / "artifacts"))
sys.path.insert(0, str(ROOT / "serving"))
sys.path.insert(0, str(ROOT / "tests"))

import torch  # noqa: E402

from colab_pipeline import bench_detection  # noqa: E402
from colab_pipeline import gliner2_ner as ner  # noqa: E402
import gliner2_exact  # noqa: E402

DEVICE = "cpu" if args.standin else "cuda"
CITY_ALONE = ner.CITY_ALONE_DESCRIPTION
LABELS = bench_detection.LABELS  # Person, Organization, Street, City

# name: the passes, each (labels, description overrides); every layout answers all four labels
LAYOUTS = {
    # today: Organization alone, the name-like labels together, City again alone (its union raises City recall)
    "current_3pass": [(["Organization"], {}), (["Person", "Street", "City"], {}), (["City"], {"City": CITY_ALONE})],
    # drop the City-alone pass
    "drop_city_pass": [(["Organization"], {}), (["Person", "Street", "City"], {})],
    # drop it, and give City the City-alone pass's description inside the shared pass
    "merged_city_desc": [(["Organization"], {}), (["Person", "Street", "City"], {"City": CITY_ALONE})],
    # move the City-alone query into the Organization pass: still two passes, City is still asked twice
    "city_with_org": [(["Organization", "City"], {"City": CITY_ALONE}), (["Person", "Street", "City"], {})],
    # as above with the description in both City queries
    "city_with_org_desc": [(["Organization", "City"], {"City": CITY_ALONE}),
                           (["Person", "Street", "City"], {"City": CITY_ALONE})],
}
if args.layouts:
    LAYOUTS = {k: LAYOUTS[k] for k in ["current_3pass"] + [n for n in args.layouts.split(",") if n != "current_3pass"]}

docs = json.loads(Path(args.corpus).read_text())
chunks = [chunk for doc in docs for chunk in doc["chunks"]]
report = {
    "corpus": {"documents": len(docs), "chunks": len(chunks),
               "sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()},
    "env": {"device": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu (stand-in)",
            "versions": {p: importlib.metadata.version(p) for p in ("torch", "gliner2", "transformers")}},
    "layouts": {name: [[labels, overrides] for labels, overrides in passes] for name, passes in LAYOUTS.items()},
}

if args.standin:
    import gliner2_standin

    ner.loaded_checkpoints[("legacy", "cpu", None)] = gliner2_standin.build()
# exact host-side cuts (outputs unchanged; tests/test_gliner2_exact.py), the same for every layout
gliner2_exact.install(count_step0=True, format_cache=False)
detector = ner.Gliner2NER()


def detect(layout):
    ner.strategy_passes = lambda labels: [("legacy", list(group), dict(overrides)) for group, overrides in LAYOUTS[layout]]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    found = detector.detect_batch(chunks, LABELS, device=DEVICE, batch_size=16)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start, [item["values"] for item in found]


results = {}
for name in LAYOUTS:  # warm-up (kernel plans, caches) and the accuracy run; detection is deterministic
    detect(name)
    _, found = detect(name)
    results[name] = {"seconds": [], "found": found, "accuracy": bench_detection.score(docs, found)}
for trial in range(args.trials):
    names = list(LAYOUTS)
    names = names[trial % len(names):] + names[:trial % len(names)]
    for name in names:
        seconds, found = detect(name)
        assert found == results[name]["found"], f"{name} is not deterministic"
        results[name]["seconds"].append(seconds)
        print("TIMING " + json.dumps({"trial": trial, "layout": name, "seconds": seconds}), flush=True)

reference = results["current_3pass"]
base = statistics.median(reference["seconds"])
for name, row in results.items():
    median = statistics.median(row["seconds"])
    accuracy, ref = row["accuracy"], reference["accuracy"]
    worse = [f"{label} {metric}" for label in LABELS for metric in ("recall", "precision")
             if accuracy[label][metric] < ref[label][metric]]
    row.update(passes=len(LAYOUTS[name]), median_seconds=median, chunks_per_s=len(chunks) / median,
               speedup=base / median, worse_than_current=worse, accepted=not worse,
               identical_chunks_vs_current=sum(a == b for a, b in zip(row["found"], reference["found"])))
report["results"] = {name: {k: v for k, v in row.items() if k != "found"} for name, row in results.items()}
(out / "pass_result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
(out / "pass_found.json").write_text(json.dumps({n: r["found"] for n, r in results.items()}, ensure_ascii=False))

header = "| layout | passes | chunks/s | speedup | " + " | ".join(f"{label} R/P" for label in LABELS) + \
         " | identical chunks | accepted |"
lines = [header, "|" + "---|" * (6 + len(LABELS))]
for name, row in results.items():
    cells = " | ".join(f"{row['accuracy'][label]['recall']:.3f}/{row['accuracy'][label]['precision']:.3f}"
                       for label in LABELS)
    verdict = "yes" if row["accepted"] else "no: " + ", ".join(row["worse_than_current"])
    lines.append(f"| {name} | {row['passes']} | {row['chunks_per_s']:.2f} | {row['speedup']:.3f}x | {cells} | "
                 f"{row['identical_chunks_vs_current']}/{len(chunks)} | {verdict} |")
(out / "pass_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
print("PASS_BENCH_DONE", flush=True)
