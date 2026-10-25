# Can one label pass match today's accuracy? One pass encodes each chunk once instead of twice (about half the GPU
# work), but labels sharing a prompt shift each other's confidences. This sweeps, through the production detect_batch
# and scored against the corpus ground truth as bench_detection.py scores it:
#   1. the reference: today's layout at today's thresholds (and the 2-pass layout that drops the City-alone pass);
#   2. single-pass layouts (label order, City description) at uniform thresholds 0.30..0.90, per-label recall/precision;
#   3. per layout, a per-label threshold that meets the reference recall and precision for that label, if one exists,
#      and then that exact setting run end to end (labels interact a little through span resolution, so the final run,
#      not the sweep, decides), timed.
# The corpus is 10 documents: thresholds chosen on it are tuned to it. A layout that passes here needs the held-out
# check (25-language corpus) before it counts.
#   python bench/single_pass_bench.py --corpus corpus.json --out results_dir [--standin]
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--corpus", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--trials", type=int, default=2)
parser.add_argument("--standin", action="store_true", help="CPU smoke test of this script with a stand-in model")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.update(DETECTION_FAST_COUNT="0", DETECTION_SPLIT_CACHE="1", DETECTION_SKIP_EVAL="1", DETECTION_STREAMS="1",
                  DETECTION_EXACT_COUNT="1", GLINER_T4_FUSE_LAYERNORM="1", DETECTION_PIPELINE="1")
if not args.standin:
    os.environ["DETECTION_ATTENTION_KERNEL"] = str(ROOT)
sys.path.insert(0, str(ROOT / "artifacts"))
sys.path.insert(0, str(ROOT / "serving"))
sys.path.insert(0, str(ROOT / "tests"))

import torch  # noqa: E402

from colab_pipeline import bench_detection  # noqa: E402
from colab_pipeline import gliner2_ner as ner  # noqa: E402

DEVICE = "cpu" if args.standin else "cuda"
LABELS = bench_detection.LABELS
CITY_ALONE = ner.CITY_ALONE_DESCRIPTION
REFERENCE_THRESHOLDS = dict(ner.LABEL_THRESHOLDS), ner.DEFAULT_THRESHOLD  # Organization 0.85, others 0.65
GRID = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
if args.standin:
    import gliner2_exact
    import gliner2_pipelined
    import gliner2_standin

    ner.loaded_checkpoints[("legacy", "cpu", None)] = gliner2_standin.build()
    gliner2_exact.install(count_step0=True, format_cache=False)
    ner.PIPELINE, ner.gliner2_pipelined = True, gliner2_pipelined
    GRID = [0.3, 0.65, 0.85]

REFERENCE_LAYOUTS = {
    "current_3pass": [(["Organization"], {}), (["Person", "Street", "City"], {}), (["City"], {"City": CITY_ALONE})],
    "drop_city_pass": [(["Organization"], {}), (["Person", "Street", "City"], {})],
}
SINGLE_LAYOUTS = {
    "single_person_first": [(["Person", "Organization", "Street", "City"], {})],
    "single_org_first": [(["Organization", "Person", "Street", "City"], {})],
    "single_org_first_city_desc": [(["Organization", "Person", "Street", "City"], {"City": CITY_ALONE})],
}

docs = json.loads(Path(args.corpus).read_text())
chunks = [chunk for doc in docs for chunk in doc["chunks"]]
detector = ner.Gliner2NER()


def detect(layout, thresholds, default):
    passes = {**REFERENCE_LAYOUTS, **SINGLE_LAYOUTS}[layout]
    ner.strategy_passes = lambda labels: [("legacy", list(group), dict(overrides)) for group, overrides in passes]
    ner.LABEL_THRESHOLDS, ner.DEFAULT_THRESHOLD = dict(thresholds), default
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    found = detector.detect_batch(chunks, LABELS, device=DEVICE, batch_size=16)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start, [item["values"] for item in found]


def score(found):
    return bench_detection.score(docs, found)


def timed(layout, thresholds, default):
    detect(layout, thresholds, default)  # warm-up
    seconds, found = [], None
    for _ in range(args.trials):
        elapsed, found = detect(layout, thresholds, default)
        seconds.append(elapsed)
    return {"chunks_per_s": len(chunks) / statistics.median(seconds), "seconds": seconds, "found": found}


report = {"corpus": {"documents": len(docs), "chunks": len(chunks),
                     "sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()},
          "device": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu (stand-in)", "grid": GRID}
reference = {}
for name in REFERENCE_LAYOUTS:
    run = timed(name, *REFERENCE_THRESHOLDS)
    reference[name] = {"chunks_per_s": run["chunks_per_s"], "accuracy": score(run["found"]), "found": run["found"]}
    print("REFERENCE " + json.dumps({name: {k: v for k, v in reference[name].items() if k != "found"}}), flush=True)
target = reference["current_3pass"]["accuracy"]


def meets(accuracy, label):
    return (accuracy[label]["recall"] >= target[label]["recall"]
            and accuracy[label]["precision"] >= target[label]["precision"])


singles = {}
for name in SINGLE_LAYOUTS:
    sweep = {}
    for t in GRID:
        _, found = detect(name, {}, t)
        sweep[t] = score(found)
        print("SWEEP " + json.dumps({"layout": name, "threshold": t, "accuracy": sweep[t]}), flush=True)
    # per label: the thresholds that meet today's numbers; take the one with the best precision, then recall
    chosen, missing = {}, []
    for label in LABELS:
        ok = [t for t in GRID if meets(sweep[t], label)]
        if ok:
            chosen[label] = max(ok, key=lambda t: (sweep[t][label]["precision"], sweep[t][label]["recall"], -t))
        else:
            missing.append(label)
            # closest miss, for the report: the threshold with the smallest total shortfall
            chosen[label] = min(GRID, key=lambda t: max(0, target[label]["recall"] - sweep[t][label]["recall"])
                                + max(0, target[label]["precision"] - sweep[t][label]["precision"]))
    run = timed(name, {label: chosen[label] for label in LABELS}, min(chosen.values()))
    accuracy = score(run["found"])
    worse = [f"{label} {metric}" for label in LABELS for metric in ("recall", "precision")
             if accuracy[label][metric] < target[label][metric]]
    singles[name] = {"sweep": {str(t): a for t, a in sweep.items()}, "thresholds": chosen,
                     "no_threshold_meets": missing, "final_accuracy": accuracy, "worse_than_current": worse,
                     "accepted": not worse, "chunks_per_s": run["chunks_per_s"],
                     "speedup_vs_drop_city": run["chunks_per_s"] / reference["drop_city_pass"]["chunks_per_s"],
                     "identical_chunks_vs_current": sum(a == b for a, b in
                                                        zip(run["found"], reference["current_3pass"]["found"]))}
    print("SINGLE " + json.dumps({name: {k: v for k, v in singles[name].items() if k != "sweep"}}), flush=True)

report["reference"] = {n: {k: v for k, v in r.items() if k != "found"} for n, r in reference.items()}
report["single"] = singles
(out / "single_result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

lines = ["| layout | passes | thresholds | chunks/s | " + " | ".join(f"{label} R/P" for label in LABELS) + " | accepted |",
         "|" + "---|" * (5 + len(LABELS))]
for name, r in reference.items():
    cells = " | ".join(f"{r['accuracy'][l]['recall']:.3f}/{r['accuracy'][l]['precision']:.3f}" for l in LABELS)
    lines.append(f"| {name} | {len(REFERENCE_LAYOUTS[name])} | today's | {r['chunks_per_s']:.1f} | {cells} | reference |")
for name, r in singles.items():
    cells = " | ".join(f"{r['final_accuracy'][l]['recall']:.3f}/{r['final_accuracy'][l]['precision']:.3f}"
                       for l in LABELS)
    shown = ", ".join(f"{l[:3]} {t:.2f}" for l, t in r["thresholds"].items())
    verdict = "yes" if r["accepted"] else "no: " + ", ".join(r["worse_than_current"])
    lines.append(f"| {name} | 1 | {shown} | {r['chunks_per_s']:.1f} | {cells} | {verdict} |")
lines.append("")
lines.append("Uniform-threshold sweep, single passes (recall/precision per label):")
lines.append("")
lines.append("| layout | t | " + " | ".join(LABELS) + " |")
lines.append("|" + "---|" * (2 + len(LABELS)))
for name, r in singles.items():
    for t, accuracy in r["sweep"].items():
        lines.append(f"| {name} | {t} | " + " | ".join(
            f"{accuracy[l]['recall']:.3f}/{accuracy[l]['precision']:.3f}" for l in LABELS) + " |")
(out / "single_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
print("SINGLE_PASS_BENCH_DONE", flush=True)
