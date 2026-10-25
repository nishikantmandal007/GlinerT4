# The document memory (production/glinert4/memory.py) measured step by step on two sets with ground truth:
#   contract:  notebooks/samples/02_service_contract_100p.docx (synthetic, 100 pages) with its manifest, read and scored
#              per page and occurrence by bench/doc_eval.py, including the control values that must not be detected
#   realistic: the realistic corpus (test_corpus/, 040 and 020 sections) as whole documents, scored per occurrence
#              against the ground-truth character offsets by bench/realistic_bench.py
# The model runs once per set (glinert4 Detector defaults: one pass, 384/64 windows), with candidates from a second
# decode at --floor; every variant then refines the same spans, so the differences come from the memory alone. Per
# variant: precision / recall / false positives / misses per label, controls flagged, how many of the spans each memory
# step added or relabelled were right, and the memory's own time.
#   python bench/memory_bench.py --out results/ [--sets contract,realistic] [--floor 0.3] [--standin]
import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Person", "Organization", "Street", "City"]
STEPS = ("labels", "pool", "spread", "aliases")


def variants(MemoryConfig, floor):
    base = MemoryConfig(candidate_floor=floor)
    return {"off": None,
            "labels": replace(base, steps=STEPS[:1]),
            "labels+pool": replace(base, steps=STEPS[:2]),
            "labels+pool+spread": replace(base, steps=STEPS[:3]),
            "all four": base,
            "all four, no bare surnames": replace(base, bare_surnames=False)}


def contract(detector, floor, configs, limit=0):
    sys.path.insert(0, str(ROOT / "bench"))
    import doc_eval

    sample = ROOT / "notebooks/samples/02_service_contract_100p.docx"
    parts = doc_eval.read_document(sample.name, sample.read_bytes())
    truth, controls, other = doc_eval.read_ground_truth(Path(str(sample) + ".manifest.json").read_text())
    if limit:
        parts = {name: text for name, text in parts.items() if not isinstance(name, int) or name <= limit}
        truth = type(truth)({key: n for key, n in truth.items() if key[0] in parts})
    starts, position = {}, 0
    for name, text in parts.items():
        starts[name] = position
        position += len(text) + 2
    document = "\n\n".join(parts.values())
    start = time.perf_counter()
    raw = detector.spans([document], LABELS, floor)
    model_seconds = time.perf_counter() - start
    rows = {}
    for name, memory in configs.items():
        start = time.perf_counter()
        found = detector.finish([document], raw, LABELS, memory)[0]
        memory_seconds = time.perf_counter() - start
        per_part = {part: [] for part in parts}
        for occurrence in found["occurrences"]:
            part = max((p for p in parts if starts[p] <= occurrence["start"]), key=starts.get)
            shift = starts[part]
            per_part[part].append({**occurrence, "start": occurrence["start"] - shift,
                                   "end": min(occurrence["end"] - shift, len(parts[part]))})
        marks = doc_eval.judge(parts, per_part, truth, controls)
        rows[name] = summarise([m for ms in marks.values() for m in ms], memory_seconds)
    return {"model_seconds": round(model_seconds, 3), "words": len(document.split()), "pages": len(parts),
            "truth_occurrences": sum(truth.values()), "controls": len(controls), "not_scored": dict(other),
            "variants": rows}


def summarise(marks, memory_seconds):
    per_label = {}
    for label in LABELS:
        kinds = Counter(m["kind"] for m in marks if m["label"] == label)
        tp, fn = kinds["correct"], kinds["missed"]
        fp = sum(kinds.values()) - tp - fn
        per_label[label] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / max(1, tp + fp), 4),
                            "recall": round(tp / max(1, tp + fn), 4)}
    added = Counter((m.get("source", "model"), "right" if m["kind"] == "correct" else m["kind"])
                    for m in marks if m["kind"] != "missed" and m.get("source", "model") != "model")
    return {"per_label": per_label, "errors": dict(Counter(m["kind"] for m in marks if m["kind"] != "correct")),
            "controls_flagged": sum(m["kind"] == "control flagged" for m in marks),
            "memory_added": {f"{source} {verdict}": n for (source, verdict), n in sorted(added.items())},
            "memory_ms": round(1000 * memory_seconds, 1)}


def realistic(detector, floor, configs, sizes, limit=0):
    sys.path.insert(0, str(ROOT / "bench"))
    import realistic_bench

    out = {}
    for size in sizes:
        docs = [{**doc, "chunks": [doc["text"]], "chunk_offsets": [0], "position_map": None}
                for doc in realistic_bench.prepare(ROOT / "test_corpus", size)][:limit or None]
        texts = [doc["text"] for doc in docs]
        start = time.perf_counter()
        raw = detector.spans(texts, LABELS, floor)
        model_seconds = time.perf_counter() - start
        rows = {}
        for name, memory in configs.items():
            start = time.perf_counter()
            found = detector.finish(texts, raw, LABELS, memory)
            memory_seconds = time.perf_counter() - start
            marks = []
            for doc, result in zip(docs, found):
                truth = {tuple(o) for o in doc["occurrences"]}
                predicted = {(o["label"], o["start"], o["end"]): o.get("source", "model") for o in result["occurrences"]}
                for (label, s, e), source in predicted.items():
                    if (label, s, e) in truth:
                        kind = "correct"
                    elif any(t[0] == label and t[1] < e and s < t[2] for t in truth):
                        kind = "boundary"
                    elif any(t[1] < e and s < t[2] for t in truth):
                        kind = "wrong label"
                    else:
                        kind = "invented"
                    marks.append({"label": label, "kind": kind, "source": source})
                for label, s, e in truth - set(predicted):
                    if not any(p[0] == label and p[1] < e and s < p[2] for p in predicted):
                        marks.append({"label": label, "kind": "missed"})
            rows[name] = summarise(marks, memory_seconds)
        out[size] = {"model_seconds": round(model_seconds, 3), "documents": len(docs),
                     "words": sum(len(t.split()) for t in texts), "variants": rows}
    return out


def table(title, result):
    lines = [f"### {title}", "", "| variant | " + " | ".join(f"{label} P / R" for label in LABELS)
             + " | FP | FN | controls flagged | memory ms |", "|---|" + "---|" * (len(LABELS) + 4)]
    for name, row in result["variants"].items():
        cells = [f"{row['per_label'][l]['precision']:.3f} / {row['per_label'][l]['recall']:.3f}" for l in LABELS]
        fp = sum(row["per_label"][l]["fp"] for l in LABELS)
        fn = sum(row["per_label"][l]["fn"] for l in LABELS)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {fp} | {fn} | {row['controls_flagged']} | "
                     f"{row['memory_ms']} |")
    lines += ["", "Spans the memory added or relabelled (all four steps), by verdict: "
              + (", ".join(f"{k} {v}" for k, v in result["variants"]["all four"]["memory_added"].items()) or "none"), ""]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--sets", default="contract,realistic")
    parser.add_argument("--sizes", default="040,020")
    parser.add_argument("--floor", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0, help="first N contract pages and realistic documents")
    parser.add_argument("--standin", action="store_true", help="CPU smoke test with a stand-in model")
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "production"), str(ROOT / "production" / "tests")]
    import torch

    from glinert4 import Detector, runtime
    from glinert4.memory import MemoryConfig

    if args.standin:
        import standin

        detector = Detector(device="cpu", model=runtime.freeze_eval(standin.build()))
    else:
        detector = Detector()
    configs = variants(MemoryConfig, args.floor)
    detector.detect(["Anna Müller of Siemens AG lives at Hauptstraße 5, Berlin."])
    report = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() and not args.standin else "cpu",
              "floor": args.floor}
    lines = [f"Document memory on {report['device']}, candidates from a second decode at {args.floor}", ""]
    if "contract" in args.sets:
        report["contract"] = contract(detector, args.floor, configs, args.limit)
        c = report["contract"]
        lines += table(f"Contract: {c['pages']} parts, {c['words']:,} words, {c['truth_occurrences']} ground-truth "
                       f"occurrences, {c['controls']} controls, model {c['model_seconds']} s", c)
        print("CONTRACT " + json.dumps(c), flush=True)
    if "realistic" in args.sets:
        report["realistic"] = realistic(detector, args.floor, configs, args.sizes.split(","), args.limit)
        for size, r in report["realistic"].items():
            lines += table(f"Realistic corpus {size}: {r['documents']} documents, {r['words']:,} words, "
                           f"model {r['model_seconds']} s", r)
            print("REALISTIC " + json.dumps({"size": size, **r}), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "memory_bench.json").write_text(json.dumps(report, indent=1))
    (out / "memory_bench.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
