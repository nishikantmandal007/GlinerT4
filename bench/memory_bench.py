# The document memory (production/glinert4/memory.py) measured step by step, for accuracy and time, on:
#   contract:  notebooks/samples/02_service_contract_100p.docx (synthetic, 100 pages) with its manifest, read and scored
#              per page and occurrence by bench/doc_eval.py, including the control values that must not be detected
#   realistic: the realistic corpus (test_corpus/, 040 and 020 sections), each of its 10 documents whole
#   long:      the ten 040 documents joined into one document (~38k words, ~76 pages), and that document repeated
#              --long-repeats times (x4 ~300 pages, x10 ~760 pages), for time against length
# realistic and long are scored per occurrence against the ground-truth character offsets: a detection is right when
# it overlaps an annotated mention of its label (so a street found without its house number counts), and a mention is
# missed when no detection of its label overlaps it. The model runs twice per input, without and with the candidate
# decode that only the opt-in pool step uses, and every variant refines the same spans, so the differences come from
# the memory alone. Reported per variant: precision / recall / false positives / misses per label, controls flagged,
# how many of the spans each memory step added were right, and the memory's own time.
#   python bench/memory_bench.py --out results/ [--sets contract,realistic,long] [--long-repeats 1,4,10] [--standin]
import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Person", "Organization", "Street", "City"]


def variants(MemoryConfig, floor):
    base = MemoryConfig(candidate_floor=floor)
    return {"off": None,
            "labels": replace(base, steps=("labels",)),
            "labels+spread": replace(base, steps=("labels", "spread")),
            "default (labels+spread+aliases)": base,
            "default + bare surnames": replace(base, bare_surnames=True),
            "default + pool": replace(base, steps=base.steps + ("pool",))}


def timed(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - start


def model_runs(detector, texts, floor):
    plain, plain_seconds = timed(detector.spans, texts, LABELS)
    with_candidates, candidate_seconds = timed(detector.spans, texts, LABELS, floor)
    return plain, with_candidates, {"model_s": round(plain_seconds, 3), "model_with_candidates_s": round(candidate_seconds, 3)}


def refine_all(detector, texts, plain, with_candidates, configs):
    for name, memory in configs.items():
        raw = with_candidates if memory is not None and "pool" in memory.steps else plain
        found, seconds = timed(detector.finish, texts, raw, LABELS, memory)
        yield name, found, seconds


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


def coverage(spans, length):
    marks = {label: bytearray(length) for label in LABELS}
    for label, start, end in spans:
        if label in marks:
            marks[label][start:end] = b"\x01" * (end - start)
    return marks


def overlap_marks(truth, occurrences, length):
    expected = coverage(truth, length)
    found = coverage([(o["label"], o["start"], o["end"]) for o in occurrences], length)
    marks = []
    for o in occurrences:
        label, s, e = o["label"], o["start"], o["end"]
        if expected[label].find(1, s, e) != -1:
            kind = "correct"
        elif any(marks_.find(1, s, e) != -1 for marks_ in expected.values()):
            kind = "wrong label"
        else:
            kind = "invented"
        marks.append({"label": label, "kind": kind, "source": o.get("source", "model")})
    marks += [{"label": label, "kind": "missed"} for label, s, e in truth
              if label in found and found[label].find(1, s, e) == -1]
    return marks


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
    plain, with_candidates, times = model_runs(detector, [document], floor)
    rows = {}
    for name, found, seconds in refine_all(detector, [document], plain, with_candidates, configs):
        per_part = {part: [] for part in parts}
        for occurrence in found[0]["occurrences"]:
            part = max((p for p in parts if starts[p] <= occurrence["start"]), key=starts.get)
            shift = starts[part]
            per_part[part].append({**occurrence, "start": occurrence["start"] - shift,
                                   "end": min(occurrence["end"] - shift, len(parts[part]))})
        marks = doc_eval.judge(parts, per_part, truth, controls)
        rows[name] = summarise([m for ms in marks.values() for m in ms], seconds)
    return {**times, "words": len(document.split()), "pages": sum(isinstance(p, int) for p in parts),
            "truth_occurrences": sum(truth.values()), "controls": len(controls), "not_scored": dict(other),
            "variants": rows}


def corpus_docs(size, limit=0):
    sys.path.insert(0, str(ROOT / "bench"))
    import realistic_bench

    return realistic_bench.prepare(ROOT / "test_corpus", size)[:limit or None]


def scored_set(detector, floor, configs, texts, truths):
    plain, with_candidates, times = model_runs(detector, texts, floor)
    rows = {}
    for name, found, seconds in refine_all(detector, texts, plain, with_candidates, configs):
        marks = [m for text, truth, result in zip(texts, truths, found)
                 for m in overlap_marks(truth, result["occurrences"], len(text))]
        rows[name] = summarise(marks, seconds)
    return {**times, "documents": len(texts), "words": sum(len(t.split()) for t in texts), "variants": rows}


def realistic(detector, floor, configs, sizes, limit=0):
    out = {}
    for size in sizes:
        docs = corpus_docs(size, limit)
        out[size] = scored_set(detector, floor, configs, [d["text"] for d in docs],
                               [[tuple(o) for o in d["occurrences"]] for d in docs])
    return out


def long_documents(detector, floor, configs, repeats, limit=0):
    docs = corpus_docs("040", limit)
    text, truth = "", []
    for doc in docs:
        shift = len(text) + (2 if text else 0)
        text = text + "\n\n" + doc["text"] if text else doc["text"]
        truth += [(label, s + shift, e + shift) for label, s, e in doc["occurrences"]]
    out = {}
    for repeat in repeats:
        big, big_truth = text, list(truth)
        for k in range(1, repeat):
            shift = k * (len(text) + 2)
            big += "\n\n" + text
            big_truth += [(label, s + shift, e + shift) for label, s, e in truth]
        out[f"x{repeat}"] = scored_set(detector, floor, configs, [big], [big_truth])
    return out


def table(title, result):
    lines = [f"### {title}", "", "| variant | " + " | ".join(f"{label} P / R" for label in LABELS)
             + " | FP | FN | controls flagged | finishing ms |", "|---|" + "---|" * (len(LABELS) + 4)]
    for name, row in result["variants"].items():
        cells = [f"{row['per_label'][l]['precision']:.3f} / {row['per_label'][l]['recall']:.3f}" for l in LABELS]
        fp = sum(row["per_label"][l]["fp"] for l in LABELS)
        fn = sum(row["per_label"][l]["fn"] for l in LABELS)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {fp} | {fn} | {row['controls_flagged']} | "
                     f"{row['memory_ms']} |")
    lines.append("")
    for name, row in result["variants"].items():
        if row["memory_added"]:
            lines.append(f"- {name}: memory added " + ", ".join(f"{k} {v}" for k, v in row["memory_added"].items()))
    return lines + [""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--sets", default="contract,realistic,long")
    parser.add_argument("--sizes", default="040,020")
    parser.add_argument("--long-repeats", default="1,4,10")
    parser.add_argument("--floor", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0, help="first N contract pages and corpus documents")
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
    sets = []
    if "contract" in args.sets:
        c = report["contract"] = contract(detector, args.floor, configs, args.limit)
        sets.append((f"Contract: {c['pages']} pages + header, footer, comments, properties; {c['words']:,} words; "
                     f"{c['truth_occurrences']} ground-truth occurrences; {c['controls']} controls", c))
        print("CONTRACT " + json.dumps(c), flush=True)
    if "realistic" in args.sets:
        report["realistic"] = realistic(detector, args.floor, configs, args.sizes.split(","), args.limit)
        for size, r in report["realistic"].items():
            sets.append((f"Realistic corpus {size}: {r['documents']} whole documents, {r['words']:,} words", r))
            print("REALISTIC " + json.dumps({"size": size, **r}), flush=True)
    if "long" in args.sets:
        repeats = [int(r) for r in args.long_repeats.split(",")]
        report["long"] = long_documents(detector, args.floor, configs, repeats, args.limit)
        for name, r in report["long"].items():
            sets.append((f"Long document {name}: one document of {r['words']:,} words (~{r['words'] // 500} pages)", r))
            print("LONG " + json.dumps({"repeat": name, **r}), flush=True)
    lines = [f"Document memory on {report['device']}. Finishing = table-cell split and overlap resolution (memory off); "
             "the memory columns are the finishing step with the memory on.", "",
             "| input | words | model s | model s with candidate decode | finishing ms | with memory (default) ms "
             "| with memory + pool ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for title, r in sets:
        v = r["variants"]
        lines.append(f"| {title.split(':')[0]} | {r['words']:,} | {r['model_s']} | {r['model_with_candidates_s']} | "
                     f"{v['off']['memory_ms']} | {v['default (labels+spread+aliases)']['memory_ms']} | "
                     f"{v['default + pool']['memory_ms']} |")
    lines.append("")
    for title, r in sets:
        lines += table(title, r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "memory_bench.json").write_text(json.dumps(report, indent=1))
    (out / "memory_bench.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
