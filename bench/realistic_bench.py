# Speed and accuracy of the detection layouts on the realistic test corpus (test_corpus/), through the production
# detect_batch at the shipping speed settings (exact count head, fused add + LayerNorm, one-thread pipeline), scored three
# ways against the ground truth:
#   doc-set:      bench_detection.py's score (a document's distinct values; Street matches by containment)
#   occ-overlap:  every annotated occurrence (test_corpus/ground_truth char offsets): recalled if a detected span of the
#                 same label overlaps it; a detected span is correct if it overlaps an annotated one of that label
#   occ-exact:    the same with identical character spans
# Input shapes: the corpus's own pre-cut chunks (~131 words, as in rounds 1-8), and whole documents, which detect_batch
# reads in GLiNER2's own overlapping word windows (docW-O: W words per window, O words of overlap; production and the
# GLiNER2 docs use 384-64, and suggest 96-128 overlap for entities near window edges and smaller windows for speed).
# Every run also gets an error analysis against the ground truth: each detected span that is not an exact annotated
# span is a boundary error (overlaps an annotated mention of its label), a wrong label (overlaps only other labels'
# mentions) or invented (overlaps nothing); each annotated mention no detection of its label overlaps is a miss.
# The 040-section corpus is the rounds 1-7 corpus (the round-7 thresholds were chosen on it); the 020-section corpus is
# the same documents cut shorter and chunked at other boundaries. Neither is held-out data.
#   python bench/realistic_bench.py --out results_dir [--sizes ...] [--inputs chunks,doc384-64,...] [--configs ...]
import argparse
import codecs
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = {"Names": "Person", "Organization": "Organization", "Street": "Street", "City": "City"}


def decode(raw):
    for bom, encoding in ((codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF16_LE, "utf-16"),
                          (codecs.BOM_UTF16_BE, "utf-16")):
        if raw.startswith(bom):
            return raw.decode(encoding)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def prepare(test_corpus, size):
    """The corpus JSON plus, per document, each chunk's offset in the decoded text and the annotated occurrences."""
    docs = json.loads((test_corpus / f"corpus_{size}sections.json").read_text())
    for doc in docs:
        relative = doc["file"][len("test_corpus/"):] if doc["file"].startswith("test_corpus/") else doc["file"]
        text = original = decode((test_corpus / relative).read_bytes())
        # the 005/010/020 corpora also turned CRLF line ends into LF; the ground truth counts the CR characters, so
        # chunks are located in the LF text and their positions mapped back through position_map
        position_map = None
        if any(chunk.lstrip("\ufeff") not in text for chunk in doc["chunks"]) and "\r\n" in text:
            position_map = [i for i, ch in enumerate(text) if not (ch == "\r" and text[i + 1:i + 2] == "\n")]
            position_map.append(len(text))
            text = text.replace("\r\n", "\n")
        offsets, position = [], 0
        for chunk in doc["chunks"]:
            # the 005/010/020 corpora keep a UTF-8 byte-order mark at the start of the first chunk; the decoded text
            # (and the ground-truth offsets) do not, so that chunk sits one character before the text it shows
            lead = len(chunk) - len(chunk.lstrip("\ufeff"))
            index = text.find(chunk[lead:], position)
            if index < 0:
                raise ValueError(f"chunk not found in {doc['file']}")
            offsets.append(index - lead)
            position = index + len(chunk) - lead
        truth = json.loads((test_corpus / "ground_truth" / (relative[len("txt/"):] + ".ground_truth.json")).read_text())
        occurrences = []
        for kind, entry in truth["by_type"].items():
            if kind in LABEL_MAP:
                for value in entry["values"]:
                    for location in value["locations"]:
                        occurrences.append([LABEL_MAP[kind], location["char_start"], location["char_end"]])
        doc["chunk_offsets"], doc["occurrences"], doc["position_map"] = offsets, occurrences, position_map
        doc["text"] = original  # the whole decoded document, in the ground truth's coordinates
    return docs


def predictions(docs, found):
    """Per document, the detected (label, start, end) spans in the ground truth's coordinates."""
    per_doc, index = [], 0
    for doc in docs:
        predicted, mapped = [], doc.get("position_map") or None
        for offset in doc["chunk_offsets"]:
            for o in found[index]["occurrences"]:
                start, end = offset + o["start"], offset + o["end"]
                if mapped:
                    start, end = mapped[start], mapped[end - 1] + 1
                predicted.append((o["label"], start, end))
            index += 1
        per_doc.append(predicted)
    return per_doc


def error_analysis(docs, found, labels, context=40):
    """Every non-exact detection by kind (boundary / wrong_label / invented) and every miss, with surrounding text."""
    def snippet(text, start, end):
        return (text[max(0, start - context):start] + "[[" + text[start:end] + "]]" + text[end:end + context]).replace(
            "\r", "").replace("\n", " ")

    def overlaps(a, b):
        return a[1] < b[2] and b[1] < a[2]

    errors = {label: {"boundary": [], "wrong_label": [], "invented": [], "missed": []} for label in labels}
    for doc, predicted in zip(docs, predictions(docs, found)):
        text, truth = doc["text"], [tuple(o) for o in doc["occurrences"]]
        exact = set(truth)
        for p in predicted:
            if p in exact or p[0] not in errors:
                continue
            same = [t for t in truth if t[0] == p[0] and overlaps(p, t)]
            other = [t for t in truth if t[0] != p[0] and overlaps(p, t)]
            kind = "boundary" if same else "wrong_label" if other else "invented"
            item = {"file": doc["file"], "detected": text[p[1]:p[2]], "context": snippet(text, p[1], p[2])}
            if same or other:
                item["annotated"] = [f"{t[0]}: {text[t[1]:t[2]]}" for t in (same or other)]
            errors[p[0]][kind].append(item)
        for t in truth:
            if t[0] in errors and not any(p[0] == t[0] and overlaps(p, t) for p in predicted):
                errors[t[0]]["missed"].append({
                    "file": doc["file"], "annotated": text[t[1]:t[2]], "context": snippet(text, t[1], t[2]),
                    "detected_as": sorted({p[0] for p in predicted if overlaps(p, t)})})
    return errors


def occurrence_scores(docs, found, labels):
    totals = {label: {"gt": 0, "gt_hit": 0, "gt_exact": 0, "pred": 0, "pred_hit": 0, "pred_exact": 0} for label in labels}
    for doc, predicted in zip(docs, predictions(docs, found)):
        truth = [tuple(o) for o in doc["occurrences"]]
        exact = set(truth) & set(predicted)
        for label in labels:
            gt = [t for t in truth if t[0] == label]
            pred = [p for p in predicted if p[0] == label]
            row = totals[label]
            row["gt"] += len(gt)
            row["pred"] += len(pred)
            row["gt_hit"] += sum(any(p[1] < t[2] and t[1] < p[2] for p in pred) for t in gt)
            row["pred_hit"] += sum(any(p[1] < t[2] and t[1] < p[2] for t in gt) for p in pred)
            row["gt_exact"] += sum(t in exact for t in gt)
            row["pred_exact"] += sum(p in exact for p in pred)
    return {label: {"overlap_recall": round(r["gt_hit"] / max(1, r["gt"]), 3),
                    "overlap_precision": round(r["pred_hit"] / max(1, r["pred"]), 3),
                    "exact_recall": round(r["gt_exact"] / max(1, r["gt"]), 3),
                    "exact_precision": round(r["pred_exact"] / max(1, r["pred"]), 3),
                    "annotated": r["gt"], "detected": r["pred"]} for label, r in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--sizes", default="040,020,010,005")
    parser.add_argument("--inputs", default="chunks,doc384-64,doc384-128,doc256-64,doc192-48",
                        help="chunks: the corpus's pre-cut chunks; docW-O: whole documents in GLiNER2's own windows of "
                             "W words overlapping by O")
    parser.add_argument("--configs", default="current_3pass,single_street_0.5", help="comma-separated, or 'all'")
    parser.add_argument("--test-corpus", default=str(ROOT / "test_corpus"))
    parser.add_argument("--prepared", help="JSON from --prepare-only (used on Colab instead of test_corpus/)")
    parser.add_argument("--prepare-only", help="write the prepared corpora to this path and exit")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--standin", action="store_true", help="CPU smoke test of this script with a stand-in model")
    args = parser.parse_args()
    sizes = args.sizes.split(",")
    if args.prepared:
        corpora = json.loads(Path(args.prepared).read_text())
    else:
        corpora = {size: prepare(Path(args.test_corpus), size) for size in sizes}
    if args.prepare_only:
        Path(args.prepare_only).write_text(json.dumps(corpora, ensure_ascii=False))
        print(f"prepared {', '.join(corpora)} -> {args.prepare_only}")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.update(DETECTION_FAST_COUNT="0", DETECTION_SPLIT_CACHE="1", DETECTION_SKIP_EVAL="1",
                      DETECTION_STREAMS="1", DETECTION_EXACT_COUNT="1", GLINER_T4_FUSE_LAYERNORM="1",
                      DETECTION_PIPELINE="1")
    if not args.standin:
        os.environ["DETECTION_ATTENTION_KERNEL"] = str(ROOT)
    for sub in ("artifacts", "serving", "tests"):
        sys.path.insert(0, str(ROOT / sub))
    import torch

    from colab_pipeline import bench_detection
    from colab_pipeline import gliner2_ner as ner

    device = "cpu" if args.standin else "cuda"
    if args.standin:
        import gliner2_exact
        import gliner2_pipelined
        import gliner2_standin

        ner.loaded_checkpoints[("legacy", "cpu", None)] = gliner2_standin.build()
        gliner2_exact.install(count_step0=True, format_cache=False)
        ner.PIPELINE, ner.gliner2_pipelined = True, gliner2_pipelined
    labels = bench_detection.LABELS
    city_alone = ner.CITY_ALONE_DESCRIPTION
    three = [(["Organization"], {}), (["Person", "Street", "City"], {}), (["City"], {"City": city_alone})]
    two = [(["Organization"], {}), (["Person", "Street", "City"], {})]
    single = [(["Person", "Organization", "Street", "City"], {})]
    today = ({"Organization": 0.85}, 0.65)
    # name: (passes, per-label thresholds, default threshold)
    configs = {
        "current_3pass": (three, *today),
        "drop_city_pass": (two, *today),
        "single_today_thresholds": (single, *today),
        "single_street_0.5": (single, {"Organization": 0.85, "Street": 0.5}, 0.65),  # one change from today
        "single_round7": (single, {"Person": 0.85, "Organization": 0.80, "Street": 0.30, "City": 0.90}, 0.30),
    }
    if args.configs != "all":
        configs = {name: configs[name] for name in args.configs.split(",")}
    detector = ner.Gliner2NER()

    def detect(name, chunks):
        passes, thresholds, default = configs[name]
        ner.strategy_passes = lambda _: [("legacy", list(group), dict(over)) for group, over in passes]
        ner.LABEL_THRESHOLDS, ner.DEFAULT_THRESHOLD = dict(thresholds), default
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        found = detector.detect_batch(chunks, labels, device=device, batch_size=16)
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start, found

    report = {"device": torch.cuda.get_device_name() if device == "cuda" else "cpu (stand-in)", "corpora": {}}
    lines, error_lines, errors_all = [], [], {}
    default_window = (ner.WINDOW_WORDS, ner.WINDOW_OVERLAP_WORDS)
    runs = [(size, shape) for size in corpora for shape in args.inputs.split(",")]
    for size, shape in runs:
        docs = corpora[size]
        ner.WINDOW_WORDS, ner.WINDOW_OVERLAP_WORDS = default_window
        if shape != "chunks":  # one text per document, at offset 0 of the ground truth's text, in GLiNER2's windows
            window, overlap = (int(v) for v in shape[len("doc"):].split("-"))
            ner.WINDOW_WORDS, ner.WINDOW_OVERLAP_WORDS = window, overlap
            docs = [{**doc, "chunks": [doc["text"]], "chunk_offsets": [0], "position_map": None} for doc in docs]
        chunks = [chunk for doc in docs for chunk in doc["chunks"]]
        words = sum(len(chunk.split()) for chunk in chunks)
        rows = {}
        for name in configs:
            detect(name, chunks)  # warm-up
            seconds, found = [], None
            for _ in range(args.trials):
                elapsed, found = detect(name, chunks)
                seconds.append(elapsed)
            median = statistics.median(seconds)
            errors = error_analysis(docs, found, labels)
            errors_all[f"{size}|{shape}|{name}"] = errors
            rows[name] = {"seconds": seconds, "median_seconds": median, "texts_per_s": len(chunks) / median,
                          "words_per_s": words / median,
                          "doc_set": bench_detection.score(docs, [item["values"] for item in found]),
                          "occurrence": occurrence_scores(docs, found, labels),
                          "errors": {label: {kind: len(items) for kind, items in kinds.items()}
                                     for label, kinds in errors.items()}}
            print("RESULT " + json.dumps({"corpus": size, "inputs": shape, "config": name, **rows[name]}), flush=True)
        ref = rows["current_3pass"]
        for row in rows.values():
            row["worse_than_current"] = [
                f"{label} {view} {metric}" for label in labels
                for view, metrics in (("doc_set", ("recall", "precision")),
                                      ("occurrence", ("overlap_recall", "overlap_precision")))
                for metric in metrics if row[view][label][metric] < ref[view][label][metric]]
        report["corpora"][f"{size}_{shape}"] = {"inputs": shape, "documents": len(docs), "texts": len(chunks),
                                                "words": words, "configs": rows}
        lines += [f"### corpus_{size}sections, {shape} ({len(docs)} documents, {len(chunks)} texts in, {words} words)",
                  "", "| config | texts/s | words/s | " + " | ".join(
                      f"{label} doc R/P · occ R/P · exact R/P" for label in labels) + " | worse than current |",
                  "|" + "---|" * (4 + len(labels))]
        for name, row in rows.items():
            cells = " | ".join(
                f"{row['doc_set'][l]['recall']:.3f}/{row['doc_set'][l]['precision']:.3f} · "
                f"{row['occurrence'][l]['overlap_recall']:.3f}/{row['occurrence'][l]['overlap_precision']:.3f} · "
                f"{row['occurrence'][l]['exact_recall']:.3f}/{row['occurrence'][l]['exact_precision']:.3f}"
                for l in labels)
            worse = ", ".join(row["worse_than_current"]) or ("reference" if name == "current_3pass" else "none")
            lines.append(f"| {name} | {row['texts_per_s']:.1f} | {row['words_per_s']:.0f} | {cells} | {worse} |")
        lines += ["", "Errors per label (boundary / wrong label / invented / missed), median seconds per run:", "",
                  "| config | seconds | " + " | ".join(labels) + " |", "|" + "---|" * (2 + len(labels))]
        for name, row in rows.items():
            lines.append(f"| {name} | {row['median_seconds']:.2f} | " + " | ".join(
                "{boundary} / {wrong_label} / {invented} / {missed}".format(**row["errors"][l]) for l in labels) + " |")
        lines.append("")
    (out / "realistic_result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out / "realistic_errors.json").write_text(json.dumps(errors_all, indent=1, ensure_ascii=False))
    # readable examples: the largest corpus, every input shape and config, up to 15 items per label and kind
    largest = max(corpora, key=lambda size: sum(len(d["chunks"]) for d in corpora[size]))
    for key, errors in errors_all.items():
        size, shape, name = key.split("|")
        if size != largest:
            continue
        error_lines += [f"## corpus_{size}sections, {shape}, {name}", ""]
        for label, kinds in errors.items():
            for kind, items in kinds.items():
                if not items:
                    continue
                error_lines.append(f"**{label} {kind}: {len(items)}**")
                for item in items[:15]:
                    extra = f" (annotated: {'; '.join(item['annotated'])})" if item.get("annotated") and kind != "missed" \
                        else f" (detected as: {', '.join(item['detected_as'])})" if item.get("detected_as") else ""
                    error_lines.append(f"- `{item['context']}`{extra} — {item['file'].split('/')[-1]}")
                error_lines.append("")
    (out / "realistic_errors.md").write_text("\n".join(error_lines) + "\n")
    (out / "realistic_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("REALISTIC_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
