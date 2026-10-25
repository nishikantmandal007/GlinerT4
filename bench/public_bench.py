# Public NER benchmarks, GLiNER-style: every label of a dataset in one prompt, threshold 0.5, strict entity-level
# micro precision/recall/F1 (label, start, end must match). Datasets:
#   crossner_{ai,literature,music,politics,science}  the GLiNER2 paper's zero-shot NER benchmark (English test sets)
#   naamapadam_{as,bn,gu,hi,kn,ml,mr,or,pa,ta,te}    ai4bharat/naamapadam test sets, PER / ORG / LOC (Indian languages)
#   wikiann_{lang}                                    unimelb-nlp/wikiann test sets, PER / ORG / LOC
# Setups per checkpoint: stock_fp32 (gliner2 defaults), stock_fp16 (fp16 + disentangled_flash) and glinert4
# (production/: sm_75 kernel, fused add + LayerNorm, exact count head, pipeline, length buckets). Reports F1 and speed
# per setup, glinert4's agreement with stock_fp16, and for Person the missed and false names with their sentences.
#   python bench/public_bench.py --crossner-dir crossner/ --out results/ [--checkpoints ...] [--limit 1000]
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CROSSNER_DOMAINS = ("ai", "literature", "music", "politics", "science")
NAAMAPADAM_LANGS = ("as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te")
WIKIANN_LANGS = ("en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "de", "fr", "es")
CROSSNER_NAMES = {"programlang": "programming language", "misc": "miscellaneous", "musicalartist": "musical artist",
                  "musicalinstrument": "musical instrument", "musicgenre": "music genre",
                  "literarygenre": "literary genre", "politicalparty": "political party",
                  "academicjournal": "academic journal", "astronomicalobject": "astronomical object",
                  "chemicalcompound": "chemical compound", "chemicalelement": "chemical element"}
PER_ORG_LOC = {"PER": "person", "ORG": "organization", "LOC": "location"}
BIO_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def spans_from_bio(tokens, tags, names):
    """Text joined by single spaces, and the gold (label, start, end) character spans of its BIO tags."""
    starts, position = [], 0
    for token in tokens:
        starts.append(position)
        position += len(token) + 1
    text = " ".join(tokens)
    spans, current = [], None
    for i, tag in enumerate(list(tags) + ["O"]):
        kind = tag[2:] if tag[:2] in ("B-", "I-") else None
        if current and (tag.startswith("B-") or kind != current[0]):
            spans.append((names[current[0]], starts[current[1]], starts[i - 1] + len(tokens[i - 1])))
            current = None
        if kind and current is None:
            current = (kind, i)
    return text, spans


def read_conll(path):
    sentences, tokens, tags = [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines() + [""]:
        parts = line.split("\t")
        if len(parts) == 2:
            tokens.append(parts[0])
            tags.append(parts[1])
        elif tokens:
            sentences.append((tokens, tags))
            tokens, tags = [], []
    return sentences


def load_crossner(directory, limit):
    datasets = {}
    for domain in CROSSNER_DOMAINS:
        sentences = read_conll(Path(directory) / f"{domain}.txt")[:limit]
        kinds = sorted({tag[2:] for _, tags in sentences for tag in tags if tag != "O"})
        names = {kind: CROSSNER_NAMES.get(kind, kind) for kind in kinds}
        datasets[f"crossner_{domain}"] = {"labels": [names[k] for k in kinds],
                                          "examples": [spans_from_bio(t, g, names) for t, g in sentences]}
    return datasets


def load_hf(name, config, limit):
    from datasets import load_dataset

    attempts = [dict(), dict(revision="refs/convert/parquet"), dict(trust_remote_code=True)]
    errors = []
    for options in attempts:
        try:
            data = load_dataset(name, config, split="test", **options)
            break
        except Exception as error:
            errors.append(f"{options}: {type(error).__name__}: {str(error)[:200]}")
    else:
        raise RuntimeError(f"{name}/{config} could not be loaded: " + " | ".join(errors))
    feature = data.features["ner_tags"].feature
    tag_names = list(feature.names) if hasattr(feature, "names") else BIO_TAGS
    rows = data.select(range(min(limit, len(data))))
    return [spans_from_bio(row["tokens"], [tag_names[t] if isinstance(t, int) else t for t in row["ner_tags"]],
                           PER_ORG_LOC) for row in rows]


def load_public(limit, which):
    datasets, failures = {}, {}
    sources = [("naamapadam", "ai4bharat/naamapadam", NAAMAPADAM_LANGS), ("wikiann", "unimelb-nlp/wikiann", WIKIANN_LANGS)]
    for prefix, name, langs in sources:
        if prefix not in which:
            continue
        for lang in langs:
            try:
                examples = [e for e in load_hf(name, lang, limit) if e[0].strip()]
                datasets[f"{prefix}_{lang}"] = {"labels": list(PER_ORG_LOC.values()), "examples": examples}
            except Exception as error:
                failures[f"{prefix}_{lang}"] = str(error)
                print(f"DATASET_FAILED {prefix}_{lang}: {error}", flush=True)
    return datasets, failures


def score(examples, predictions, labels):
    totals = {label: [0, 0, 0] for label in labels}
    for (_, gold), predicted in zip(examples, predictions):
        gold, predicted = set(gold), set(predicted)
        for label in labels:
            g = {s for s in gold if s[0] == label}
            p = {s for s in predicted if s[0] == label}
            totals[label][0] += len(g & p)
            totals[label][1] += len(p)
            totals[label][2] += len(g)

    def prf(tp, npred, ngold):
        precision, recall = tp / npred if npred else 0.0, tp / ngold if ngold else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
                "predicted": npred, "gold": ngold}

    micro = prf(*(sum(v[i] for v in totals.values()) for i in range(3)))
    return {"micro": micro, "per_label": {label: prf(*v) for label, v in totals.items()}}


def to_spans(results, labels):
    return [[(label, e["start"], e["end"]) for label in labels for e in result.get("entities", {}).get(label, [])
             if e.get("start") is not None] for result in results]


def person_errors(examples, predictions, label="person", limit=25):
    missed, false = [], []
    for (text, gold), predicted in zip(examples, predictions):
        g = {s for s in gold if s[0] == label}
        p = {s for s in predicted if s[0] == label}
        for s in sorted(g - p):
            if len(missed) < limit:
                missed.append({"name": text[s[1]:s[2]], "sentence": text,
                               "predicted_here": [f"{q[0]}: {text[q[1]:q[2]]}" for q in predicted
                                                  if q[1] < s[2] and s[1] < q[2]]})
        for s in sorted(p - g):
            if len(false) < limit:
                false.append({"detected": text[s[1]:s[2]], "sentence": text,
                              "gold_here": [f"{q[0]}: {text[q[1]:q[2]]}" for q in gold if q[1] < s[2] and s[1] < q[2]]})
    return {"missed": missed, "false": false}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossner-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoints", default="fastino/gliner2-base-v1,fastino/gliner2-multi-v1")
    parser.add_argument("--setups", default="stock_fp32,stock_fp16,glinert4")
    parser.add_argument("--datasets", default="crossner,naamapadam,wikiann")
    parser.add_argument("--limit", type=int, default=1000, help="max test sentences per dataset")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--standin", action="store_true", help="CPU smoke test with a stand-in model")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "production"))
    sys.path.insert(0, str(ROOT / "production" / "tests"))
    import torch

    import glinert4
    from glinert4 import pipeline, runtime

    which = args.datasets.split(",")
    datasets = load_crossner(args.crossner_dir, args.limit) if "crossner" in which else {}
    public, failures = load_public(args.limit, which) if not args.standin else ({}, {})
    datasets.update(public)
    print("DATASETS " + json.dumps({k: len(v["examples"]) for k, v in datasets.items()}), flush=True)
    device = "cpu" if args.standin else "cuda"

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def run(model, setup, data):
        texts, labels = [e[0] for e in data["examples"]], data["labels"]
        schema = model.create_schema().entities(labels)

        def predict(batch_texts):
            if setup == "glinert4":
                job = {"texts": batch_texts, "schemas": schema, "batch_size": args.batch_size,
                       "threshold": args.threshold, "include_confidence": True, "include_spans": True}
                return pipeline.extract_many(model, [job])[0]
            return model.batch_extract(batch_texts, schema, args.batch_size, args.threshold, 0, True, True, True)

        predict(texts[:2 * args.batch_size])
        sync()
        start = time.perf_counter()
        results = predict(texts)
        sync()
        seconds = time.perf_counter() - start
        return to_spans(results, labels), seconds

    def load(checkpoint, setup):
        if args.standin:
            import standin

            return runtime.freeze_eval(standin.build())
        from gliner2 import AutoExtractor

        if setup == "stock_fp32":
            model = AutoExtractor.from_pretrained(checkpoint, map_location="cuda")
        else:
            model = AutoExtractor.from_pretrained(checkpoint, map_location="cuda", quantize=True,
                                                  attention_backend="disentangled_flash")
        return runtime.freeze_eval(model)

    report = {"device": torch.cuda.get_device_name() if device == "cuda" else "cpu (stand-in)",
              "threshold": args.threshold, "limit": args.limit, "failures": failures, "results": {}}
    order = ["stock_fp32", "stock_fp16", "glinert4"]
    setups = [s for s in order if s in args.setups.split(",")]
    for checkpoint in args.checkpoints.split(","):
        english_only = "base-v1" in checkpoint or "large-v1" in checkpoint
        names = [n for n in datasets if not english_only or n.startswith("crossner") or n == "wikiann_en"]
        predictions = {}
        model, precision = None, None
        for setup in setups:
            glinert4.uninstall()
            wanted = "fp32" if setup == "stock_fp32" else "fp16"
            if model is None or precision != wanted:
                model = None
                if device == "cuda":
                    torch.cuda.empty_cache()
                model, precision = load(checkpoint, setup), wanted
            if setup == "glinert4":
                if device == "cuda":
                    runtime.add_length_buckets(model, 32)
                glinert4.install(kernels=device == "cuda")
            for name in names:
                data = datasets[name]
                spans, seconds = run(model, setup, data)
                predictions[(setup, name)] = spans
                words = sum(len(e[0].split()) for e in data["examples"])
                row = {"checkpoint": checkpoint, "setup": setup, "dataset": name, "sentences": len(spans),
                       "seconds": round(seconds, 3), "sentences_per_s": round(len(spans) / seconds, 1),
                       "words_per_s": round(words / seconds), **score(data["examples"], spans, data["labels"])}
                if setup == "glinert4" and ("stock_fp16", name) in predictions:
                    reference = predictions[("stock_fp16", name)]
                    row["identical_to_stock_fp16"] = sum(a == b for a, b in zip(spans, reference)) / len(spans)
                if setup == "glinert4" and "person" in data["labels"]:
                    row["person_errors"] = person_errors(data["examples"], spans)
                report["results"][f"{checkpoint}|{setup}|{name}"] = row
                print("RESULT " + json.dumps({k: v for k, v in row.items() if k != "person_errors"}), flush=True)
            glinert4.uninstall()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    (out / "public_result.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    write_summary(report, out / "public_summary.md")
    print((out / "public_summary.md").read_text(), flush=True)
    print("PUBLIC_BENCH_DONE", flush=True)


def write_summary(report, path):
    rows = list(report["results"].values())
    lines = []
    for checkpoint in dict.fromkeys(r["checkpoint"] for r in rows):
        mine = [r for r in rows if r["checkpoint"] == checkpoint]
        setups = list(dict.fromkeys(r["setup"] for r in mine))
        lines += [f"### {checkpoint}", "",
                  "| dataset | sentences | " + " | ".join(f"{s} F1" for s in setups) + " | "
                  + " | ".join(f"{s} sent/s" for s in setups) + " | glinert4 speedup vs fp16 / fp32 | identical to fp16 |",
                  "|" + "---|" * (4 + 2 * len(setups))]
        for name in dict.fromkeys(r["dataset"] for r in mine):
            by = {r["setup"]: r for r in mine if r["dataset"] == name}
            f1 = " | ".join(f"{by[s]['micro']['f1']:.4f}" if s in by else "" for s in setups)
            speed = " | ".join(f"{by[s]['sentences_per_s']:.1f}" if s in by else "" for s in setups)
            fast = by.get("glinert4")
            ratio = " / ".join(f"{by[s]['seconds'] / fast['seconds']:.2f}x" for s in ("stock_fp16", "stock_fp32")
                               if fast and s in by)
            same = f"{fast['identical_to_stock_fp16']:.1%}" if fast and "identical_to_stock_fp16" in fast else ""
            lines.append(f"| {name} | {next(iter(by.values()))['sentences']} | {f1} | {speed} | {ratio} | {same} |")
        crossner = [r for r in mine if r["dataset"].startswith("crossner")]
        if crossner:
            lines.append("")
            lines.append("CrossNER average micro F1 (the paper reports 0.590 for GLiNER2): " + ", ".join(
                f"{s} {sum(r['micro']['f1'] for r in crossner if r['setup'] == s) / max(1, sum(r['setup'] == s for r in crossner)):.4f}"
                for s in setups))
        person = [r for r in mine if r["setup"] == "glinert4" and "person" in r["per_label"]]
        if person:
            lines += ["", "Person (glinert4): precision / recall / F1, gold names", "",
                      "| dataset | P | R | F1 | gold |", "|---|---|---|---|---|"]
            for r in person:
                p = r["per_label"]["person"]
                lines.append(f"| {r['dataset']} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['gold']} |")
        lines.append("")
    if report.get("failures"):
        lines += ["Datasets that could not be loaded:", ""] + [f"- {k}: {v[:300]}" for k, v in report["failures"].items()]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
