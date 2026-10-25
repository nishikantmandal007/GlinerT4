#!/usr/bin/env python3
"""Evaluate single-pass GLiNER2 detection speed and precision/recall on realistic test corpus documents.

Usage:
    python bench/test_single_pass_realistic.py [--corpus test_corpus/corpus_040sections.json]
                                              [--layout single_person_first]
                                              [--device cuda|cpu]
                                              [--device-index 0]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts" / "colab_pipeline"))

# Single-pass threshold configurations discovered in Round 7
THRESHOLDS = {
    "single_person_first": {
        "Person": 0.85,
        "Organization": 0.80,
        "Street": 0.30,
        "City": 0.90,
    },
    "single_org_first": {
        "Person": 0.80,
        "Organization": 0.80,
        "Street": 0.30,
        "City": 0.90,
    },
    "single_org_first_city_desc": {
        "Person": 0.85,
        "Organization": 0.90,
        "Street": 0.30,
        "City": 0.80,
    },
}

LAYOUTS = {
    "single_person_first": [
        (["Person", "Organization", "Street", "City"], {})
    ],
    "single_org_first": [
        (["Organization", "Person", "Street", "City"], {})
    ],
    "single_org_first_city_desc": [
        (["Organization", "Person", "Street", "City"], {"City": "Name of a city, town or village."})
    ],
}


def score_predictions(predictions, truth):
    """Compute precision, recall, and detected counts per label."""
    labels = ["Person", "Organization", "Street", "City"]
    scores = {}
    for label in labels:
        found_set = set(predictions.get(label, []))
        true_set = set(truth.get(label, []))

        tp = len(found_set & true_set)
        fp = len(found_set - true_set)
        fn = len(true_set - found_set)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0

        scores[label] = {
            "recall": round(rec, 3),
            "precision": round(prec, 3),
            "detected": len(found_set),
            "true_count": len(true_set),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(ROOT / "test_corpus" / "corpus_040sections.json"),
                        help="Path to corpus JSON file (default: test_corpus/corpus_040sections.json)")
    parser.add_argument("--layout", default="single_person_first", choices=list(LAYOUTS.keys()),
                        help="Single-pass layout to benchmark (default: single_person_first)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--standin", action="store_true", help="Use CPU stand-in model for local testing")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: corpus file {corpus_path} not found.")
        sys.exit(1)

    docs = json.loads(corpus_path.read_text())
    total_chunks = [c for d in docs for c in d["chunks"]]
    total_words = sum(len(c.split()) for c in total_chunks)

    print(f"Loaded {len(docs)} documents ({len(total_chunks)} chunks, {total_words:,} words) from {corpus_path.name}")
    print(f"Layout: {args.layout}")
    print(f"Recalibrated thresholds: {THRESHOLDS[args.layout]}")

    if args.standin or args.device == "cpu":
        print("\n[Notice] Running in stand-in / verification mode without GPU weights.")
        print("To run the full model with CUDA acceleration, execute on a GPU runtime (Colab T4 or VM).")
        # Verify ground truth counts across docs
        tot_truth = {"Person": 0, "Organization": 0, "Street": 0, "City": 0}
        for d in docs:
            for k in tot_truth:
                tot_truth[k] += len(d.get("truth", {}).get(k, []))
        print(f"Ground truth entities across corpus: {tot_truth}")
        return

    # If running on GPU with gliner2
    try:
        import torch
        from gliner2_ner import Gliner2NER
    except ImportError as e:
        print(f"Import error: {e}. gliner2 and pipeline dependencies required.")
        sys.exit(1)

    print(f"\nInitializing model on {args.device}:{args.device_index}...")
    thresholds = THRESHOLDS[args.layout]
    model = Gliner2NER(device=f"{args.device}:{args.device_index}")

    # Warmup
    print("Warming up...")
    _ = model.detect_batch(total_chunks[:min(16, len(total_chunks))],
                           labels=list(thresholds.keys()),
                           thresholds=thresholds)

    # Timed run
    print("Running single-pass extraction...")
    t0 = time.perf_counter()
    results = model.detect_batch(total_chunks,
                                 labels=list(thresholds.keys()),
                                 thresholds=thresholds)
    elapsed = time.perf_counter() - t0

    chunks_per_s = len(total_chunks) / elapsed
    words_per_s = total_words / elapsed
    ms_per_chunk = (elapsed / len(total_chunks)) * 1000

    print(f"\n--- Performance Results ---")
    print(f"Wall time:      {elapsed:.2f} s")
    print(f"Throughput:     {chunks_per_s:.1f} chunks/s ({words_per_s:,.0f} words/s)")
    print(f"Latency:        {ms_per_chunk:.2f} ms/chunk")

    # Evaluate accuracy across documents
    cum_preds = {k: set() for k in thresholds}
    cum_truth = {k: set() for k in thresholds}

    offset = 0
    for doc in docs:
        doc_chunks = doc["chunks"]
        doc_preds = results[offset:offset + len(doc_chunks)]
        offset += len(doc_chunks)
        for chunk_res in doc_preds:
            for label, vals in chunk_res.items():
                if label in cum_preds:
                    cum_preds[label].update(vals)
        for label, vals in doc.get("truth", {}).items():
            if label in cum_truth:
                cum_truth[label].update(vals)

    scores = score_predictions(cum_preds, cum_truth)
    print(f"\n--- Accuracy Results (vs Ground Truth) ---")
    print(f"{'Label':<15} {'Recall':<10} {'Precision':<12} {'Detected':<10} {'Ground Truth':<12}")
    for label, sc in scores.items():
        print(f"{label:<15} {sc['recall']:<10.3f} {sc['precision']:<12.3f} {sc['detected']:<10} {sc['true_count']:<12}")


if __name__ == "__main__":
    main()
