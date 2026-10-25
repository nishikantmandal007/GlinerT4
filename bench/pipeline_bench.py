# One-T4 GLiNER2 2.1.0 extraction throughput: the 54.81 chunks/s baseline path against no-semantic-change pipeline
# options (collation prefetch on a helper thread, DataLoader worker processes, batch size), with full output parity
# against the baseline and a per-stage time breakdown of the baseline.
#
# Baseline = benchmark_results/colab_2026-10-24/t4_optimization/baseline.py: fastino/gliner2-multi-v1, fp16,
# disentangled_flash with the sm_75 custom attention kernel (gliner_t4), word-split cache, eval() once, exact bulk
# token-id lookup; the 290 corpus chunks sorted by length; three label passes; batch 16; threshold 0.65.
#   python bench/pipeline_bench.py --corpus corpus.json --out results_dir [--trials 3]
# --standin runs the same harness on the CPU with tests/gliner2_standin.py (no GPU, no Hub) to check the script itself.
import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))
sys.path.insert(0, str(ROOT / "serving"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "benchmark_results" / "colab_2026-10-24" / "exact_optimization"))

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", required=True, help="corpus.json: a list of documents, each with a 'chunks' list")
parser.add_argument("--out", required=True)
parser.add_argument("--trials", type=int, default=3)
parser.add_argument("--configs", default="", help="comma-separated config names, or 'all'; default: the exact set")
parser.add_argument("--standin", action="store_true", help="CPU smoke test of this script with a stand-in model")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

report = {}


def emit(label, value):
    print(label + " " + json.dumps(value), flush=True)


def save():
    (out / "result.json").write_text(json.dumps(report, indent=2))


CUDA = not args.standin


def sync():
    if CUDA:
        torch.cuda.synchronize()


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
report["env"] = {
    "gpu": torch.cuda.get_device_name() if CUDA else "cpu (stand-in)",
    "capability": torch.cuda.get_device_capability() if CUDA else None,
    "versions": {p: version(p) for p in ("torch", "triton", "gliner2", "disentangled-flash", "transformers")},
    "cuda": torch.version.cuda, "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
    "lscpu": subprocess.run(["lscpu"], capture_output=True, text=True).stdout.strip() or None,
    "git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip() or None,
}
emit("ENV", report["env"])
if CUDA:
    if tuple(report["env"]["capability"]) != (7, 5):
        print("WARNING: not a T4 (sm_75); the custom kernel is built for sm_75 only", flush=True)
    subprocess.run(["nvidia-smi"], check=False)

import gliner2_exact  # noqa: E402
import gliner2_pipeline  # noqa: E402
from gliner2.processing.word_splitter import WhitespaceTokenSplitter  # noqa: E402
from gliner2.training.trainer import ExtractorCollator  # noqa: E402
from exact_prep import install_bulk_token_id_lookup  # noqa: E402

calls = [0]
if CUDA:
    import gliner_t4  # noqa: E402  (JIT-builds the sm_75 kernel on first import)
    from gliner2 import AutoExtractor  # noqa: E402
    from disentangled_flash import kernel  # noqa: E402

    gliner_t4.install()
    _attention = gliner_t4.attention

    def counted(*a, **k):
        calls[0] += 1
        return _attention(*a, **k)

    gliner_t4.attention = counted
docs = json.loads(Path(args.corpus).read_text())
chunks = sorted([s for d in docs for s in d["chunks"]], key=len)
report["corpus"] = {"chunks": len(chunks), "sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()}



def load_model(first=None):
    if CUDA:
        loaded = AutoExtractor.from_pretrained("fastino/gliner2-multi-v1", map_location="cuda", quantize=True,
                                               attention_backend="disentangled_flash")
        loaded.eval()
        attention = [m for m in loaded.encoder.modules() if isinstance(m, kernel.InferenceDisentangledSelfAttention)]
        assert len(attention) == 12 and all(m.backend != "torch" for m in attention), "fused attention did not activate"
    else:
        import copy
        import gliner2_standin  # noqa: E402

        loaded = gliner2_standin.build() if first is None else copy.deepcopy(first)
    if first is not None:
        # as the serving code does (DETECTION_SHARE_WEIGHTS): a second serving copy reads the first copy's fp16 weights
        pairs = list(zip(loaded.named_parameters(), first.named_parameters()))
        assert all(a[0] == b[0] and a[1].shape == b[1].shape for a, b in pairs)
        for (_, mine), (_, theirs) in pairs:
            mine.data = theirs.data
        if CUDA:
            torch.cuda.empty_cache()
    loaded.eval = types.MethodType(lambda self: self, loaded)
    report.setdefault("bulk_lookup_installed", []).append(install_bulk_token_id_lookup(loaded.processor.tokenizer))
    return loaded


model = load_model()

# the baseline's exact preparation (as in t4_optimization/baseline.py)
_split = WhitespaceTokenSplitter.__call__


@functools.lru_cache(maxsize=1024)
def _split_cached(text, lower):
    return tuple(_split(WhitespaceTokenSplitter(), text, lower))


def _split_call(self, text, lower=True):
    if type(self) is not WhitespaceTokenSplitter:
        return _split(self, text, lower)
    return iter(_split_cached(text, lower))


WhitespaceTokenSplitter.__call__ = _split_call

labels = {
    "Person": "First and last names of people. NOT company names, street addresses, cities, or identifier codes.",
    "Organization": "Company, corporation, institution, agency, or legal entity name. NOT person names, street "
                    "addresses, cities, or identifier codes.",
    "Street": "Street, road, avenue, boulevard, or lane name. NOT city names, building names, or postal codes.",
    "City": "City"}
passes = [{"Organization": labels["Organization"]}, {k: labels[k] for k in ("Person", "Street", "City")},
          {"City": "Name of a city, town or village."}]



def config(depth=0, workers=0, batch=16, count0=False, fmt=False, streams=1):
    return {"prefetch_depth": depth, "num_workers": workers, "batch_size": batch, "count_step0": count0,
            "format_cache": fmt, "streams": streams}


# depth 0, no exact flags, one stream is stock gliner2 (plus the baseline preparation above)
EXACT_CONFIGS = {
    "baseline": config(),
    "count0": config(count0=True),
    "fmtcache": config(fmt=True),
    "exact": config(count0=True, fmt=True),
    "exact_prefetch1": config(depth=1, count0=True, fmt=True),
    "streams2": config(streams=2),
    "exact_streams2": config(count0=True, fmt=True, streams=2),
}
# tested on 2026-10-25 (none beat the baseline on Colab's 2 vCPUs); kept for hosts with more cores
EARLIER_CONFIGS = {
    "prefetch1": config(depth=1),
    "prefetch2": config(depth=2),
    "workers1": config(workers=1),
    "workers2": config(workers=2),
    "batch8": config(batch=8),
    "batch32": config(batch=32),
}
if args.configs == "all":
    CONFIGS = {**EXACT_CONFIGS, **EARLIER_CONFIGS}
elif args.configs:
    every = {**EXACT_CONFIGS, **EARLIER_CONFIGS}
    CONFIGS = {k: every[k] for k in ["baseline"] + [n for n in args.configs.split(",") if n != "baseline"]}
else:
    CONFIGS = EXACT_CONFIGS

second_model = load_model(first=model) if any(c["streams"] > 1 for c in CONFIGS.values()) else None
streams = [torch.cuda.Stream() for _ in range(2)] if CUDA else [None, None]


def configure(name):
    c = CONFIGS[name]
    if c["prefetch_depth"]:
        gliner2_pipeline.install(c["prefetch_depth"])
    else:
        gliner2_pipeline.uninstall()
    if c["count_step0"] or c["format_cache"]:
        gliner2_exact.install(count_step0=c["count_step0"], format_cache=c["format_cache"])
    else:
        gliner2_exact.uninstall()
    for m in (model, second_model):
        if m is not None:
            m.processor.__dict__.pop("_exact_text_parts", None)  # each timed run starts with a cold text cache


def schema(extractor, labels_of_pass):
    return extractor.create_schema().entities(labels_of_pass)


def extract_two_streams(data, batch_size):
    # two serving threads, each with its own model copy (shared weights) on its own CUDA stream, as DETECTION_STREAMS=2
    # serves: the same batches of batch_size as the one-stream loop, handed out in order, results put back in order
    import queue
    import threading

    work = queue.Queue()
    for p_index in range(len(passes)):
        for start in range(0, len(data), batch_size):
            work.put((p_index, start))
    results, errors = {}, []

    def serve(extractor, stream):
        try:
            with torch.inference_mode(), (torch.cuda.stream(stream) if CUDA else contextlib.nullcontext()):
                while True:
                    try:
                        p_index, start = work.get_nowait()
                    except queue.Empty:
                        break
                    results[(p_index, start)] = extractor.batch_extract(
                        data[start:start + batch_size], schema(extractor, passes[p_index]), batch_size, 0.65, 0,
                        True, True, True)
                if CUDA:
                    stream.synchronize()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=serve, args=(m, s)) for m, s in zip((model, second_model), streams)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return [[item for start in range(0, len(data), batch_size) for item in results[(p_index, start)]]
            for p_index in range(len(passes))]


@torch.inference_mode()
def extract(name, data=chunks):
    configure(name)
    c = CONFIGS[name]
    _split_cached.cache_clear()
    sync()
    start = time.perf_counter()
    if c["streams"] > 1:
        outputs = extract_two_streams(data, c["batch_size"])
    else:
        outputs = [model.batch_extract(data, schema(model, p), c["batch_size"], 0.65, c["num_workers"], True, True,
                                       True) for p in passes]
    sync()
    return time.perf_counter() - start, outputs


def kept_values(outputs):
    # the label-level view the serving code keeps: values above each label's threshold
    merged = [dict() for _ in chunks]
    for part in outputs:
        for i, item in enumerate(part):
            for label, values in item.get("entities", {}).items():
                minimum = 0.85 if label == "Organization" else 0.65
                for value in values:
                    if value.get("confidence", 1) >= minimum and value.get("text"):
                        merged[i].setdefault(label, set()).add(value["text"])
    return merged


# warm-up of every config (kernel plans for every padded length each batch size produces), and the reference
results = {name: {"seconds": [], "errors": []} for name in CONFIGS}
outputs_by = {}
for name in CONFIGS:
    try:
        calls[0] = 0
        extract(name)
        assert calls[0] > 0 or not CUDA, "custom attention kernel was not called"
        _, outputs_by[name] = extract(name)
    except Exception as error:  # a config that cannot run here is reported, not fatal
        results[name]["errors"].append(repr(error))
        emit("CONFIG_FAILED", {"config": name, "error": repr(error)})
reference = outputs_by["baseline"]
(out / "baseline_outputs.json").write_text(json.dumps(reference))
reference_values = kept_values(reference)

for trial in range(args.trials):
    names = [n for n in CONFIGS if n in outputs_by]
    names = names[trial % len(names):] + names[:trial % len(names)]  # rotate the order each trial
    for name in names:
        elapsed, outputs = extract(name)
        results[name]["seconds"].append(elapsed)
        if outputs != reference and name in CONFIGS:
            outputs_by[name] = outputs  # keep a differing run for the parity report
        emit("EXTRACTION", {"trial": trial, "config": name, "seconds": elapsed, "chunks_per_s": len(chunks) / elapsed})

base_median = statistics.median(results["baseline"]["seconds"])
for name, row in results.items():
    row.update(CONFIGS[name])
    if not row["seconds"]:
        continue
    median = statistics.median(row["seconds"])
    values = kept_values(outputs_by[name])
    row.update(median_seconds=median, chunks_per_s=len(chunks) / median, speedup_vs_baseline=base_median / median,
               identical_full_output=outputs_by[name] == reference,
               identical_kept_values_chunks=sum(a == b for a, b in zip(values, reference_values)))
report["configs"] = results
save()


# per-stage breakdown of one stock run (host syncs added around each stage, so the stages are serialized; the
# total is a little above the timed baseline)
def stage_breakdown(name):
    configure(name)
    stages = {k: 0.0 for k in ("collate", "to_device", "encoder", "embeddings", "span_head", "decode", "format")}

    def timed(stage, fn, barrier):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            if barrier:
                sync()
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                if barrier:
                    sync()
                stages[stage] += time.perf_counter() - t
        return wrapper

    configure(name)
    cls = type(model)
    patched = {
        (ExtractorCollator, "__call__"): timed("collate", ExtractorCollator.__call__, False),
        (cls, "encode_tokens"): timed("encoder", cls.encode_tokens, True),
        (cls, "_compute_direct_span_logits_batched"): timed("span_head", cls._compute_direct_span_logits_batched, True),
        (cls, "_extract_from_batch_sync_collapsed"): timed("decode", cls._extract_from_batch_sync_collapsed, True),
        (cls, "format_results"): timed("format", cls.format_results, False),
    }
    from gliner2.processor import PreprocessedBatch, SchemaTransformer
    patched[(PreprocessedBatch, "to")] = timed("to_device", PreprocessedBatch.to, True)
    patched[(SchemaTransformer, "extract_embeddings_from_batch")] = timed(
        "embeddings", SchemaTransformer.extract_embeddings_from_batch, True)
    missing = object()
    # restore exactly what each class itself held: re-setting an inherited method on a subclass would shadow later
    # patches of its base (gliner2_exact patches the mixin)
    originals = {key: key[0].__dict__.get(key[1], missing) for key in patched}
    try:
        for (owner, attr), fn in patched.items():
            setattr(owner, attr, fn)
        total, _ = extract(name)
    finally:
        for (owner, attr), fn in originals.items():
            if fn is missing:
                delattr(owner, attr)
            else:
                setattr(owner, attr, fn)
    stages["other"] = total - sum(stages.values())
    return {"total_seconds": total, "stages_seconds": stages,
            "stages_ms_per_chunk": {k: 1000 * v / len(chunks) for k, v in stages.items()}}


report["breakdown_baseline"] = stage_breakdown("baseline")
emit("BREAKDOWN", report["breakdown_baseline"])
if "exact" in CONFIGS:
    report["breakdown_exact"] = stage_breakdown("exact")
    emit("BREAKDOWN_EXACT", report["breakdown_exact"])
save()


# GPU busy fraction under the profiler: total CUDA kernel time over wall time, baseline and best prefetch config
def busy(name):
    configure(name)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        wall, _ = extract(name)
    kernel_us = sum(device_us(e) for e in prof.key_averages())
    return {"wall_seconds": wall, "kernel_seconds": kernel_us / 1e6, "gpu_busy_fraction": kernel_us / 1e6 / wall,
            "top_kernels": [(e.key[:80], device_us(e) / 1e3) for e in
                            sorted(prof.key_averages(), key=lambda e: -device_us(e))[:12]]}


def device_us(event):
    return getattr(event, "self_device_time_total", None) or getattr(event, "self_cuda_time_total", 0)


best = max((n for n in results if results[n].get("identical_full_output") and n != "baseline"),
           key=lambda n: results[n]["chunks_per_s"], default=None)
if CUDA:
    report["gpu_busy"] = {"baseline": busy("baseline")}
    if best:
        report["gpu_busy"][best] = busy(best)
    emit("GPU_BUSY", {k: {kk: vv for kk, vv in v.items() if kk != "top_kernels"}
                      for k, v in report["gpu_busy"].items()})
    report["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
gliner2_pipeline.uninstall()
save()

lines = ["| config | count0 | fmtcache | streams | depth | workers | batch | chunks/s | vs baseline | identical full output "
         "| identical kept values |", "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
for name, row in results.items():
    shape = (f"| {name} | {row['count_step0']} | {row['format_cache']} | {row['streams']} | {row['prefetch_depth']} | "
             f"{row['num_workers']} | {row['batch_size']} | ")
    if "chunks_per_s" not in row:
        lines.append(shape + f"failed: {row['errors'][:1]} | | | |")
        continue
    lines.append(shape + f"{row['chunks_per_s']:.2f} | {row['speedup_vs_baseline']:.3f}x | {row['identical_full_output']} | "
                 f"{row['identical_kept_values_chunks']}/{len(chunks)} |")
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
emit("RESULT", {k: v for k, v in report.items() if k != "gpu_busy"})
print("PIPELINE_BENCH_DONE", flush=True)
