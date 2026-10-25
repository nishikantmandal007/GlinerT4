# Writes artifacts/colab_pipeline_bench.py: one self-contained file for `colab run` (which uploads a single file) that
# unpacks the kernel sources, serving modules, the benchmarks, the production detection code and the 290-chunk corpus
# into /content/gliner-pipe, installs gliner2 2.1.0 + disentangled-flash 1.1.1, runs bench/pipeline_bench.py and
# bench/pass_bench.py on the T4 and prints their result files between markers so they can be copied back from the log.
#   python bench/make_colab_pipeline_bench.py [--trials 5] [--pass-trials 2] [--only pipeline|pass]
import argparse
import base64
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=5)
parser.add_argument("--pass-trials", type=int, default=2)
parser.add_argument("--only", choices=["pipeline", "pass", "single", "realistic"], default=None)
args = parser.parse_args()

names = ["csrc/torch_binding.cpp", "csrc/disentangled_attention.cu", "csrc/disentangled_attention_v2.cu",
         "csrc/disentangled_attention_v3.cu", "csrc/disentangled_attention_v4.cu", "csrc/add_layernorm.cu",
         "torch-ext/gliner_t4/__init__.py", "torch-ext/gliner_t4/_build.py", "serving/gliner2_pipeline.py",
         "serving/gliner2_exact.py", "serving/gliner2_pipelined.py", "bench/pass_bench.py",
         "bench/single_pass_bench.py", "bench/realistic_bench.py", "artifacts/colab_pipeline/gliner2_ner.py",
         "artifacts/colab_pipeline/detection_queue.py", "artifacts/colab_pipeline/bench_detection.py",
         "bench/pipeline_bench.py", "benchmark_results/colab_2026-10-24/exact_optimization/exact_prep.py"]
files = {name: (root / name).read_text() for name in names}
# the corpus of the 54.81 chunks/s baseline, as shipped in the 2026-10-24 Colab payload
files["corpus.json"] = json.loads((root / "benchmark_results/colab_2026-10-24/payload.json").read_text())["corpus.json"]
if args.only == "realistic":
    # the realistic test corpus, prepared (chunk offsets, annotated occurrences) so the runner needs no test_corpus/
    import importlib.util

    spec = importlib.util.spec_from_file_location("realistic_bench", root / "bench/realistic_bench.py")
    realistic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(realistic)
    files["realistic.json"] = json.dumps({size: realistic.prepare(root / "test_corpus", size) for size in ("040", "020")},
                                         ensure_ascii=False)
blob = base64.b64encode(gzip.compress(json.dumps(files).encode())).decode()

runner = f'''import base64, gzip, json, os, subprocess, sys
from pathlib import Path
root = Path("/content/gliner-pipe")
for name, text in json.loads(gzip.decompress(base64.b64decode("{blob}"))).items():
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gliner2[local]==2.1.0", "disentangled-flash==1.1.1",
                       "ninja"])
print("PACKAGES_READY", flush=True)
out = root / "results"
if {args.only!r} in (None, "pipeline"):
    subprocess.run([sys.executable, str(root / "bench/pipeline_bench.py"), "--corpus", str(root / "corpus.json"),
                    "--out", str(out), "--trials", "{args.trials}"], check=True)
    import hashlib
    print("BEGIN_RESULT_JSON"); print((out / "result.json").read_text()); print("END_RESULT_JSON")
    print("BEGIN_SUMMARY_MD"); print((out / "summary.md").read_text()); print("END_SUMMARY_MD")
    print("BASELINE_OUTPUTS_SHA256", hashlib.sha256((out / "baseline_outputs.json").read_bytes()).hexdigest(), flush=True)
if {args.only!r} == "realistic":
    subprocess.run([sys.executable, str(root / "bench/realistic_bench.py"), "--prepared", str(root / "realistic.json"),
                    "--out", str(out)], check=True)
    print("BEGIN_REALISTIC_JSON"); print((out / "realistic_result.json").read_text()); print("END_REALISTIC_JSON")
    print("BEGIN_REALISTIC_MD"); print((out / "realistic_summary.md").read_text()); print("END_REALISTIC_MD", flush=True)
if {args.only!r} == "single":
    subprocess.run([sys.executable, str(root / "bench/single_pass_bench.py"), "--corpus", str(root / "corpus.json"),
                    "--out", str(out)], check=True)
    print("BEGIN_SINGLE_JSON"); print((out / "single_result.json").read_text()); print("END_SINGLE_JSON")
    print("BEGIN_SINGLE_MD"); print((out / "single_summary.md").read_text()); print("END_SINGLE_MD", flush=True)
if {args.only!r} in (None, "pass"):
    subprocess.run([sys.executable, str(root / "bench/pass_bench.py"), "--corpus", str(root / "corpus.json"),
                    "--out", str(out), "--trials", "{args.pass_trials}"], check=True)
    print("BEGIN_PASS_JSON"); print((out / "pass_result.json").read_text()); print("END_PASS_JSON")
    print("BEGIN_PASS_MD"); print((out / "pass_summary.md").read_text()); print("END_PASS_MD", flush=True)
'''
target = root / "artifacts" / "colab_pipeline_bench.py"
target.write_text(runner)
print(target, target.stat().st_size // 1024, "KB")
