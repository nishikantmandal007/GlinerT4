# Writes artifacts/colab_pipeline_bench.py: one self-contained file for `colab run` (which uploads a single file) that
# unpacks the kernel sources, serving/gliner2_pipeline.py, bench/pipeline_bench.py and the 290-chunk corpus into
# /content/gliner-pipe, installs gliner2 2.1.0 + disentangled-flash 1.1.1, runs the benchmark on the T4 and prints
# result.json, summary.md and the baseline outputs' sha256 between markers so they can be copied back from the log.
#   python bench/make_colab_pipeline_bench.py [--trials 3]
import argparse
import base64
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=3)
args = parser.parse_args()

names = ["csrc/torch_binding.cpp", "csrc/disentangled_attention.cu", "csrc/disentangled_attention_v2.cu",
         "csrc/disentangled_attention_v3.cu", "csrc/disentangled_attention_v4.cu", "csrc/add_layernorm.cu",
         "torch-ext/gliner_t4/__init__.py", "torch-ext/gliner_t4/_build.py", "serving/gliner2_pipeline.py",
         "bench/pipeline_bench.py", "benchmark_results/colab_2026-10-24/exact_optimization/exact_prep.py"]
files = {name: (root / name).read_text() for name in names}
# the corpus of the 54.81 chunks/s baseline, as shipped in the 2026-10-24 Colab payload
files["corpus.json"] = json.loads((root / "benchmark_results/colab_2026-10-24/payload.json").read_text())["corpus.json"]
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
subprocess.run([sys.executable, str(root / "bench/pipeline_bench.py"), "--corpus", str(root / "corpus.json"),
                "--out", str(out), "--trials", "{args.trials}"], check=True)
import hashlib
print("BEGIN_RESULT_JSON"); print((out / "result.json").read_text()); print("END_RESULT_JSON")
print("BEGIN_SUMMARY_MD"); print((out / "summary.md").read_text()); print("END_SUMMARY_MD")
print("BASELINE_OUTPUTS_SHA256", hashlib.sha256((out / "baseline_outputs.json").read_bytes()).hexdigest(), flush=True)
'''
target = root / "artifacts" / "colab_pipeline_bench.py"
target.write_text(runner)
print(target, target.stat().st_size // 1024, "KB")
