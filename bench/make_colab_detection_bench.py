# Writes artifacts/colab_detection_bench.py: one self-contained file for `colab run` that measures the shipping
# candidates end to end through the production serving path: ai-utils' detection queue (artifacts/colab_pipeline:
# gliner2_ner.py, detection_queue.py) driven by bench_detection.py in queue mode on one T4, 4 repeats of the
# 290-chunk detbench corpus (1160 chunks, as the reference runs), two runs per configuration. Per configuration it
# prints chunks/s, ground-truth recall/precision per label, GPU busy and memory, and how many chunks return the same
# values as the 3-pass exact reference.
#   python bench/make_colab_detection_bench.py [--runs 2] [--repeat 4]
import argparse
import base64
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=2)
parser.add_argument("--repeat", type=int, default=4)
args = parser.parse_args()

names = ["csrc/torch_binding.cpp", "csrc/disentangled_attention.cu", "csrc/disentangled_attention_v2.cu",
         "csrc/disentangled_attention_v3.cu", "csrc/disentangled_attention_v4.cu", "csrc/add_layernorm.cu",
         "torch-ext/gliner_t4/__init__.py", "torch-ext/gliner_t4/_build.py", "serving/gliner2_exact.py"]
files = {name: (root / name).read_text() for name in names}
pipeline = root / "artifacts/colab_pipeline"
files.update({"app/__init__.py": "", "app/models/__init__.py": "",
              "app/models/gliner2_ner.py": (pipeline / "gliner2_ner.py").read_text(),
              "app/models/detection_queue.py": (pipeline / "detection_queue.py").read_text(),
              "scripts/bench_detection.py": (pipeline / "bench_detection.py").read_text()})
files["corpus.json"] = json.loads((root / "benchmark_results/colab_2026-10-24/payload.json").read_text())["corpus.json"]
blob = base64.b64encode(gzip.compress(json.dumps(files).encode())).decode()

# name: environment on top of the common serving settings; ref_3pass is the accuracy reference (one stream, exact)
CONFIGS = {
    "ref_3pass": {"DETECTION_STREAMS": "1"},
    "prod_3pass": {"DETECTION_STREAMS": "2", "DETECTION_FAST_COUNT": "1"},
    "exact_3pass": {"DETECTION_STREAMS": "2", "DETECTION_EXACT_COUNT": "1", "GLINER_T4_FUSE_LAYERNORM": "1"},
    "exact_2pass": {"DETECTION_STREAMS": "2", "DETECTION_EXACT_COUNT": "1", "GLINER_T4_FUSE_LAYERNORM": "1",
                    "DETECTION_CITY_PASS": "0"},
    "fast_2pass": {"DETECTION_STREAMS": "2", "DETECTION_FAST_COUNT": "1", "GLINER_T4_FUSE_LAYERNORM": "1",
                   "DETECTION_CITY_PASS": "0"},
}

runner = f'''import base64, gzip, json, os, statistics, subprocess, sys
from pathlib import Path
root = Path("/content/gliner-detect")
for name, text in json.loads(gzip.decompress(base64.b64decode("{blob}"))).items():
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gliner2[local]==2.1.0", "disentangled-flash==1.1.1",
                       "ninja"])
print("PACKAGES_READY", flush=True)
subprocess.run(["nvidia-smi"])
subprocess.run(["lscpu"])
COMMON = dict(DETECTION_DEVICES="cuda:0", DETECTION_SPLIT_CACHE="1", DETECTION_SKIP_EVAL="1",
              DETECTION_SHARE_WEIGHTS="1", DETECTION_FAST_COUNT="0", DETECTION_ATTENTION_KERNEL=str(root),
              TOKENIZERS_PARALLELISM="false")
CONFIGS = {json.dumps(CONFIGS)}
results = {{}}
for run in range({args.runs}):
    for name, extra in CONFIGS.items():
        out = root / f"out_{{name}}_{{run}}.json"
        env = dict(os.environ, **COMMON, **extra)
        done = subprocess.run([sys.executable, "scripts/bench_detection.py", "run", "corpus.json", "--mode", "queue",
                               "--repeat", "{args.repeat}", "--out", str(out)], cwd=root, env=env,
                              capture_output=True, text=True)
        if done.returncode:
            print("CONFIG_FAILED", name, done.stderr[-3000:], flush=True)
            continue
        report = json.loads(out.read_text())
        results.setdefault(name, []).append(report)
        print("RUN", json.dumps({{"config": name, "run": run, "chunks_per_s": report["chunks_per_s"],
                                 "accuracy": report["accuracy"], "util": report["mean_gpu_util"],
                                 "serving_gpu_mb": report["serving_gpu_mb"]}}), flush=True)

reference = results["ref_3pass"][0]["found"]
labels = ["Person", "Organization", "Street", "City"]
summary = {{}}
for name, reports in results.items():
    found = reports[0]["found"]
    same = sum(all(set(a.get(l, [])) == set(b.get(l, [])) for l in labels) for a, b in zip(found, reference))
    deterministic = all(r["found"] == found for r in reports)
    summary[name] = {{"env": CONFIGS[name], "chunks_per_s_runs": [r["chunks_per_s"] for r in reports],
                     "chunks_per_s": statistics.median(r["chunks_per_s"] for r in reports),
                     "accuracy": reports[0]["accuracy"], "mean_gpu_util": [r["mean_gpu_util"] for r in reports],
                     "serving_gpu_mb": reports[0]["serving_gpu_mb"], "chunks": reports[0]["chunks"],
                     "identical_chunks_vs_ref": same, "same_across_runs": deterministic}}
base = summary["prod_3pass"]["chunks_per_s"] if "prod_3pass" in summary else None
lines = ["| config | chunks/s | vs prod_3pass | " + " | ".join(l + " R/P" for l in labels) +
         " | identical vs ref | GPU util % | serving MB |"]
lines.append("|" + "---|" * lines[0].count(" | ") + "---|")
for name, row in summary.items():
    acc = row["accuracy"]
    lines.append(f"| {{name}} | {{row['chunks_per_s']:.1f}} | " + (f"{{row['chunks_per_s'] / base:.3f}}x" if base else "") +
                 " | " + " | ".join(f"{{acc[l]['recall']:.3f}}/{{acc[l]['precision']:.3f}}" for l in labels) +
                 f" | {{row['identical_chunks_vs_ref']}}/{{row['chunks']}} | {{row['mean_gpu_util']}} | {{row['serving_gpu_mb']}} |")
print("BEGIN_DETECT_JSON"); print(json.dumps(summary, indent=2)); print("END_DETECT_JSON")
print("BEGIN_DETECT_MD"); print("\\n".join(lines)); print("END_DETECT_MD", flush=True)
'''
target = root / "artifacts" / "colab_detection_bench.py"
target.write_text(runner)
print(target, target.stat().st_size // 1024, "KB")
