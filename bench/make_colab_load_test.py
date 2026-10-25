# Writes artifacts/colab_load_test.py: one file for `colab run` that unpacks production/ (glinert4), bench/load_test.py
# and the 040-section realistic corpus, builds the ops for the T4 and runs three load tests:
#   1. chunk requests (~131 words), closed loop at 1..128 clients, glinert4 and plain gliner2 fp16, no queue limit
#   2. chunk requests, open loop (Poisson) at fixed arrival rates, glinert4, queue limit 256, 3 retries with backoff
#   3. page requests (500 words), closed loop at 1..64 clients, glinert4
# Each prints its table and JSON between BEGIN_LOAD_<n> / END_LOAD_<n> markers.
#   python bench/make_colab_load_test.py [--duration 30]
import argparse
import base64
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=30.0)
args = parser.parse_args()

paths = [path for path in sorted((root / "production").rglob("*")) if path.is_file() and "__pycache__" not in path.parts]
paths += [root / "bench/load_test.py", root / "bench/realistic_bench.py", root / "test_corpus/corpus_040sections.json"]
for doc in json.loads((root / "test_corpus/corpus_040sections.json").read_text()):
    text = Path(doc["file"].removeprefix("test_corpus/"))
    paths += [root / "test_corpus" / text,
              root / "test_corpus/ground_truth" / text.relative_to("txt").with_name(text.name + ".ground_truth.json")]
files = {str(path.relative_to(root)): base64.b64encode(path.read_bytes()).decode() for path in paths}
blob = base64.b64encode(gzip.compress(json.dumps(files).encode())).decode()

d = args.duration
runs = [
    ["--request", "chunk", "--setups", "glinert4,gliner2_fp16", "--concurrency", "1,2,4,8,16,32,64,128",
     "--duration", str(d)],
    ["--request", "chunk", "--setups", "glinert4", "--concurrency", "", "--rates", "25,50,75,100,125,150,200",
     "--queue-limit", "256", "--retries", "3", "--backoff-ms", "100", "--duration", str(d)],
    ["--request", "page", "--setups", "glinert4", "--concurrency", "1,4,16,64", "--duration", str(d)],
]

runner = f'''import base64, gzip, json, subprocess, sys
from pathlib import Path
root = Path("/content/glinert4-load")
for name, data in json.loads(gzip.decompress(base64.b64decode("{blob}"))).items():
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(data))


def run(command, cwd=root):
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    print(done.stdout + done.stderr, flush=True)
    return done.returncode


run(["nvidia-smi"])
run([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py", str(root / "production")])
print("BUILD rc=" + str(run([sys.executable, "scripts/build_ops.py"], cwd=root / "production")), flush=True)
for number, arguments in enumerate({runs!r}, 1):
    out = root / f"load_{{number}}"
    print(f"BEGIN_RUN_{{number}} " + " ".join(arguments), flush=True)
    code = run([sys.executable, "bench/load_test.py", "--out", str(out), *arguments])
    print(f"END_RUN_{{number}} rc={{code}}", flush=True)
    if (out / "load_test.json").exists():
        print(f"BEGIN_LOAD_{{number}}"); print((out / "load_test.md").read_text()); print((out / "load_test.json").read_text())
        print(f"END_LOAD_{{number}}", flush=True)
'''
target = root / "artifacts" / "colab_load_test.py"
target.write_text(runner)
print(target, target.stat().st_size // 1024, "KB")
