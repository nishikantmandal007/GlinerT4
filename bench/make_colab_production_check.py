# Writes artifacts/colab_production_check.py: one file for `colab run` that installs production/ (the glinert4 package),
# builds its CUDA ops for the T4, runs its test suite (CPU parity and GPU kernel tests), and runs glinert4.Detector on
# the realistic corpus (whole documents, its defaults), scored against the ground truth as bench/realistic_bench.py
# scores it. The round-9 reference for the same setting: corpus 040 in 2.60 s, Person 1.000/.997, Organization
# 1.000/.992, Street .871/1.000, City 1.000/.994 (doc-set recall/precision).
#   python bench/make_colab_production_check.py
import base64
import gzip
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = {str(path.relative_to(root)): path.read_text() for path in sorted((root / "production").rglob("*"))
         if path.is_file() and "__pycache__" not in path.parts}
for name in ("bench/realistic_bench.py", "artifacts/colab_pipeline/bench_detection.py"):
    files[name] = (root / name).read_text()
spec = importlib.util.spec_from_file_location("realistic_bench", root / "bench/realistic_bench.py")
realistic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(realistic)
files["realistic.json"] = json.dumps({size: realistic.prepare(root / "test_corpus", size) for size in ("040", "020")},
                                     ensure_ascii=False)
blob = base64.b64encode(gzip.compress(json.dumps(files).encode())).decode()

check = r'''
import importlib.util, json, statistics, sys, time
from pathlib import Path
import torch
root = Path("/content/glinert4-check")
spec = importlib.util.spec_from_file_location("realistic_bench", root / "bench/realistic_bench.py")
realistic = importlib.util.module_from_spec(spec); spec.loader.exec_module(realistic)
sys.path.insert(0, str(root / "artifacts"))
from colab_pipeline import bench_detection
from glinert4 import Detector
detector = Detector()
labels = list(detector.config.labels)
report = {"device": torch.cuda.get_device_name()}
for size, docs in json.loads((root / "realistic.json").read_text()).items():
    docs = [{**doc, "chunks": [doc["text"]], "chunk_offsets": [0], "position_map": None} for doc in docs]
    texts = [doc["text"] for doc in docs]
    words = sum(len(text.split()) for text in texts)
    detector.detect(texts)
    seconds = []
    for _ in range(3):
        torch.cuda.synchronize(); start = time.perf_counter()
        found = detector.detect(texts)
        torch.cuda.synchronize(); seconds.append(time.perf_counter() - start)
    median = statistics.median(seconds)
    errors = realistic.error_analysis(docs, found, labels)
    report[size] = {"seconds": seconds, "median_seconds": median, "words": words, "words_per_s": words / median,
                    "doc_set": bench_detection.score(docs, [f["values"] for f in found]),
                    "occurrence": realistic.occurrence_scores(docs, found, labels),
                    "errors": {l: {k: len(v) for k, v in kinds.items()} for l, kinds in errors.items()}}
    print("PRODUCTION " + json.dumps({"corpus": size, **report[size]}), flush=True)
print("BEGIN_PRODUCTION_JSON"); print(json.dumps(report, indent=2)); print("END_PRODUCTION_JSON", flush=True)
'''

runner = f'''import base64, gzip, json, subprocess, sys
from pathlib import Path
root = Path("/content/glinert4-check")
for name, text in json.loads(gzip.decompress(base64.b64decode("{blob}"))).items():
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def run(command, cwd=root):
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    print(done.stdout + done.stderr, flush=True)
    return done.returncode


run(["nvidia-smi"])
run([sys.executable, "-m", "pip", "install", "-q", str(root / "production") + "[test]"])
print("BUILD rc=" + str(run([sys.executable, "scripts/build_ops.py"], cwd=root / "production")), flush=True)
print("BEGIN_PYTEST", flush=True)
code = run([sys.executable, "-m", "pytest", "-v", "-rs", "-p", "no:cacheprovider", "tests"], cwd=root / "production")
print("END_PYTEST rc=" + str(code), flush=True)
(root / "check.py").write_text({check!r})
print("CHECK rc=" + str(run([sys.executable, str(root / "check.py")])), flush=True)
'''
target = root / "artifacts" / "colab_production_check.py"
target.write_text(runner)
print(target, target.stat().st_size // 1024, "KB")
