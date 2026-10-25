# Writes artifacts/colab_memory_bench.py: one file for `colab run` that unpacks production/ (glinert4 with the document
# memory), bench/memory_bench.py with its scorers, the realistic corpus (040 and 020 sections with ground truth) and the
# synthetic 100-page contract with its manifest, builds the ops for the T4, runs the production tests, then
# bench/memory_bench.py (contract, realistic corpus, and the corpus joined into one ~76-page document at x1, x4 and x10),
# and prints its table and JSON between BEGIN_MEMORY / END_MEMORY.
#   python bench/make_colab_memory_bench.py [--floor 0.3]
import argparse
import base64
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--floor", type=float, default=0.3)
args = parser.parse_args()

paths = [path for path in sorted((root / "production").rglob("*")) if path.is_file() and "__pycache__" not in path.parts]
paths += [root / name for name in ("bench/memory_bench.py", "bench/doc_eval.py", "bench/realistic_bench.py",
                                    "notebooks/samples/02_service_contract_100p.docx",
                                    "notebooks/samples/02_service_contract_100p.docx.manifest.json")]
for size in ("040", "020"):
    corpus = root / f"test_corpus/corpus_{size}sections.json"
    paths.append(corpus)
    for doc in json.loads(corpus.read_text()):
        text = Path(doc["file"].removeprefix("test_corpus/"))
        paths += [root / "test_corpus" / text,
                  root / "test_corpus/ground_truth" / text.relative_to("txt").with_name(text.name + ".ground_truth.json")]
files = {str(path.relative_to(root)): base64.b64encode(path.read_bytes()).decode() for path in dict.fromkeys(paths)}
blob = base64.b64encode(gzip.compress(json.dumps(files).encode())).decode()

runner = f'''import base64, gzip, json, subprocess, sys
from pathlib import Path
root = Path("/content/glinert4-memory")
for name, data in json.loads(gzip.decompress(base64.b64decode("{blob}"))).items():
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(data))


def run(command, cwd=root):
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    print(done.stdout + done.stderr, flush=True)
    return done.returncode


run(["nvidia-smi"])
run([sys.executable, "-m", "pip", "install", "-q", "python-docx", str(root / "production") + "[test]"])
print("BUILD rc=" + str(run([sys.executable, "scripts/build_ops.py"], cwd=root / "production")), flush=True)
print("PYTEST rc=" + str(run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "tests"],
                             cwd=root / "production")), flush=True)
code = run([sys.executable, "bench/memory_bench.py", "--out", str(root / "memory"), "--floor", "{args.floor}"])
print("BENCH rc=" + str(code), flush=True)
if (root / "memory/memory_bench.md").exists():
    print("BEGIN_MEMORY"); print((root / "memory/memory_bench.md").read_text())
    print((root / "memory/memory_bench.json").read_text()); print("END_MEMORY", flush=True)
'''
target = root / "artifacts" / "colab_memory_bench.py"
target.write_text(runner)
print(target, target.stat().st_size // 1024, "KB")
