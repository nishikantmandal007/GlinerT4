# Writes artifacts/colab_public_bench.py: one file for `colab run` that unpacks production/ (glinert4) and
# bench/public_bench.py, downloads CrossNER's five test sets from GitHub (zliucr/CrossNER), installs gliner2 2.1.0,
# disentangled-flash 1.1.1 and datasets, and runs the public benchmark (CrossNER, Naamapadam, WikiANN; one pass per
# dataset, stock fp32 / stock fp16 / glinert4) on the T4. Results print between BEGIN/END markers.
#   python bench/make_colab_public_bench.py [--limit 1000] [--datasets crossner,naamapadam,wikiann]
import argparse
import base64
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=1000)
parser.add_argument("--datasets", default="crossner,naamapadam,wikiann")
parser.add_argument("--checkpoints", default="fastino/gliner2-base-v1,fastino/gliner2-multi-v1")
parser.add_argument("--setups", default="stock_fp32,stock_fp16,glinert4")
args = parser.parse_args()
files = {str(path.relative_to(root)): path.read_text() for path in sorted((root / "production").rglob("*"))
         if path.is_file() and "__pycache__" not in path.parts}
files["bench/public_bench.py"] = (root / "bench/public_bench.py").read_text()
blob = base64.b64encode(gzip.compress(json.dumps(files).encode())).decode()

runner = f'''import base64, gzip, json, subprocess, sys, urllib.request
from pathlib import Path
root = Path("/content/glinert4-public")
for name, text in json.loads(gzip.decompress(base64.b64decode("{blob}"))).items():
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
crossner = root / "crossner"
crossner.mkdir(exist_ok=True)
for domain in ("ai", "literature", "music", "politics", "science"):
    url = f"https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/{{domain}}/test.txt"
    (crossner / f"{{domain}}.txt").write_bytes(urllib.request.urlopen(url).read())
subprocess.run(["nvidia-smi"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gliner2[local]==2.1.0", "disentangled-flash==1.1.1",
                       "ninja", "datasets", "huggingface_hub"])
print("PACKAGES_READY", flush=True)
subprocess.run([sys.executable, "bench/public_bench.py", "--crossner-dir", str(crossner), "--out", "results",
                "--limit", "{args.limit}", "--datasets", "{args.datasets}", "--checkpoints", "{args.checkpoints}",
                "--setups", "{args.setups}"],
               cwd=root, check=True)
print("BEGIN_PUBLIC_JSON"); print((root / "results/public_result.json").read_text()); print("END_PUBLIC_JSON")
print("BEGIN_PUBLIC_MD"); print((root / "results/public_summary.md").read_text()); print("END_PUBLIC_MD", flush=True)
'''
target = root / "artifacts" / "colab_public_bench.py"
target.write_text(runner)
print(target, target.stat().st_size // 1024, "KB")
