# Writes artifacts/colab_gpu_compare.py: gpu_compare.py with the corpus embedded, since `colab run` uploads one file.
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
corpus = json.load(open(sys.argv[1]))
source = (here / "gpu_compare.py").read_text()
target = here.parent / "artifacts" / "colab_gpu_compare.py"
target.write_text(f"EMBEDDED_CORPUS = {json.dumps(json.dumps(corpus))}\n" + source)
print(target, target.stat().st_size // 1024, "KB")
