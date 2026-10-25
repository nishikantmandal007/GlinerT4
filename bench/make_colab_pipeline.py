# Writes artifacts/colab_pipeline_run.py: the ai-utils detection pipeline (gliner2_ner.py, detection_queue.py,
# bench_detection.py as on the VM) plus the corpus, unpacked and run in queue mode on a Colab GPU.
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent.parent / "artifacts"
src = here / "colab_pipeline"
files = {"app/__init__.py": "", "app/models/__init__.py": "",
         "app/models/gliner2_ner.py": (src / "gliner2_ner.py").read_text(),
         "app/models/detection_queue.py": (src / "detection_queue.py").read_text(),
         "scripts/bench_detection.py": (src / "bench_detection.py").read_text(),
         "corpus.json": json.dumps(json.load(open(sys.argv[1])))}
runner = f'''import os, subprocess, sys
FILES = {json.dumps(files)}
for path, text in FILES.items():
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write(text)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gliner2[local]==2.1.0", "disentangled-flash==1.1.1"])
env = dict(os.environ, DETECTION_STRATEGY="grouped", DETECTION_FAST_COUNT="1", DETECTION_SPLIT_CACHE="1",
           DETECTION_SKIP_EVAL="1", DETECTION_SHARE_WEIGHTS="1", DETECTION_DEVICES="cuda:0")
for streams in ("1", "2"):
    env["DETECTION_STREAMS"] = streams
    subprocess.run([sys.executable, "scripts/bench_detection.py", "run", "corpus.json", "--repeat", "4", "--mode", "queue",
                    "--out", f"out_{{streams}}.json"], env=env, check=True, stdout=subprocess.DEVNULL)
    import json as j
    r = j.load(open(f"out_{{streams}}.json"))
    print("RESULT", j.dumps({{"streams": streams, "chunks_per_s": r["chunks_per_s"], "serving_gpu_mb": r["serving_gpu_mb"],
                              "util": r["mean_gpu_util"], "accuracy": r["accuracy"]}}))
'''
(here / "colab_pipeline_run.py").write_text(runner)
print(here / "colab_pipeline_run.py", (here / "colab_pipeline_run.py").stat().st_size // 1024, "KB")
