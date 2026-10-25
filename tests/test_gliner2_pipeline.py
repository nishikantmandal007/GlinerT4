# serving/gliner2_pipeline.py must return exactly what stock gliner2 2.1.0 returns (==, including every confidence
# float, span and order), for single and long-document batch extraction, several batch sizes and two serving threads;
# errors in collation must reach the caller and an abandoned iteration must not leave the helper thread running.
# Runs on the CPU with a random-weight stand-in model (no Hub access needed):
#   python tests/test_gliner2_pipeline.py
import copy
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serving"))
sys.path.insert(0, str(ROOT / "tests"))
import gliner2_pipeline  # noqa: E402
import gliner2_standin  # noqa: E402

LABELS = [{"Person": "people", "Organization": "company"}, {"Street": "street", "City": "town"}]


def run(model, texts, batch_size):
    out = []
    for labels in LABELS:
        out.append(model.batch_extract_entities(texts, labels, batch_size=batch_size, threshold=0.3,
                                                include_confidence=True, include_spans=True))
        out.append(model.batch_extract_entities_long(texts, labels, batch_size=batch_size, threshold=0.3,
                                                     include_confidence=True, include_spans=True,
                                                     chunk_size=96, chunk_overlap=16))
    return out


def count(outputs):
    return sum(len(v) for part in outputs for item in part for v in item.get("entities", {}).values())


model = gliner2_standin.build()
texts = sorted(gliner2_standin.corpus(70, seed=1), key=len)
ok = True
for batch_size in (1, 7, 16):
    gliner2_pipeline.uninstall()
    reference = run(model, texts, batch_size)
    for depth in (1, 2, 4):
        gliner2_pipeline.install(depth)
        same = run(model, texts, batch_size) == reference
        ok &= same
        print(f"batch {batch_size:2}  depth {depth}  entities {count(reference):6}  identical {same}")

# two serving threads, each with its own model copy (as the serving code does), at the same time
gliner2_pipeline.uninstall()
second = copy.deepcopy(model)  # a rebuilt stand-in would train a different tokenizer
reference = run(model, texts, 16)
gliner2_pipeline.install(2)
results = {}
threads = [threading.Thread(target=lambda key, m: results.__setitem__(key, run(m, texts, 16)), args=(key, m))
           for key, m in (("a", model), ("b", second))]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
same = results["a"] == reference and results["b"] == reference
ok &= same
print(f"two concurrent serving threads identical {same}")

# a collation error is raised in the caller, and no helper thread is left behind
processor = model.processor
original = type(processor).collate_fn_inference
calls = [0]


def failing(self, *args, **kwargs):
    calls[0] += 1
    if calls[0] == 3:
        raise ValueError("collation failed")
    return original(self, *args, **kwargs)


type(processor).collate_fn_inference = failing
try:
    model.batch_extract_entities(texts, LABELS[0], batch_size=8)
    raised = False
except ValueError as error:
    raised = str(error) == "collation failed"
finally:
    type(processor).collate_fn_inference = original
ok &= raised
print(f"collation error reaches the caller {raised}")


# a consumer that stops early closes the helper
def slow_batches():
    for i in range(100):
        yield i


prefetch = gliner2_pipeline._Prefetch(slow_batches(), 2)
for item in prefetch:
    if item == 3:
        break
prefetch.close()
left = [t for t in threading.enumerate() if t.name == "gliner2-collate"]
ok &= not left
print(f"no helper threads left {not left}")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
