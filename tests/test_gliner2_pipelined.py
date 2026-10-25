# serving/gliner2_pipelined.py must return exactly what stock gliner2 2.1.0 returns (==, every confidence float, span
# and order): extract_many against one batch_extract per job, extract_entities_long_many against one
# batch_extract_entities_long per request, for several batch sizes, jobs of one batch and of many, with and without the
# exact count head. CPU, random-weight stand-in model (no streams on the CPU: this checks the reordered work):
#   python tests/test_gliner2_pipelined.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serving"))
sys.path.insert(0, str(ROOT / "tests"))
import gliner2_exact  # noqa: E402
import gliner2_pipelined  # noqa: E402
import gliner2_standin  # noqa: E402

PASSES = [{"Organization": "company"}, {"Person": "people", "Street": "street", "City": "town"}, {"City": "a town"}]
model = gliner2_standin.build()
texts = sorted(gliner2_standin.corpus(45, seed=4), key=len)
ok = True
for exact in (False, True):
    if exact:
        gliner2_exact.install(count_step0=True, format_cache=False)
    for batch_size in (16, 7, 64):
        for subset in (texts[:16], texts):  # one batch per job (a queue run) and many
            jobs = [{"texts": subset, "schemas": model.create_schema().entities(p), "batch_size": batch_size,
                     "threshold": 0.3, "include_confidence": True, "include_spans": True} for p in PASSES]
            stock = [model.batch_extract(subset, j["schemas"], batch_size, 0.3, 0, True, True, True) for j in jobs]
            same = gliner2_pipelined.extract_many(model, jobs) == stock
            stock_long = [model.batch_extract_entities_long(subset, p, batch_size=batch_size, threshold=0.3,
                                                            include_confidence=True, include_spans=True,
                                                            chunk_size=96, chunk_overlap=16) for p in PASSES]
            same_long = gliner2_pipelined.extract_entities_long_many(
                model, [(subset, p) for p in PASSES], batch_size=batch_size, threshold=0.3, include_confidence=True,
                include_spans=True, chunk_size=96, chunk_overlap=16) == stock_long
            ok &= same and same_long
            print(f"exact count {exact!s:5}  batch {batch_size:2}  texts {len(subset):2}  extract_many identical {same}"
                  f"  long identical {same_long}")
    gliner2_exact.uninstall()
assert "encode_tokens" not in model.__dict__

# the production detection code: detect_batch with the pipeline must equal detect_batch without it, 3 and 2 passes,
# a queue-sized run and the whole set
sys.path.insert(0, str(ROOT / "artifacts"))
from colab_pipeline import gliner2_ner as ner  # noqa: E402

ner.loaded_checkpoints[("legacy", "cpu", None)] = model
ner.gliner2_pipelined = gliner2_pipelined
gliner2_exact.install(count_step0=True, format_cache=False)
labels = ["Person", "Organization", "Street", "City"]
for city_pass in (True, False):
    ner.CITY_PASS = city_pass
    for subset in (texts[:16], texts):
        ner.PIPELINE = False
        stock = ner.Gliner2NER().detect_batch(subset, labels, device="cpu")
        ner.PIPELINE = True
        same = ner.Gliner2NER().detect_batch(subset, labels, device="cpu") == stock
        ok &= same
        print(f"detect_batch  city pass {city_pass!s:5}  texts {len(subset):2}  pipeline identical {same}")
gliner2_exact.uninstall()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
