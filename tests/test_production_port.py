# production/glinert4's Detector must return exactly what the validated research path returns for the shipping
# setting (single pass, Street 0.5, whole documents in 384/64 windows, exact count head, one-thread pipeline):
# artifacts/colab_pipeline/gliner2_ner.py detect_batch with serving/gliner2_exact.py and serving/gliner2_pipelined.py.
# CPU, random-weight stand-in model:
#   python tests/test_production_port.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "production", ROOT / "production" / "tests", ROOT / "artifacts", ROOT / "serving"):
    sys.path.insert(0, str(path))

import glinert4  # noqa: E402
import gliner2_exact  # noqa: E402
import gliner2_pipelined  # noqa: E402
import standin  # noqa: E402
from colab_pipeline import gliner2_ner as ner  # noqa: E402

model = glinert4.runtime.freeze_eval(standin.build())
texts = sorted(standin.corpus(30, seed=7), key=len) + ["Anna  Mueller\tAcme Corp\nBerlin   Main Street 12."]
labels = ["Person", "Organization", "Street", "City"]
ok = True

assert {k: v for k, v in ner.LABEL_DESCRIPTIONS.items()} == glinert4.detector.DESCRIPTIONS
for window, overlap in ((384, 64), (96, 16)):
    ner.loaded_checkpoints[("legacy", "cpu", None)] = model
    ner.strategy_passes = lambda _: [("legacy", labels, {})]
    ner.LABEL_THRESHOLDS, ner.DEFAULT_THRESHOLD = {"Organization": 0.85, "Street": 0.5}, 0.65
    ner.WINDOW_WORDS, ner.WINDOW_OVERLAP_WORDS = window, overlap
    ner.PIPELINE, ner.gliner2_pipelined = True, gliner2_pipelined
    glinert4.uninstall()
    gliner2_exact.install(count_step0=True, format_cache=False)
    research = ner.Gliner2NER().detect_batch(texts, labels, device="cpu", batch_size=16)
    gliner2_exact.uninstall()
    config = glinert4.DetectorConfig(window_words=window, window_overlap=overlap)
    production = glinert4.Detector(config, device="cpu", model=model).detect(texts)
    glinert4.uninstall()
    same = production == research
    ok &= same
    print(f"window {window}/{overlap}: production Detector identical to the research path: {same}, "
          f"{sum(len(f['occurrences']) for f in research)} occurrences")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
