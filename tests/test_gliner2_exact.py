# serving/gliner2_exact.py must return exactly what stock gliner2 2.1.0 returns (==, every confidence float, span and
# order), each change alone and together, for single and long-document extraction; count_instance0 must equal instance
# 0 of the stock CountLSTM bit for bit for every count, and the multi-step path must actually be exercised.
# CPU, random-weight stand-in model, no Hub access:
#   python tests/test_gliner2_exact.py
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serving"))
sys.path.insert(0, str(ROOT / "tests"))
import gliner2_exact  # noqa: E402
import gliner2_standin  # noqa: E402

LABELS = [{"Person": "people", "Organization": "company"}, {"Street": "street", "City": "town"}, {"City": "town"}]


def run(model, texts):
    out = []
    for labels in LABELS:
        out.append(model.batch_extract_entities(texts, labels, batch_size=16, threshold=0.3,
                                                include_confidence=True, include_spans=True))
        out.append(model.batch_extract_entities_long(texts, labels, batch_size=16, threshold=0.3,
                                                     include_confidence=True, include_spans=True,
                                                     chunk_size=96, chunk_overlap=16))
    return out


ok = True

model = gliner2_standin.build()

# count_instance0 against the stock module, every count, fp32 and fp16 (CPU)
for dtype in (torch.float32, torch.float16):
    embed = model.count_embed.to(dtype)
    worst = 0
    for count in range(1, 22):
        for fields in (1, 3, 7):
            pc = torch.randn(fields, embed.hidden_size, dtype=dtype)
            with torch.inference_mode():
                same = torch.equal(embed(pc, count)[:1], gliner2_exact.count_instance0(embed, pc, count))
            worst += not same
    ok &= worst == 0
    print(f"count_instance0 {str(dtype):14} counts 1..21 x fields 1/3/7  mismatches {worst}")
model.count_embed.float()

# the multi-step branch must run on this corpus
steps = []
_instance0 = gliner2_exact.count_instance0
gliner2_exact.count_instance0 = lambda e, p, c: (steps.append(c), _instance0(e, p, c))[1]

texts = sorted(gliner2_standin.corpus(60, seed=2), key=len) + ["Contains [SEP_TEXT] marker John Smith."]
gliner2_exact.uninstall()
reference = run(model, texts)
for flags in ((True, False), (False, True), (True, True)):
    gliner2_exact.install(*flags)
    same = run(model, texts) == reference
    ok &= same
    print(f"count_step0 {flags[0]!s:5}  format_cache {flags[1]!s:5}  identical {same}")
gliner2_exact.uninstall()
gliner2_exact.count_instance0 = _instance0
multi = sum(c > 1 for c in steps)
ok &= multi > 0
print(f"entity count calls {len(steps)}, of which pred_count > 1: {multi}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
