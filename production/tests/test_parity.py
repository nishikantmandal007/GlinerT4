import copy

import pytest
import torch

from glinert4 import Detector, DetectorConfig, count, pipeline, runtime
from standin import build, corpus

ENTITY_SETS = [{"Organization": "company"}, {"Person": "people", "Street": "street", "City": "town"}]


@pytest.fixture(scope="module")
def model():
    return runtime.freeze_eval(build())


@pytest.fixture(scope="module")
def texts():
    return sorted(corpus(40, seed=4), key=len)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_count_instance0_is_bit_identical(model, dtype):
    embed = copy.deepcopy(model.count_embed).to(dtype)
    torch.manual_seed(0)
    with torch.inference_mode():
        for steps in range(1, 22):
            for fields in (1, 3, 7):
                pc = torch.randn(fields, embed.hidden_size, dtype=dtype)
                assert torch.equal(embed(pc, steps)[:1], count.count_instance0(embed, pc, steps))


@pytest.mark.parametrize("batch_size", [7, 16, 64])
def test_extract_many_matches_batch_extract(model, texts, batch_size):
    jobs = [{"texts": texts, "schemas": model.create_schema().entities(entities), "batch_size": batch_size,
             "threshold": 0.3, "include_confidence": True, "include_spans": True} for entities in ENTITY_SETS]
    stock = [model.batch_extract(texts, job["schemas"], batch_size, 0.3, 0, True, True, True) for job in jobs]
    assert pipeline.extract_many(model, jobs) == stock


def test_long_extraction_matches_stock(model, texts):
    stock = [model.batch_extract_entities_long(texts, entities, batch_size=16, threshold=0.3, include_confidence=True,
                                               include_spans=True, chunk_size=96, chunk_overlap=16)
             for entities in ENTITY_SETS]
    fast = pipeline.extract_entities_long_many(model, [(texts, entities, 0.3) for entities in ENTITY_SETS],
                                               batch_size=16, include_confidence=True, include_spans=True,
                                               chunk_size=96, chunk_overlap=16)
    assert fast == stock


def test_detector_fast_path_matches_stock_gliner2(model, texts):
    config = DetectorConfig(window_words=96, window_overlap=16)
    stock = Detector(config, device="cpu", model=model, fast=False).detect(texts)
    fast = Detector(config, device="cpu", model=model, fast=True).detect(texts)
    assert fast == stock
    assert any(found["occurrences"] for found in fast)
