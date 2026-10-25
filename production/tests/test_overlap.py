import random

import pytest
from gliner2.inference.overlap import resolve_overlaps as stock

from glinert4.overlap import resolve_overlaps


def spans(seed, count, as_dicts):
    rng = random.Random(seed)
    items, position = [], 0
    for _ in range(count):
        position += rng.randint(0, 40)
        start, end = position, position + rng.randint(1, 30)
        score = rng.choice([0.5, 0.7, 0.9, rng.random()])
        items.append({"start": start, "end": end, "confidence": score} if as_dicts else ("Person", start, end, score))
        if rng.random() < 0.3:
            items.append(dict(items[-1]) if as_dicts else items[-1])
        if rng.random() < 0.3:
            other_start = max(0, start + rng.randint(-5, 5))
            other_end = max(other_start + 1, end + rng.randint(-3, 8))
            items.append({"start": other_start, "end": other_end, "confidence": score}
                         if as_dicts else ("City", other_start, other_end, score))
    return items


@pytest.mark.parametrize("policy", ["flat", "disallow", "longest", "nested", "allow", None])
@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("as_dicts", [False, True])
def test_matches_gliner2_for_every_policy(policy, seed, as_dicts):
    items = spans(seed, 300, as_dicts)
    if as_dicts:
        keys = dict(score=lambda i: i["confidence"], start=lambda i: i["start"], end=lambda i: i["end"])
    else:
        keys = dict(score=lambda i: i[3], start=lambda i: i[1], end=lambda i: i[2])
    assert resolve_overlaps(items, policy, default="disallow", **keys) == stock(items, policy, default="disallow", **keys)


def test_invalid_spans_fall_back_to_gliner2():
    items = [("Person", 10 * i, 10 * i + 5, 0.5) for i in range(80)] + [("Person", 50, 50, 0.9), ("City", 60, 55, 0.4)]
    keys = dict(score=lambda i: i[3], start=lambda i: i[1], end=lambda i: i[2])
    for policy in ("flat", "longest", "nested"):
        assert resolve_overlaps(items, policy, **keys) == stock(items, policy, **keys)


def test_scales_linearly_with_far_apart_groups():
    import time

    items = spans(1, 6000, False)
    keys = dict(score=lambda i: i[3], start=lambda i: i[1], end=lambda i: i[2])
    start = time.perf_counter()
    resolve_overlaps(items, "longest", **keys)
    assert time.perf_counter() - start < 2.0
