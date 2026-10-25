import pytest

from glinert4 import Detector, DetectorConfig, runtime
from glinert4.memory import MemoryConfig, name_like, refine
from standin import build, corpus

TEXT = ("Between Kaveri Analytics Private Limited, represented by Lakshmi Varma, and Asterfield Research Ltd. "
        "Notices go to Ms. Varma at Kaveri Analytics. L. Varma signed for Lakshmi Varma. Varma agreed. "
        "Kochi office, Kochi branch, Kochi team. Kochi said yes. Reference 2031-02-0001 and 2031-02-0001. "
        "May the parties agree. May is fine. Reviewer Anand Rao approved. Later Anand Rao left.")


def span(label, form, score, nth=0):
    start = -1
    for _ in range(nth + 1):
        start = TEXT.index(form, start + 1)
    return label, start, start + len(form), score


def threshold(label):
    return {"Organization": 0.85, "Street": 0.5}.get(label, 0.65)


def run(accepted, candidates=(), **options):
    found = refine(TEXT, accepted, list(candidates), threshold, MemoryConfig(**options))
    return {(label, TEXT[start:end], start, source) for label, start, end, _, source in found}


def added(result, source):
    return sorted((label, text) for label, text, _, kind in result if kind == source)


def test_no_steps_returns_the_model_spans():
    accepted = [span("Person", "Lakshmi Varma", 0.95)]
    assert run(accepted, steps=()) == {("Person", "Lakshmi Varma", TEXT.index("Lakshmi Varma"), "model")}


def test_confident_names_spread_to_every_exact_occurrence():
    result = run([span("Person", "Lakshmi Varma", 0.95)], steps=("spread",))
    assert added(result, "memory:spread") == [("Person", "Lakshmi Varma")]


def test_weak_names_do_not_spread():
    assert added(run([span("Person", "Lakshmi Varma", 0.8)], steps=("spread",)), "memory:spread") == []


def test_ids_and_common_words_never_spread():
    accepted = [span("Person", "2031-02-0001", 0.99), span("Person", "May", 0.99)]
    assert added(run(accepted, steps=("spread",)), "memory:spread") == []


def test_short_forms_mark_the_name_without_its_title():
    accepted = [span("Person", "Lakshmi Varma", 0.95), span("Organization", "Kaveri Analytics Private Limited", 0.97)]
    result = run(accepted, steps=("spread", "aliases"))
    assert added(result, "memory:alias") == [("Organization", "Kaveri Analytics"), ("Person", "L. Varma"),
                                             ("Person", "Varma")]
    titled = TEXT.index("Ms. Varma") + len("Ms. ")
    assert ("Person", "Varma", titled, "memory:alias") in result
    with_bare = run(accepted, steps=("spread", "aliases"), bare_surnames=True)
    assert added(with_bare, "memory:alias") == [("Organization", "Kaveri Analytics"), ("Person", "L. Varma"),
                                                ("Person", "Varma"), ("Person", "Varma")]


def test_short_forms_shared_by_two_people_are_left_alone():
    accepted = [span("Person", "Lakshmi Varma", 0.95), ("Person", 0, 0, 0.0)]
    text_with_two = TEXT + " Rahul Varma joined."
    second = text_with_two.index("Rahul Varma")
    found = refine(text_with_two, [accepted[0], ("Person", second, second + len("Rahul Varma"), 0.95)], [], threshold,
                   MemoryConfig(steps=("aliases",), bare_surnames=True))
    forms = {text_with_two[start:end] for _, start, end, _, source in found if source == "memory:alias"}
    assert forms == {"L. Varma"}


def test_one_label_per_text_when_the_margin_is_clear():
    accepted = [span("City", "Kochi", 0.9, n) for n in range(3)] + [span("Person", "Kochi", 0.7, 3)]
    result = run(accepted, steps=("labels",))
    assert {label for label, text, _, _ in result} == {"City"}
    assert added(result, "memory:label") == [("City", "Kochi")]


def test_close_label_votes_are_left_alone():
    accepted = [span("City", "Kochi", 0.9, 0), span("Person", "Kochi", 0.8, 1)]
    assert added(run(accepted, steps=("labels",)), "memory:label") == []


def test_pooled_evidence_from_different_contexts_is_accepted():
    candidates = [span("Person", "Anand Rao", 0.55, 0), span("Person", "Anand Rao", 0.55, 1)]
    assert added(run([], candidates, steps=("pool",)), "memory:pooled") == [("Person", "Anand Rao")] * 2


def test_pooled_evidence_needs_two_contexts_and_a_minimum_score():
    one = [span("Person", "Anand Rao", 0.6, 0)]
    weak = [span("Person", "Anand Rao", 0.35, 0), span("Person", "Anand Rao", 0.35, 1)]
    assert added(run([], one, steps=("pool",)), "memory:pooled") == []
    assert added(run([], weak, steps=("pool",)), "memory:pooled") == []


@pytest.mark.parametrize("form,label,expected", [
    ("Lakshmi Varma", "Person", True), ("lakshmi varma", "Person", False), ("Ann", "Person", False),
    ("Anand", "Person", True), ("May", "Person", False), ("114 Malabar Lane", "Street", True),
    ("114", "Street", False), ("Agent 007", "Person", False), ("a@b.example", "Person", False),
    ("प्रकाश जावड़ेकर", "Person", True), ("प्रकाश", "Person", False),
])
def test_name_like(form, label, expected):
    assert name_like(form, label) is expected


def test_detector_without_memory_steps_matches_the_plain_detector():
    model = runtime.freeze_eval(build())
    texts = sorted(corpus(12, seed=7), key=len)
    plain = Detector(DetectorConfig(window_words=96, window_overlap=16), device="cpu", model=model).detect(texts)
    config = DetectorConfig(window_words=96, window_overlap=16, memory=MemoryConfig(steps=()))
    with_memory = Detector(config, device="cpu", model=model).detect(texts)
    strip = [{"values": found["values"], "occurrences": [{k: v for k, v in o.items() if k != "source"}
                                                         for o in found["occurrences"]]} for found in with_memory]
    assert strip == plain
    assert all(o["source"] == "model" for found in with_memory for o in found["occurrences"])


def test_matches_stay_inside_word_boundaries_in_indic_scripts():
    from glinert4.memory import occurrences

    text = "राम ने रामायण पढ़ी, राम खुश था"
    assert [text[s:e] for s, e in occurrences(text, "राम")] == ["राम", "राम"]
    assert occurrences("Varmas and Varma", "Varma") == [(11, 16)]


def test_candidates_come_from_a_second_decode_and_leave_the_output_unchanged():
    model = runtime.freeze_eval(build())
    texts = sorted(corpus(12, seed=7), key=len)
    config = DetectorConfig(window_words=96, window_overlap=16)
    for fast in (True, False):
        detector = Detector(config, device="cpu", model=model, fast=fast)
        labels = list(config.labels)
        plain = detector.spans(texts, labels)
        with_candidates = detector.spans(texts, labels, candidate_floor=0.3)
        assert [accepted for accepted, _ in with_candidates] == [accepted for accepted, _ in plain]
        assert any(below for _, below in with_candidates)
        assert all(score < config.threshold(label) for _, below in with_candidates for label, _, _, score in below)
