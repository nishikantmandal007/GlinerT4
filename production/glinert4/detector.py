import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import pipeline
from .memory import MemoryConfig, refine
from .runtime import load_model

DESCRIPTIONS = {
    "Person": "First and last names of people. NOT company names, street addresses, cities, or identifier codes.",
    "Insurance Number": ("Health, life, or social-security insurance policy identifier. NOT order numbers, invoice "
                         "numbers, reference numbers, SKUs, or part numbers."),
    "Organization": ("Company, corporation, institution, agency, or legal entity name. NOT person names, street "
                     "addresses, cities, or identifier codes."),
    "Street": ("Street, road, avenue, boulevard, or lane name. NOT city names, building names, or postal codes."),
}

CELL_GAP = re.compile(r"\s{2,}|[\t\n\r\f\v]")
WORD_SPAN = re.compile(r"\w(?:.*\w)?", re.S)
HOUSE_NUMBER = re.compile(r"\b\d+\b")


@dataclass(frozen=True)
class DetectorConfig:
    checkpoint: str = "fastino/gliner2-multi-v1"
    labels: Tuple[str, ...] = ("Person", "Organization", "Street", "City")
    descriptions: Dict[str, str] = field(default_factory=lambda: dict(DESCRIPTIONS))
    thresholds: Dict[str, float] = field(default_factory=lambda: {"Organization": 0.85, "Street": 0.5})
    default_threshold: float = 0.65
    window_words: int = 384
    window_overlap: int = 64
    batch_size: int = 16
    length_bucket: int = 32
    memory: Optional[MemoryConfig] = None

    def threshold(self, label):
        return self.thresholds.get(label, self.default_threshold)


def cell_pieces(text, start, end):
    pieces, piece_start = [], start
    for gap in [*CELL_GAP.finditer(text, start, end), None]:
        piece_end = gap.start() if gap else end
        word = WORD_SPAN.search(text, piece_start, piece_end)
        if word:
            pieces.append((word.start(), word.end()))
        piece_start = gap.end() if gap else end
    return pieces


def resolve_spans(text, spans):
    from .overlap import resolve_overlaps

    pieces = [(label, s, e, *rest) for label, start, end, *rest in spans for s, e in cell_pieces(text, start, end)]
    rank = {"score": lambda span: span[3], "start": lambda span: span[1], "end": lambda span: span[2]}
    largest = resolve_overlaps(pieces, "longest", **rank)
    return sorted(resolve_overlaps(largest, "flat", **rank), key=lambda span: span[1])


class Detector:
    def __init__(self, config: Optional[DetectorConfig] = None, device: str = "cuda", model=None, fast: bool = True):
        from . import install

        self.config = config or DetectorConfig()
        self.device = device
        self.fast = fast
        if fast:
            install(kernels=device.startswith("cuda"))
        self.model = model if model is not None else load_model(self.config.checkpoint, device,
                                                                 self.config.length_bucket)

    def _extract(self, texts, entity_types, threshold, candidate_threshold=None):
        config = self.config
        if self.fast:
            return pipeline.extract_entities_long_many(
                self.model, [(texts, entity_types, threshold)], batch_size=config.batch_size, include_confidence=True,
                include_spans=True, chunk_size=config.window_words, chunk_overlap=config.window_overlap,
                candidate_threshold=candidate_threshold)[0]

        def stock(cut):
            return self.model.batch_extract_entities_long(
                texts, entity_types, batch_size=config.batch_size, threshold=cut, include_confidence=True,
                include_spans=True, chunk_size=config.window_words, chunk_overlap=config.window_overlap)

        return stock(threshold) if candidate_threshold is None else list(zip(stock(threshold), stock(candidate_threshold)))

    def detect(self, texts: Sequence[str], labels: Optional[Sequence[str]] = None) -> List[dict]:
        labels = list(labels or self.config.labels)
        memory = self.config.memory
        floor = memory.candidate_floor if memory is not None and "pool" in memory.steps else None
        found = self.spans(texts, labels, floor)
        return self.finish(texts, found, labels, memory)

    def spans(self, texts: Sequence[str], labels: Sequence[str], candidate_floor: Optional[float] = None):
        config = self.config
        entity_types = {label: config.descriptions.get(label, label) for label in labels}
        threshold = min(config.threshold(label) for label in labels)
        floor = None if candidate_floor is None or candidate_floor >= threshold else candidate_floor
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        results = self._extract([texts[index] for index in order], entity_types, threshold, floor)
        spans = [[] for _ in texts]
        below = [[] for _ in texts]
        candidates = []
        for index, result in zip(order, results):
            if floor is not None:
                result, extra = result
                below[index] = [(label, entity["start"], entity["end"], entity.get("confidence", 1.0))
                                for label, entities in extra.get("entities", {}).items() if label in entity_types
                                for entity in entities if entity.get("start") is not None
                                and entity.get("confidence", 1.0) < config.threshold(label)]
            for label, entities in result.get("entities", {}).items():
                if label not in entity_types:
                    continue
                for entity in entities:
                    confidence = entity.get("confidence", 1.0)
                    if confidence < config.threshold(label):
                        continue
                    pieces = cell_pieces(texts[index], entity["start"], entity["end"])
                    if len(pieces) >= 2:
                        candidates.extend((index, label, start, end) for start, end in pieces)
                    else:
                        spans[index].append((label, entity["start"], entity["end"], confidence))
        if candidates:
            confirmations = self.model.batch_extract_entities(
                [texts[index][start:end] for index, _, start, end in candidates], entity_types,
                batch_size=config.batch_size, include_confidence=True, include_spans=True, threshold=threshold)
            for (index, label, piece_start, piece_end), result in zip(candidates, confirmations):
                for entity in result.get("entities", {}).get(label, []):
                    confidence = entity.get("confidence", 1.0)
                    start, end = entity.get("start"), entity.get("end")
                    if (confidence >= config.threshold(label) and start is not None and end is not None
                            and 0 <= start < end <= piece_end - piece_start):
                        spans[index].append((label, piece_start + start, piece_start + end, confidence))
        return list(zip(spans, below))

    def finish(self, texts: Sequence[str], found, labels: Sequence[str], memory: Optional[MemoryConfig] = None):
        if memory is not None:
            found = [(refine(text, accepted, below, self.config.threshold, memory), below)
                     for text, (accepted, below) in zip(texts, found)]
        return [self._found(text, accepted, list(labels)) for text, (accepted, _) in zip(texts, found)]

    @staticmethod
    def _found(text, spans, labels):
        values = {label: [] for label in labels}
        occurrences = []
        for label, start, end, confidence, *source in resolve_spans(text, spans):
            value = HOUSE_NUMBER.sub("", text[start:end]).strip() if label == "Street" else text[start:end]
            if value and value not in values[label]:
                values[label].append(value)
            occurrence = {"label": label, "start": start, "end": end, "score": round(confidence, 4)}
            occurrences.append({**occurrence, "source": source[0]} if source else occurrence)
        return {"values": values, "occurrences": occurrences}
