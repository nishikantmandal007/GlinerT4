import functools
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Sequence, Tuple

import ahocorasick

Span = Tuple[str, int, int, float, str]

TITLES = ("Mr", "Mrs", "Ms", "Miss", "Mx", "Dr", "Prof", "Sir", "Dame", "Shri", "Smt", "Sri", "Kumari", "Herr", "Frau",
          "Mme", "Mlle", "Sr", "Sra", "Srta", "Sig", "Sig.ra")
LEGAL_SUFFIXES = ("Private Limited", "Pvt. Ltd.", "Pvt Ltd", "Pty Ltd", "Limited", "Ltd.", "Ltd", "GmbH & Co. KG", "GmbH",
                  "AG", "Inc.", "Inc", "LLC", "LLP", "PLC", "plc", "Corp.", "Corp", "Corporation", "Co.", "S.A.", "SA",
                  "SAS", "S.p.A.", "SpA", "S.r.l.", "B.V.", "BV", "N.V.", "NV", "K.K.", "KK", "Oy", "AB", "ASA")
STOPWORDS = frozenset(word.casefold() for word in (
    "The", "This", "That", "These", "Those", "A", "An", "And", "Or", "Of", "In", "On", "At", "To", "For", "By", "With",
    "From", "As", "Is", "Are", "Was", "Be", "It", "We", "You", "They", "He", "She", "His", "Her", "Our", "Your", "Their",
    "May", "Will", "Can", "Shall", "Must", "Bill", "Mark", "Art", "Rose", "Grace", "Hope", "Faith", "June", "April",
    "August", "Page", "Section", "Clause", "Schedule", "Annex", "Appendix", "Agreement", "Contract", "Customer",
    "Supplier", "Client", "Party", "Parties", "Company", "Notice", "Notices", "Reviewer", "Manager", "Director",
    "Team", "Office", "Department", "Services", "Service", "Group", "Board", "Committee", "Account", "Name", "Date",
    "Address", "Street", "Road", "City", "Country", "Phone", "Email", "Total", "Note", "Notes", "Re", "Subject",
    "Dear", "Regards", "Sincerely", "Yes", "No", "None", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "January", "February", "March", "July", "September", "October", "November", "December"))


@dataclass(frozen=True)
class MemoryConfig:
    steps: Tuple[str, ...] = ("labels", "spread", "aliases")
    candidate_floor: float = 0.3
    seed_confidence: float = 0.9
    label_margin: float = 2.0
    pool_min_score: float = 0.4
    pool_min_contexts: int = 2
    pool_top: int = 5
    context_chars: int = 30
    spread_labels: Tuple[str, ...] = ("Person", "Organization", "City", "Street")
    alias_labels: Tuple[str, ...] = ("Person", "Organization")
    bare_surnames: bool = False
    stopwords: FrozenSet[str] = STOPWORDS


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def cased(text):
    return any(ch.isupper() or ch.islower() for ch in text)


def word_char(ch):
    return ch.isalnum() or ch == "_" or unicodedata.category(ch)[0] == "M"


@functools.lru_cache(maxsize=65536)
def name_like(form, label, stopwords=STOPWORDS):
    form = form.strip()
    words = form.split()
    if len(form) < 3 or "@" in form or not words or norm(form) in stopwords:
        return False
    if not any(ch.isalpha() for ch in form):
        return False
    if label == "Street":
        return len(words) >= 2
    if any(ch.isdigit() for ch in form):
        return False
    if not cased(form):
        return len(words) >= 2
    if not form[0].isupper():
        return False
    if len(words) == 1:
        return len(form) >= 4
    return True


def occurrences(text, form):
    found = []
    for match in re.finditer(re.escape(form), text):
        start, end = match.span()
        if not (start and word_char(text[start - 1])) and not (end < len(text) and word_char(text[end])):
            found.append((start, end))
    return found


class Coverage:
    def __init__(self, length, spans=()):
        self.marks = bytearray(length)
        for start, end in spans:
            self.add(start, end)

    def add(self, start, end):
        self.marks[start:end] = b"\x01" * (end - start)

    def hits(self, start, end):
        return self.marks.find(1, start, end) != -1


def bounded(text, start, end):
    return not (start and word_char(text[start - 1])) and not (end < len(text) and word_char(text[end]))


def vote_labels(text, spans, config):
    totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for label, start, end, score, _ in spans:
        totals[norm(text[start:end])][label] += score
    winners = {}
    for form, by_label in totals.items():
        if len(by_label) > 1:
            ranked = sorted(by_label.items(), key=lambda item: -item[1])
            if ranked[0][1] >= config.label_margin * ranked[1][1]:
                winners[form] = ranked[0][0]
    voted = []
    for label, start, end, score, source in spans:
        winner = winners.get(norm(text[start:end]), label)
        voted.append((winner, start, end, score, "memory:label") if winner != label else (label, start, end, score,
                                                                                            source))
    return voted


def pool(text, spans, candidates, threshold, config):
    taken = Coverage(len(text), ((start, end) for _, start, end, _, _ in spans))
    groups: Dict[Tuple[str, str], List[Tuple[int, int, float]]] = defaultdict(list)
    for label, start, end, score in candidates:
        if score >= config.pool_min_score and not taken.hits(start, end):
            groups[label, norm(text[start:end])].append((start, end, score))
    added = []
    width = config.context_chars
    for (label, _), members in groups.items():
        members = sorted(set(members))
        if not name_like(text[members[0][0]:members[0][1]], label, config.stopwords):
            continue
        contexts = {(norm(text[max(0, s - width):s]), norm(text[e:e + width])) for s, e, _ in members}
        if len(contexts) < config.pool_min_contexts:
            continue
        best = sorted((score for _, _, score in members), reverse=True)[:config.pool_top]
        pooled = 1 - math.prod(1 - score for score in best)
        if pooled >= threshold(label):
            added += [(label, s, e, pooled, "memory:pooled") for s, e, _ in members]
    return added


def seeds(text, spans, labels, config):
    found: Dict[str, Dict[str, float]] = defaultdict(dict)
    for label, start, end, score, _ in spans:
        form = text[start:end].strip()
        if label in labels and score >= config.seed_confidence and name_like(form, label, config.stopwords):
            found[form][label] = max(score, found[form].get(label, 0.0))
    return {form: next(iter(by_label.items())) for form, by_label in found.items() if len(by_label) == 1}


def strip_suffix(name):
    for suffix in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        if name.endswith(" " + suffix):
            return name[:-len(suffix) - 1].rstrip(" ,")
    return name


TITLE_BEFORE = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(title) for title in TITLES) + r")\.?\s+$")
INITIAL_BEFORE = re.compile(r"(?<!\w)(\w)\.\s?$")


def scan(text, spans, config, spread_names, short_forms):
    labels = set(config.spread_labels if spread_names else ()) | set(config.alias_labels if short_forms else ())
    every = seeds(text, spans, labels, config)
    patterns: Dict[str, list] = defaultdict(list)
    if spread_names:
        for form, (label, score) in every.items():
            if label in config.spread_labels:
                patterns[form].append(("spread", label, score, None))
    known = {form: seed for form, seed in every.items() if short_forms and seed[0] in config.alias_labels}
    owners: Dict[Tuple[str, str], set] = defaultdict(set)
    initials: Dict[Tuple[str, str], set] = defaultdict(set)
    for form, (label, _) in known.items():
        words = form.split()
        if label == "Person" and len(words) >= 2 and cased(form) and name_like(words[-1], label, config.stopwords):
            owners[label, words[-1]].add(form)
            initials[words[0][0], words[-1]].add(form)
        elif label == "Organization":
            base = strip_suffix(form)
            if base != form and name_like(base, label, config.stopwords):
                owners[label, base].add(form)
    for (label, core), names in owners.items():
        if core not in known:
            patterns[core].append(("alias", label, None, names))
    if not patterns:
        return []
    automaton = ahocorasick.Automaton()
    for pattern, entries in patterns.items():
        automaton.add_word(pattern, (len(pattern), entries))
    automaton.make_automaton()
    matches = sorted(((last + 1 - length, last + 1, entries) for last, (length, entries) in automaton.iter(text)),
                     key=lambda match: (match[0], match[0] - match[1]))
    taken = Coverage(len(text), ((start, end) for _, start, end, _, _ in spans))
    added = []
    for start, end, entries in matches:
        if not bounded(text, start, end) or taken.hits(start, end):
            continue
        for kind, label, score, names in entries:
            if kind == "spread":
                added.append((label, start, end, score, "memory:spread"))
                taken.add(start, end)
                break
            core = text[start:end]
            owner = next(iter(names)) if len(names) == 1 else None
            begin = start
            if label == "Person":
                before = text[max(0, start - 12):start]
                initial = INITIAL_BEFORE.search(before)
                if initial and len(initials.get((initial.group(1), core), ())) == 1:
                    owner = next(iter(initials[initial.group(1), core]))
                    begin = start - len(initial.group(0))
                elif not (TITLE_BEFORE.search(before) or config.bare_surnames):
                    owner = None
            if owner is not None and not taken.hits(begin, end):
                added.append((label, begin, end, known[owner][1], "memory:alias"))
                taken.add(begin, end)
                break
    return added


def refine(text: str, accepted: Sequence[Tuple[str, int, int, float]],
           candidates: Sequence[Tuple[str, int, int, float]], threshold: Callable[[str], float],
           config: MemoryConfig) -> List[Span]:
    spans: List[Span] = [(label, start, end, score, "model") for label, start, end, score in accepted]
    steps = set(config.steps)
    if "labels" in steps:
        spans = vote_labels(text, spans, config)
    if "pool" in steps:
        spans += pool(text, spans, candidates, threshold, config)
    if "spread" in steps or "aliases" in steps:
        spans += scan(text, spans, config, "spread" in steps, "aliases" in steps)
    return spans
