import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Sequence, Tuple

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
    steps: Tuple[str, ...] = ("labels", "pool", "spread", "aliases")
    candidate_floor: float = 0.3
    seed_confidence: float = 0.9
    label_margin: float = 2.0
    pool_min_score: float = 0.4
    pool_min_contexts: int = 2
    pool_top: int = 5
    context_chars: int = 30
    spread_labels: Tuple[str, ...] = ("Person", "Organization", "City", "Street")
    alias_labels: Tuple[str, ...] = ("Person", "Organization")
    bare_surnames: bool = True
    stopwords: FrozenSet[str] = STOPWORDS


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def cased(text):
    return any(ch.isupper() or ch.islower() for ch in text)


def word_char(ch):
    return ch.isalnum() or ch == "_" or unicodedata.category(ch)[0] == "M"


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


class Taken:
    def __init__(self, spans=(), width=64):
        self.width, self.cells = width, defaultdict(list)
        for start, end in spans:
            self.add(start, end)

    def _cells(self, start, end):
        return range(start // self.width, max(start, end - 1) // self.width + 1)

    def add(self, start, end):
        for cell in self._cells(start, end):
            self.cells[cell].append((start, end))

    def hits(self, start, end):
        return any(s < end and start < e for cell in self._cells(start, end) for s, e in self.cells.get(cell, ()))


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
    taken = Taken((start, end) for _, start, end, _, _ in spans)
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


def spread(text, spans, config):
    taken = Taken((start, end) for _, start, end, _, _ in spans)
    added = []
    for form, (label, score) in sorted(seeds(text, spans, config.spread_labels, config).items(),
                                       key=lambda item: -len(item[0])):
        for start, end in occurrences(text, form):
            if not taken.hits(start, end):
                added.append((label, start, end, score, "memory:spread"))
                taken.add(start, end)
    return added


def strip_suffix(name):
    for suffix in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        if name.endswith(" " + suffix):
            return name[:-len(suffix) - 1].rstrip(" ,")
    return name


def alias_forms(form, label, config):
    words = form.split()
    if label == "Person" and len(words) >= 2 and cased(form):
        surname, initial = words[-1], words[0][0]
        forms = {f"{initial}. {surname}": f"{initial}. {surname}", f"{initial}.{surname}": f"{initial}.{surname}"}
        forms.update({f"{title}{dot} {surname}": surname for title in TITLES for dot in (".", "")})
        if config.bare_surnames:
            forms[surname] = surname
        return {alias: core for alias, core in forms.items() if name_like(surname, label, config.stopwords)}
    if label == "Organization":
        base = strip_suffix(form)
        return {base: base} if base != form and name_like(base, label, config.stopwords) else {}
    return {}


def aliases(text, spans, config):
    known = seeds(text, spans, config.alias_labels, config)
    owners: Dict[str, set] = defaultdict(set)
    cores: Dict[str, str] = {}
    for form, (label, _) in known.items():
        for alias, core in alias_forms(form, label, config).items():
            if alias != form:
                owners[alias].add(form)
                cores[alias] = core
    taken = Taken((start, end) for _, start, end, _, _ in spans)
    added = []
    for alias in sorted(owners, key=len, reverse=True):
        if len(owners[alias]) != 1 or alias in known:
            continue
        label, score = known[next(iter(owners[alias]))]
        for start, end in occurrences(text, alias):
            start = end - len(cores[alias])
            if not taken.hits(start, end):
                added.append((label, start, end, score, "memory:alias"))
                taken.add(start, end)
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
    if "spread" in steps:
        spans += spread(text, spans, config)
    if "aliases" in steps:
        spans += aliases(text, spans, config)
    return spans
