from gliner2.inference import chunking
from gliner2.inference.overlap import normalize_overlap_policy
from gliner2.inference.overlap import resolve_overlaps as stock_resolve_overlaps

_installed = {}


def groups(items, start, end):
    order = sorted(range(len(items)), key=lambda index: (int(start(items[index])), int(end(items[index]))))
    found, current, reach = [], [], None
    for index in order:
        item_start, item_end = int(start(items[index])), int(end(items[index]))
        if current and item_start < reach:
            current.append(index)
            reach = max(reach, item_end)
        else:
            if current:
                found.append(current)
            current, reach = [index], item_end
    if current:
        found.append(current)
    return found


def resolve_overlaps(items, policy, *, score, start, end, default=None):
    if (len(items) < 64 or normalize_overlap_policy(policy, default=default) == "allow"
            or any(int(end(item)) <= int(start(item)) for item in items)):
        return stock_resolve_overlaps(items, policy, score=score, start=start, end=end, default=default)
    kept = []
    for group in groups(items, start, end):
        members = sorted(group)
        position = {}
        for index in members:
            position.setdefault(id(items[index]), []).append(index)
        for item in stock_resolve_overlaps([items[index] for index in members], policy, score=score, start=start,
                                           end=end, default=default):
            kept.append((position[id(item)].pop(0), item))
    kept.sort(key=lambda row: (-float(score(row[1])), int(start(row[1])), int(end(row[1])), row[0]))
    return [item for _, item in kept]


def install():
    if chunking not in _installed:
        _installed[chunking] = chunking.resolve_overlaps
        chunking.resolve_overlaps = resolve_overlaps


def uninstall():
    for module, stock in _installed.items():
        module.resolve_overlaps = stock
    _installed.clear()
