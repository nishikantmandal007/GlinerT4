import importlib.util
import os
import functools
import threading
import types
import torch
from gliner2 import AutoExtractor, GLiNER2
from gliner2.inference.overlap import resolve_overlaps
import re

from .detection_queue import STREAMS, DetectionQueue, detection_devices

model = None

# Per-label descriptions improve precision — a GLiNER2-only capability.
# Labels without a description fall back to the bare label name.
LABEL_DESCRIPTIONS = {
    "Person": (
        "First and last names of people. "
        "NOT company names, street addresses, cities, or identifier codes."
    ),
    "Insurance Number": (
        "Health, life, or social-security insurance policy identifier. "
        "NOT order numbers, invoice numbers, reference numbers, SKUs, or part numbers."
    ),
    "Organization": (
        "Company, corporation, institution, agency, or legal entity name. "
        "NOT person names, street addresses, cities, or identifier codes."
    ),
    "Street": (
        "Street, road, avenue, boulevard, or lane name. "
        "NOT city names, building names, or postal codes."
    ),
}


# the threshold each label is kept at; the same values the one-label detectors below use
LABEL_THRESHOLDS = {"Organization": 0.85}
DEFAULT_THRESHOLD = 0.65


# GLiNER2 encodes all labels of a pass together with the text, so labels shift each other's confidence. Measured on
# a T4: Organization recall is 0.93 alone, 0.98 next to Insurance Number, and falls to 0.62 when Person, City and
# Street share its pass (each name-like label takes some of its confidence); Street gains next to Person (0.66 -> 0.95).
# So Organization shares a pass only with labels that sharpen it, and the name-like labels run together.
PASS_GROUPS = [{"Organization", "Insurance Number"}]


def label_passes(labels):
    passes = [[label for label in labels if label in group] for group in PASS_GROUPS]
    grouped = set().union(*PASS_GROUPS)
    passes.append([label for label in labels if label not in grouped])
    return [labels_in_pass for labels_in_pass in passes if labels_in_pass]


# which checkpoints answer which labels; chosen per deployment with DETECTION_STRATEGY
DETECTION_STRATEGY = os.getenv("DETECTION_STRATEGY", "per_label")
CHECKPOINTS = {
    "legacy": "fastino/gliner2-multi-v1",
    "v25": "fastino/gliner2.5-multi-v1",
    "pii": "fastino/gliner2-privacy-filter-PII-multi",
}
# the PII checkpoint's own names for the labels it covers; it has no organization or insurance label
PII_LABELS = {"Person": "person", "Street": "street_address", "City": "city"}
loaded_checkpoints = {}


# a city scored alone is caught most often (0.96), but Street needs City beside it (0.95 vs 0.85), so City also gets
# a pass of its own and the cities of both passes are kept: City 0.86 -> 0.97 with Street unchanged
CITY_ALONE_DESCRIPTION = "Name of a city, town or village."
# measured again on the detbench corpus with today's descriptions, thresholds and cell confirmation (GlinerT4
# pass_bench, 2026-10-25): without the City-alone pass City is 1.000/.947 (with it 1.000/.930), every other label is
# unchanged, and the run is 1.46x faster (two passes instead of three); off until a held-out corpus confirms it
CITY_PASS = os.getenv("DETECTION_CITY_PASS", "1") == "1"


# fp16 with DisentangledFlash on the GPU: DeBERTa's attention fused into one kernel, about 1.6-2x faster on a T4 with
# results equal to fp32. On a T4 it needs Triton 3.2 (torch 2.6): Triton 3.3 fails to compile it there and later
# versions run it without tensor cores. Without the package, compiled kernels (1.4x) are the fallback; the compile
# cache is raised because the count head sees a new shape for every entity count and stops compiling after 8 by default.
FUSED_ATTENTION = importlib.util.find_spec("disentangled_flash") is not None
# the sm_75 CUDA attention kernel (gliner-t4-kernel checkout) in place of disentangled_flash's Triton one, opt-in;
# gliner2 still sends heavily padded batches to its packed path; every other batch runs on this kernel
ATTENTION_KERNEL = os.getenv("DETECTION_ATTENTION_KERNEL")
if ATTENTION_KERNEL and FUSED_ATTENTION:
    import sys

    sys.path.insert(0, os.path.join(ATTENTION_KERNEL, "torch-ext"))
    import gliner_t4

    gliner_t4.install()  # GLINER_T4_FUSE_LAYERNORM=1 also fuses residual add + LayerNorm (bit-identical)

# exact count head (serving/gliner2_exact.py of the same checkout): entity count queries run only the GRU step they
# keep, with the stock kernel shapes, so results are bit-identical (unlike DETECTION_FAST_COUNT); opt-in
if ATTENTION_KERNEL and os.getenv("DETECTION_EXACT_COUNT") == "1":
    import sys

    sys.path.insert(0, os.path.join(ATTENTION_KERNEL, "serving"))
    import gliner2_exact

    gliner2_exact.install(count_step0=True, format_cache=False)


# with more than one run in flight per GPU every serving thread has its own copy: the fused encoder keeps per-call
# state on the module (the active sequence-length plan) that two threads would overwrite
def checkpoint(key, device="cuda"):
    slot = (key, device, threading.current_thread().name if STREAMS > 1 else None)
    if slot not in loaded_checkpoints:
        on_gpu = device != "cpu"
        options = {"attention_backend": "disentangled_flash"} if on_gpu and FUSED_ATTENTION and not FP32_REFERENCE else {}
        loaded = AutoExtractor.from_pretrained(CHECKPOINTS[key], map_location=device,
                                               quantize=on_gpu and not FP32_REFERENCE, **options)
        if options and LENGTH_BUCKET > 1:
            loaded.encoder.register_forward_pre_hook(pad_to_bucket, with_kwargs=True, prepend=True)
            loaded.encoder.register_forward_hook(cut_to_length, with_kwargs=True)
        if on_gpu and not FUSED_ATTENTION:
            torch._dynamo.config.cache_size_limit = 64
            loaded.compile()
        if FAST_COUNT and type(loaded.count_embed).__name__ == "CountLSTM":
            loaded._compute_direct_span_logits_batched = types.MethodType(direct_span_logits, loaded)
        if SKIP_EVAL:
            loaded.eval()
            loaded.eval = types.MethodType(lambda model: model, loaded)
        # a GPU's serving threads each need their own model (the fused encoder keeps per-call state), but the fp16
        # weights are the same read-only tensors: later copies point at the first copy's, about 600 MB less per copy
        first = next((model for (k, d, _), model in loaded_checkpoints.items() if (k, d) == (key, device)), None)
        pairs = list(zip(loaded.named_parameters(), first.named_parameters())) if first is not None else []
        if SHARE_WEIGHTS and pairs and all(a[0] == b[0] and a[1].shape == b[1].shape for a, b in pairs):
            for (_, mine), (_, theirs) in pairs:
                mine.data = theirs.data
        loaded_checkpoints[slot] = loaded
        # loading converts fp32 weights to fp16; return the freed fp32 blocks the allocator would otherwise keep
        # reserved (~1 GB per copy)
        if on_gpu:
            torch.cuda.empty_cache()
    return loaded_checkpoints[slot]


# DisentangledFlash prepares position tables per token length, and on every unseen length gliner2 rebuilds the
# tables of all lengths seen so far; chunks bring hundreds of lengths, so the rebuild never stops. Padding each run
# to a multiple of LENGTH_BUCKET (masked tokens, sliced off after the encoder) leaves a few lengths, each built once.
LENGTH_BUCKET = int(os.getenv("DETECTION_LENGTH_BUCKET", "32"))


def pad_to_bucket(encoder, args, kwargs):
    input_ids, attention_mask = kwargs["input_ids"], kwargs["attention_mask"]
    encoder.unpadded_length = length = input_ids.shape[1]
    extra = -length % LENGTH_BUCKET
    if extra:
        kwargs["input_ids"] = torch.nn.functional.pad(input_ids, (0, extra), value=encoder.config.pad_token_id or 0)
        kwargs["attention_mask"] = torch.nn.functional.pad(attention_mask, (0, extra), value=0)
    return args, kwargs


def cut_to_length(encoder, args, kwargs, output):
    output.last_hidden_state = output.last_hidden_state[:, :encoder.unpadded_length]
    return output


# GLiNER2 builds each (sample, schema) count-aware query with a GRU run for pred_count steps, one small-kernel Python
# loop per pair, yet entity decoding keeps only instance 0. The GRU and projector are row-wise and step 0 depends on
# step 0 alone, so one step over every entity pair's rows in a single call yields the instances it keeps: thousands
# of kernel launches per run become a handful. A copy of gliner2 2.1.0's _compute_direct_span_logits_batched with
# only that change; other schema types keep the per-pair GRU.
FAST_COUNT = os.getenv("DETECTION_FAST_COUNT") == "1"


def direct_span_logits(self, batch, all_token_embs, all_schema_embs):
    batch_size = len(batch)
    raw_logits = [[None] * len(batch.task_types[i]) for i in range(batch_size)]
    pred_counts = [[0] * len(batch.task_types[i]) for i in range(batch_size)]
    count_records, count_inputs = [], []
    for sample_idx in range(batch_size):
        if all_token_embs[sample_idx].numel() == 0:
            continue
        for task_idx, task_type in enumerate(batch.task_types[sample_idx]):
            if task_type == "classifications" or task_idx >= len(all_schema_embs[sample_idx]):
                continue
            schema_emb_list = all_schema_embs[sample_idx][task_idx]
            schema_tokens = batch.schema_tokens_list[sample_idx][task_idx]
            if not schema_emb_list or len(schema_tokens) < 4:
                continue
            if not any(schema_tokens[j] in ("[E]", "[C]", "[R]") for j in range(len(schema_tokens) - 1)):
                continue
            embs = torch.stack(schema_emb_list)
            count_records.append((sample_idx, task_idx, embs))
            count_inputs.append(embs[0])
    counts = self.count_pred(torch.stack(count_inputs)).argmax(dim=-1).tolist() if count_inputs else []

    # instance 0 of every entity pair with a positive count, in one GRU step
    entity_rows = [embs[1:] for (sample_idx, task_idx, embs), count in zip(count_records, counts)
                   if int(count) > 0 and batch.task_types[sample_idx][task_idx] == "entities"]
    entity_projections = iter(torch.split(self.count_embed(torch.cat(entity_rows), 1),
                                          [rows.shape[0] for rows in entity_rows], dim=1)) if entity_rows else iter(())

    query_chunks = [[] for _ in range(batch_size)]
    query_slices = {}
    query_offsets = [0] * batch_size
    for (sample_idx, task_idx, embs), pred_count in zip(count_records, counts):
        pred_count = int(pred_count)
        pred_counts[sample_idx][task_idx] = pred_count
        if pred_count <= 0:
            continue
        if batch.task_types[sample_idx][task_idx] == "entities":
            struct_proj = next(entity_projections)
        else:
            struct_proj = self.count_embed(embs[1:], pred_count)
        if struct_proj.numel() == 0:
            continue
        count_dim, field_dim, hidden = struct_proj.shape
        flat_queries = struct_proj.reshape(count_dim * field_dim, hidden)
        start = query_offsets[sample_idx]
        stop = start + flat_queries.shape[0]
        query_offsets[sample_idx] = stop
        query_chunks[sample_idx].append(flat_queries)
        query_slices[(sample_idx, task_idx)] = (start, stop, count_dim, field_dim)

    active_indices = [i for i, chunks in enumerate(query_chunks) if chunks]
    if not active_indices:
        return raw_logits, pred_counts
    direct_logits = self.compute_span_scores_batched([all_token_embs[i] for i in active_indices],
                                                     [torch.cat(query_chunks[i], dim=0) for i in active_indices],
                                                     mask_invalid=True)
    logits_by_sample = dict(zip(active_indices, direct_logits))
    for (sample_idx, task_idx), (start, stop, count_dim, field_dim) in query_slices.items():
        task_logits = logits_by_sample[sample_idx][start:stop]
        raw_logits[sample_idx][task_idx] = task_logits.reshape(count_dim, field_dim, task_logits.shape[-2],
                                                               task_logits.shape[-1])
    return raw_logits, pred_counts


# GLiNER2 splits the same text into words six times per request (windowing and collation, in each of the three label
# passes). Its default WhitespaceTokenSplitter is stateless (one fixed regex): its output depends only on (text,
# lower), so each text is split once and the immutable result reused; any other splitter is untouched.
SPLIT_CACHE = os.getenv("DETECTION_SPLIT_CACHE", "1") == "1"
if SPLIT_CACHE:
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter

    _split_words = WhitespaceTokenSplitter.__call__

    # 1024 texts cover both serving threads' runs across the three passes (about 30 MB at most)
    @functools.lru_cache(maxsize=1024)
    def _words(text, lower):
        return tuple(_split_words(WhitespaceTokenSplitter(), text, lower))

    def _cached_words(self, text, lower=True):
        if type(self) is not WhitespaceTokenSplitter:  # a subclass may change the pattern
            return _split_words(self, text, lower)
        return iter(_words(text, lower))

    WhitespaceTokenSplitter.__call__ = _cached_words

# batch_extract calls model.eval() on every call, walking every submodule; serving models are put in eval mode once
# at load and never trained, so later calls return at once
SKIP_EVAL = os.getenv("DETECTION_SKIP_EVAL", "1") == "1"
SHARE_WEIGHTS = os.getenv("DETECTION_SHARE_WEIGHTS", "1") == "1"


# fp32 reference (tests only): no fp16 weights and plain attention, to measure every serving path against it
FP32_REFERENCE = os.getenv("DETECTION_PRECISION") == "fp32"


def strategy_passes(labels):
    if DETECTION_STRATEGY == "v25":
        return [("v25", list(labels), {})]
    if DETECTION_STRATEGY == "hybrid":
        covered = [label for label in labels if label in PII_LABELS]
        rest = [label for label in labels if label not in PII_LABELS]
        return [(key, group, {}) for key, group in (("pii", covered), ("legacy", rest)) if group]
    passes = [("legacy", group, {}) for group in label_passes(labels)]
    if "City" in labels and CITY_PASS:
        passes.append(("legacy", ["City"], {"City": CITY_ALONE_DESCRIPTION}))
    return passes


def _labels_with_desc(labels):
    """Convert a list of labels into the dict form GLiNER2 accepts,
    using LABEL_DESCRIPTIONS where available."""
    if isinstance(labels, dict):
        return labels
    return {label: LABEL_DESCRIPTIONS.get(label, label) for label in labels}


def _flatten(gliner2_result, labels):
    flat = []
    for label in labels:
        for ent in gliner2_result.get('entities', {}).get(label, []):
            flat.append({
                'text': ent['text'],
                'label': label,
                'start': ent.get('start', 0),
                'end': ent.get('end', 0),
                'score': ent.get('confidence', 1.0),
            })
    return flat


def detect_entities(text, labels=["Person", "Organization"], threshold=0.75):
    global model
    result = model.extract_entities(
        text, _labels_with_desc(labels),
        include_confidence=True, include_spans=True,
    )
    flat = _flatten(result, labels)
    return [e for e in flat if e['score'] >= threshold]


class Gliner2NER:
    def __init__(self):
        pass

    def loadmodel(self):
        global model
        model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
        # from_pretrained loads onto the CPU; detection belongs on the GPU when the machine has one
        if torch.cuda.is_available():
            model = model.to("cuda")

    def _extract(self, text, label, threshold=0.65):
        global model
        result = model.extract_entities(
            text, _labels_with_desc([label]), include_confidence=True, threshold=threshold
        )
        return [e['text'] for e in result.get('entities', {}).get(label, [])]

    def detect_names(self, text, threshold=0.65):
        global model
        return self._extract(text, "Person", threshold=threshold)

    def detect_organizations(self, text, threshold=0.85):
        global model
        return self._extract(text, "Organization", threshold=threshold)

    def detect_streets(self, text, threshold=0.65):
        global model
        streets = self._extract(text, "Street", threshold=threshold)
        return [re.sub(r'\b\d+\b', '', s).strip() for s in streets]

    def detect_cities(self, text, threshold=0.65):
        global model
        return self._extract(text, "City", threshold=threshold)

    def detect_dates(self, text, threshold=0.65):
        global model
        return self._extract(text, "Dates", threshold=threshold)

    def detect_insurance_numbers(self, text, threshold=0.65):
        global model
        return self._extract(text, "Insurance Number", threshold=threshold)


    # many texts per GPU call, shortest first so each batch pads little, in the passes the strategy chooses; a text can be
    # a whole document, and the spans of every pass are resolved together per text (resolve_spans), since GLiNER2 only
    # resolves overlaps within a label
    def detect_batch(self, texts, labels, device="cuda", batch_size=16):
        # found = [{label: [] for label in labels} for _ in texts]
        spans = [[] for _ in texts]
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        ordered = [texts[index] for index in order]
        for checkpoint_key, group, descriptions in strategy_passes(labels):
            if checkpoint_key == "pii":
                entity_types = [PII_LABELS[label] for label in group]
                back = {PII_LABELS[label]: label for label in group}
            else:
                entity_types = {**_labels_with_desc(group), **descriptions}
                back = {label: label for label in group}
            # results = checkpoint(checkpoint_key, device).batch_extract_entities(
            #     ordered, entity_types, batch_size=batch_size, include_confidence=True, include_spans=True,
            #     threshold=min(LABEL_THRESHOLDS.get(label, DEFAULT_THRESHOLD) for label in group),
            # )
            results = checkpoint(checkpoint_key, device).batch_extract_entities_long(
                ordered, entity_types, batch_size=batch_size, include_confidence=True, include_spans=True,
                threshold=min(LABEL_THRESHOLDS.get(label, DEFAULT_THRESHOLD) for label in group),
                chunk_size=WINDOW_WORDS, chunk_overlap=WINDOW_OVERLAP_WORDS,
            )
            candidates = []
            for index, result in zip(order, results):
                for entity_type, entities in result.get("entities", {}).items():
                    label = back.get(entity_type)
                    if label is None:
                        continue
                    minimum = LABEL_THRESHOLDS.get(label, DEFAULT_THRESHOLD)
                    for entity in entities:
                        confidence = entity.get("confidence", 1.0)
                        if confidence < minimum:
                            continue
                        start, end = entity["start"], entity["end"]
                        pieces = cell_pieces(texts[index], start, end)
                        if len(pieces) >= 2:
                            candidates.extend((index, label, start, end) for start, end in pieces)
                        else:
                            spans[index].append((label, start, end, confidence))
                    # kept = [e["text"] for e in entities if e.get("confidence", 1.0) >= minimum]
                    # if label == "Street":
                    #     kept = [re.sub(r"\b\d+\b", "", street).strip() for street in kept]
                    # found[index][label] = list(dict.fromkeys(found[index][label] + kept))
            if candidates:
                confirmations = checkpoint(checkpoint_key, device).batch_extract_entities(
                    [texts[index][start:end] for index, label, start, end in candidates],
                    entity_types, batch_size=batch_size, include_confidence=True, include_spans=True,
                    threshold=min(LABEL_THRESHOLDS.get(label, DEFAULT_THRESHOLD) for label in group),
                )
                for (index, label, piece_start, piece_end), result in zip(candidates, confirmations):
                    for entity_type, entities in result.get("entities", {}).items():
                        if back.get(entity_type) != label:
                            continue
                        for entity in entities:
                            confidence = entity.get("confidence", 1.0)
                            start, end = entity.get("start"), entity.get("end")
                            if (
                                confidence >= LABEL_THRESHOLDS.get(label, DEFAULT_THRESHOLD)
                                and start is not None and end is not None
                                and 0 <= start < end <= piece_end - piece_start
                            ):
                                spans[index].append((label, piece_start + start, piece_start + end, confidence))
        found = []
        for text, text_spans in zip(texts, spans):
            values = {label: [] for label in labels}
            occurrences = []
            for label, start, end, confidence in resolve_spans(text, text_spans):
                value = re.sub(r"\b\d+\b", "", text[start:end]).strip() if label == "Street" else text[start:end]
                if value and value not in values[label]:
                    values[label].append(value)
                occurrences.append({"label": label, "start": start, "end": end, "score": round(confidence, 4)})
            found.append({"values": values, "occurrences": occurrences})
        return found


# whole documents are read in GLiNER2's own overlapping word windows (its long-document defaults), so no caller
# chunks or rewrites the text, a value cut by one window is whole in the next, and spans keep document offsets
WINDOW_WORDS = 384
WINDOW_OVERLAP_WORDS = 64


# a value never crosses a table cell: extracted tables join cells with runs of spaces, tabs or line breaks
CELL_GAP = re.compile(r"\s{2,}|[\t\n\r\f\v]")
WORD_SPAN = re.compile(r"\w(?:.*\w)?", re.S)


# a span across cells is two model guesses in one, so each piece must be confirmed on its own
def cell_pieces(text, start, end):
    pieces = []
    piece_start = start
    for gap in [*CELL_GAP.finditer(text, start, end), None]:
        piece_end = gap.start() if gap else end
        word = WORD_SPAN.search(text, piece_start, piece_end)
        if word:
            pieces.append((word.start(), word.end()))
        piece_start = gap.end() if gap else end
    return pieces


# every pass's spans of one text resolved together: cut at table-cell gaps and trimmed to word characters, a span
# inside a larger one gives way so no part of a value is left uncovered (Presidio's rule), and of spans with the same
# boundaries or crossing each other the higher total confidence wins (GLiNER2's own weighted interval scheduling)
def resolve_spans(text, spans):
    pieces = []
    for label, start, end, confidence in spans:
        pieces.extend((label, start, end, confidence) for start, end in cell_pieces(text, start, end))
    rank = {"score": lambda span: span[3], "start": lambda span: span[1], "end": lambda span: span[2]}
    largest = resolve_overlaps(pieces, "longest", **rank)
    return sorted(resolve_overlaps(largest, "flat", **rank), key=lambda span: span[1])


shared_queue = None
shared_queue_lock = threading.Lock()


def detection_queue():
    global shared_queue
    with shared_queue_lock:
        if shared_queue is None:
            shared_queue = DetectionQueue(Gliner2NER().detect_batch, detection_devices())
    return shared_queue
