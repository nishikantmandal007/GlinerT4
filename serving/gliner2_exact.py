# Host-side work cuts for gliner2 2.1.0 inference that return exactly what stock gliner2 returns (same bits, not "close").
#
# count_step0: entity decoding keeps only instance 0 of the count-aware queries, yet gliner2 runs the CountLSTM GRU for
#   pred_count steps per (sample, schema), a Python loop of ~12 small kernels per step. GRU step 0 depends only on
#   step-0 input and the initial state, and a GEMM's output row depends only on that row's input once the shapes (and so
#   the cuBLAS kernel) are the same. So the x-side GEMM and the projector GEMMs keep their full [pred_count * M] shapes,
#   only step 0 of the recurrence runs, and the rows past instance 0 (which are dropped) hold copies of it. Instance 0 is
#   then computed by the same kernels on the same values: bit-identical by construction. Other task types, other count
#   layers and pred_count == 1 keep the stock call. Unlike DETECTION_FAST_COUNT (which batches every pair into one GRU
#   call, changing GEMM shapes, and changed 5 of 895 multilingual chunks), nothing is batched across pairs.
#
# format_cache: every label pass re-tokenizes the same chunk text word by word to build input ids and token mappings.
#   The text part of _format_input_with_mapping depends only on the text words; the schema part only on the schema. The
#   schema part is computed by the stock function (with no text words: the loop is sequential, so that is exactly the
#   prefix of the full result) and the text part is cached per text and shifted by the prefix lengths. Texts containing
#   the literal [SEP_TEXT] marker or a word that tokenizes to nothing take the stock path (it treats both specially).
import functools

import torch
import torch.nn.functional as F

GLINER2_VERSION = "2.1.0"
_installed = {}


def _check_version():
    import gliner2

    if gliner2.__version__ != GLINER2_VERSION:
        raise RuntimeError(f"gliner2_exact mirrors gliner2 {GLINER2_VERSION} internals; found {gliner2.__version__}")


# ---------------------------------------------------------------------------------------------------------------- count


def count_instance0(count_embed, pc_emb, pred_count):
    """Instance 0 of CountLSTM.forward(pc_emb, pred_count), as [1, M, D], with the stock kernels and shapes."""
    from gliner2.layers import CompileSafeGRU, CountLSTM

    gru = getattr(count_embed, "gru", None)
    if type(count_embed) is not CountLSTM or type(gru) is not CompileSafeGRU:
        return count_embed(pc_emb, pred_count)[:1]
    M, D = pc_emb.shape
    steps = min(pred_count, count_embed.max_count)
    if steps <= 1:
        return count_embed(pc_emb, pred_count)[:1]
    # CountLSTM.forward up to the GRU, unchanged
    count_indices = torch.arange(steps, device=pc_emb.device)
    pos_seq = count_embed.pos_embedding(count_indices)
    pos_seq = pos_seq.unsqueeze(1).expand(steps, M, D)
    # CompileSafeGRU.forward: the full-length x-side GEMM, then step 0 of the recurrence only
    gi_all = F.linear(pos_seq, gru.weight_ih_l0, gru.bias_ih_l0)
    h = pc_emb
    gi = gi_all[0]
    gh = F.linear(h, gru.weight_hh_l0, gru.bias_hh_l0)
    i_r, i_z, i_n = gi.chunk(3, dim=-1)
    h_r, h_z, h_n = gh.chunk(3, dim=-1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    h = (1 - z) * n + z * h
    # torch.stack(outputs) is a contiguous [steps, M, D]; rows past step 0 are never read downstream
    output = h.unsqueeze(0).expand(steps, M, D).contiguous()
    return count_embed.projector(torch.cat([output, pc_emb.unsqueeze(0).expand_as(output)], dim=-1))[:1]


def _compute_direct_span_logits_batched(self, batch, all_token_embs, all_schema_embs):
    # A copy of gliner2 2.1.0's ExtractorRuntimeMixin._compute_direct_span_logits_batched; the only change is the
    # entity branch, which asks count_instance0 for the one instance it keeps.
    batch_size = len(batch)
    raw_logits = [[None] * len(batch.task_types[i]) for i in range(batch_size)]
    pred_counts = [[0] * len(batch.task_types[i]) for i in range(batch_size)]

    count_records = []
    count_inputs = []
    for sample_idx in range(batch_size):
        if all_token_embs[sample_idx].numel() == 0:
            continue
        for task_idx, task_type in enumerate(batch.task_types[sample_idx]):
            if task_type == "classifications":
                continue
            if task_idx >= len(all_schema_embs[sample_idx]):
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

    if count_inputs:
        counts = self.count_pred(torch.stack(count_inputs)).argmax(dim=-1).tolist()
    else:
        counts = []

    query_chunks = [[] for _ in range(batch_size)]
    query_slices = {}
    query_offsets = [0] * batch_size

    for (sample_idx, task_idx, embs), pred_count in zip(count_records, counts):
        pred_count = int(pred_count)
        pred_counts[sample_idx][task_idx] = pred_count
        if pred_count <= 0:
            continue

        if batch.task_types[sample_idx][task_idx] == "entities":
            # Entity decoding consumes only structural instance 0.
            struct_proj = count_instance0(self.count_embed, embs[1:], pred_count)
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

    active_tokens = [all_token_embs[i] for i in active_indices]
    active_queries = [torch.cat(query_chunks[i], dim=0) for i in active_indices]
    direct_logits = self.compute_span_scores_batched(active_tokens, active_queries, mask_invalid=True)

    logits_by_sample = dict(zip(active_indices, direct_logits))
    for (sample_idx, task_idx), (start, stop, count_dim, field_dim) in query_slices.items():
        task_logits = logits_by_sample[sample_idx][start:stop]
        raw_logits[sample_idx][task_idx] = task_logits.reshape(
            count_dim, field_dim, task_logits.shape[-2], task_logits.shape[-1]
        )

    return raw_logits, pred_counts


# --------------------------------------------------------------------------------------------------------------- format

TEXT_CACHE_SIZE = 2048  # chunk texts per processor; a few hundred KB each at most


def _text_part(processor, text_tokens):
    """The text words' subwords, ids, per-word subword counts and first-subword offsets; None for the stock path."""
    if processor.SEP_TEXT in text_tokens:
        return None
    token_cache, tokenize = processor._token_cache, processor._tokenize_cached
    subwords, counts, firsts = [], [], []
    for token in text_tokens:
        sub_tokens = token_cache[token] if token in token_cache else tokenize(token)
        if not sub_tokens:
            return None
        firsts.append(len(subwords))
        counts.append(len(sub_tokens))
        subwords.extend(sub_tokens)
    return tuple(subwords), tuple(processor.tokenizer.convert_tokens_to_ids(subwords)), tuple(counts), tuple(firsts)


def _make_format(stock):
    def _format_input_with_mapping(self, schema_tokens_list, text_tokens):
        cache = self.__dict__.get("_exact_text_parts")
        if cache is None:
            cache = self.__dict__["_exact_text_parts"] = functools.lru_cache(maxsize=TEXT_CACHE_SIZE)(
                functools.partial(_text_part, self))
        part = cache(tuple(text_tokens))
        if part is None:
            return stock(self, schema_tokens_list, text_tokens)
        subwords, ids, counts, firsts = part
        prefix = stock(self, schema_tokens_list, [])
        base_orig = sum(len(struct) for struct in schema_tokens_list) + max(len(schema_tokens_list) - 1, 0) + 1
        base_sub = len(prefix["subword_list"])
        text_schema_idx = len(schema_tokens_list)
        mappings = prefix["mapped_indices"]
        for index, count in enumerate(counts):
            mappings.extend([("text", base_orig + index, text_schema_idx)] * count)
        return {
            "input_ids": list(prefix["input_ids"]) + list(ids),
            "mapped_indices": mappings,
            "subword_list": prefix["subword_list"] + list(subwords),
            "text_word_first_positions": [base_sub + first for first in firsts],
            "schema_special_positions": prefix["schema_special_positions"],
        }

    return _format_input_with_mapping


# -------------------------------------------------------------------------------------------------------------- install


def install(count_step0=True, format_cache=True):
    _check_version()
    from gliner2.inference.runtime import ExtractorRuntimeMixin
    from gliner2.processor import SchemaTransformer

    uninstall()
    if count_step0:
        _installed[(ExtractorRuntimeMixin, "_compute_direct_span_logits_batched")] = \
            ExtractorRuntimeMixin._compute_direct_span_logits_batched
        ExtractorRuntimeMixin._compute_direct_span_logits_batched = _compute_direct_span_logits_batched
    if format_cache:
        stock = SchemaTransformer._format_input_with_mapping
        _installed[(SchemaTransformer, "_format_input_with_mapping")] = stock
        SchemaTransformer._format_input_with_mapping = _make_format(stock)


def uninstall():
    for (owner, name), original in _installed.items():
        setattr(owner, name, original)
    _installed.clear()
