import torch
import torch.nn.functional as F

_installed = {}


def count_instance0(count_embed, pc_emb, pred_count):
    from gliner2.layers import CompileSafeGRU, CountLSTM

    gru = getattr(count_embed, "gru", None)
    if type(count_embed) is not CountLSTM or type(gru) is not CompileSafeGRU:
        return count_embed(pc_emb, pred_count)[:1]
    rows, hidden = pc_emb.shape
    steps = min(pred_count, count_embed.max_count)
    if steps <= 1:
        return count_embed(pc_emb, pred_count)[:1]
    positions = count_embed.pos_embedding(torch.arange(steps, device=pc_emb.device))
    positions = positions.unsqueeze(1).expand(steps, rows, hidden)
    gi = F.linear(positions, gru.weight_ih_l0, gru.bias_ih_l0)[0]
    gh = F.linear(pc_emb, gru.weight_hh_l0, gru.bias_hh_l0)
    i_r, i_z, i_n = gi.chunk(3, dim=-1)
    h_r, h_z, h_n = gh.chunk(3, dim=-1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    state = (1 - z) * n + z * pc_emb
    output = state.unsqueeze(0).expand(steps, rows, hidden).contiguous()
    return count_embed.projector(torch.cat([output, pc_emb.unsqueeze(0).expand_as(output)], dim=-1))[:1]


def span_logits(self, batch, all_token_embs, all_schema_embs):
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
    query_chunks = [[] for _ in range(batch_size)]
    query_slices = {}
    query_offsets = [0] * batch_size
    for (sample_idx, task_idx, embs), pred_count in zip(count_records, counts):
        pred_count = int(pred_count)
        pred_counts[sample_idx][task_idx] = pred_count
        if pred_count <= 0:
            continue
        if batch.task_types[sample_idx][task_idx] == "entities":
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
    active = [i for i, chunks in enumerate(query_chunks) if chunks]
    if not active:
        return raw_logits, pred_counts
    direct = self.compute_span_scores_batched([all_token_embs[i] for i in active],
                                              [torch.cat(query_chunks[i], dim=0) for i in active], mask_invalid=True)
    by_sample = dict(zip(active, direct))
    for (sample_idx, task_idx), (start, stop, count_dim, field_dim) in query_slices.items():
        task_logits = by_sample[sample_idx][start:stop]
        raw_logits[sample_idx][task_idx] = task_logits.reshape(count_dim, field_dim, task_logits.shape[-2],
                                                               task_logits.shape[-1])
    return raw_logits, pred_counts


def install():
    from gliner2.inference.runtime import ExtractorRuntimeMixin

    if ExtractorRuntimeMixin not in _installed:
        _installed[ExtractorRuntimeMixin] = ExtractorRuntimeMixin._compute_direct_span_logits_batched
        ExtractorRuntimeMixin._compute_direct_span_logits_batched = span_logits


def uninstall():
    for cls, stock in _installed.items():
        cls._compute_direct_span_logits_batched = stock
    _installed.clear()
