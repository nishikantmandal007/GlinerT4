import contextlib

import torch


def _streams(model, device):
    if device.type != "cuda":
        return None, None
    streams = model.__dict__.get("_glinert4_streams")
    if streams is None or streams[0].device != device:
        streams = model.__dict__["_glinert4_streams"] = (torch.cuda.Stream(device), torch.cuda.Stream(device))
    return streams


def _on(stream):
    return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()


def _batches(model, job):
    from gliner2.training.trainer import ExtractorCollator

    texts, schemas, batch_size, max_len = job["texts"], job["schemas"], job["batch_size"], job.get("max_len")
    if isinstance(schemas, list) and len(schemas) != len(texts):
        raise ValueError(f"schema count ({len(schemas)}) != text count ({len(texts)})")
    schema_list = schemas if isinstance(schemas, list) else [schemas] * len(texts)
    schema_dicts, metadata_list = model._build_schema_dicts_and_metadata(schema_list)
    overlap_policy = job.get("overlap_policy")
    if overlap_policy is not None:
        from gliner2.inference.overlap import normalize_overlap_policy

        overlap_policy = normalize_overlap_policy(overlap_policy)
        for metadata in metadata_list:
            metadata["_overlap_policy"] = overlap_policy
    if max_len is None:
        if getattr(model, "_inference_collator", None) is None:
            model._inference_collator = ExtractorCollator(model.processor, is_training=False,
                                                          architecture=model.architecture)
        collator = model._inference_collator
    else:
        collator = ExtractorCollator(model.processor, is_training=False, max_len=max_len,
                                     architecture=model.architecture)
    dataset = list(zip(texts, schema_dicts))
    step = len(dataset) if len(dataset) <= batch_size else batch_size
    for start in range(0, len(dataset), step):
        yield collator(dataset[start:start + step]), metadata_list[start:start + step]


@torch.inference_mode()
def extract_many(model, jobs):
    model.eval()
    model.processor.change_mode(is_training=False)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    encoder_stream, copy_stream = _streams(model, device)
    serving_stream = torch.cuda.current_stream(device) if encoder_stream is not None else None
    results = [[] for _ in jobs]

    def work():
        for index, job in enumerate(jobs):
            if not job["texts"]:
                continue
            if job["batch_size"] <= 0:
                raise ValueError("batch_size must be greater than 0")
            for batch, metadata in _batches(model, job):
                yield index, batch, metadata

    def launch(item):
        index, batch, metadata = item
        with _on(copy_stream):
            batch = batch.to(device, dtype if dtype != torch.float32 else None)
        done = None
        if encoder_stream is not None:
            encoder_stream.wait_stream(copy_stream)
        with _on(encoder_stream):
            states = model.encode_tokens(batch, None)
        if encoder_stream is not None:
            for tensor in [*vars(batch).values(), states]:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    tensor.record_stream(serving_stream)
            done = torch.cuda.Event()
            done.record(encoder_stream)
        return index, batch, metadata, states, done

    def finish(launched):
        index, batch, metadata, states, done = launched
        job = jobs[index]
        if done is not None:
            serving_stream.wait_event(done)
        model.encode_tokens = lambda batch_, hidden_states=None: states
        thresholds = [job["threshold"]] + ([job["candidate_threshold"]] if job.get("candidate_threshold") else [])
        decoded = []
        try:
            for threshold in thresholds:
                batch_results = model._extract_from_batch(batch, threshold, metadata, job["include_confidence"],
                                                          job["include_spans"], hidden_states=None)
                decoded.append([model.format_results(result, job["include_confidence"],
                                                     meta.get("relation_order", []),
                                                     meta.get("classification_tasks", []))
                                for result, meta in zip(batch_results, metadata)])
        finally:
            del model.encode_tokens
        results[index].extend(decoded[0] if len(decoded) == 1 else zip(*decoded))

    pending = None
    for item in work():
        launched = launch(item)
        if pending is not None:
            finish(pending)
        pending = launched
    if pending is not None:
        finish(pending)
    return results


def extract_entities_long_many(model, requests, batch_size=8, include_confidence=False, include_spans=False,
                               chunk_size=384, chunk_overlap=64, overlap_policy=None, candidate_threshold=None):
    from gliner2.inference.chunking import merge_chunk_results, split_text_into_chunks
    from gliner2.processing.word_splitter import word_splitter_from

    prepared, jobs = [], []
    for texts, entity_types, threshold in requests:
        schema = model.create_schema().entities(entity_types)
        chunk_texts, chunk_schemas, doc_chunks, counts = [], [], [], []
        for text in texts:
            chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                            word_splitter=word_splitter_from(model))
            doc_chunks.append(chunks)
            counts.append(len(chunks))
            chunk_texts.extend(chunk.text for chunk in chunks)
            chunk_schemas.extend(schema for _ in chunks)
        prepared.append((texts, schema, doc_chunks, counts))
        jobs.append({"texts": chunk_texts, "schemas": chunk_schemas, "batch_size": batch_size, "threshold": threshold,
                     "include_confidence": True, "include_spans": True, "max_len": chunk_size,
                     "overlap_policy": overlap_policy, "candidate_threshold": candidate_threshold})
    merged_all = []
    for (texts, schema, doc_chunks, counts), results in zip(prepared, extract_many(model, jobs)):

        def merge(text, chunks, chunk_results):
            return merge_chunk_results(text, chunks, chunk_results, include_confidence=include_confidence,
                                       include_spans=include_spans,
                                       scalar_entity_labels=model._scalar_entity_labels(schema),
                                       overlap_policy=model._resolved_overlap_policy(overlap_policy))

        merged, offset = [], 0
        for text, chunks, count in zip(texts, doc_chunks, counts):
            window = results[offset:offset + count]
            if candidate_threshold:
                merged.append((merge(text, chunks, [main for main, _ in window]),
                               merge(text, chunks, [extra for _, extra in window])))
            else:
                merged.append(merge(text, chunks, window))
            offset += count
        merged_all.append(merged)
    return merged_all
