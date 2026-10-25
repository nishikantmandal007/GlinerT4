# Single-thread software pipeline for gliner2 2.1.0 extraction, with results identical to stock gliner2.
#
# Stock batch_extract alternates: build a batch on the CPU (GPU idle), run the encoder (CPU waits), then the span head
# and decoding (small GPU work, mostly CPU). On the shipping path the GPU does ~2.8 s and the CPU ~1.3 s per corpus run,
# and the wall time is about their sum. Two serving threads cannot overlap them on a host with one physical core (the
# two threads share it and the GIL). This pipeline overlaps them in one thread:
#
#     collate(i+1) | launch encoder(i+1) on stream E | finish(i) on the serving stream | collate(i+2) | ...
#
# finish(i) is gliner2's own _extract_from_batch for batch i, with encode_tokens returning the encoder output already
# computed on E (the serving stream waits for that encoder's event only, so the encoder of batch i+1 keeps running
# while the CPU decodes batch i). Host-to-device copies go through a copy stream, so a blocking copy never waits behind
# a running encoder. Every batch is built, encoded and decoded by the same code, kernels and shapes as stock: only the
# order in which the CPU issues the work changes, so the outputs are identical (checked in tests/test_gliner2_pipelined.py
# and on the T4 against the stock path).
#
# extract_many runs several batch_extract jobs (e.g. every label pass of a detection run) as one pipeline, so even a
# run that is one batch per pass overlaps its passes.
import contextlib

import torch

GLINER2_VERSION = "2.1.0"


def _check_version():
    import gliner2

    if gliner2.__version__ != GLINER2_VERSION:
        raise RuntimeError(f"gliner2_pipelined mirrors gliner2 {GLINER2_VERSION} internals; found {gliner2.__version__}")


def _streams(model, device):
    if device.type != "cuda":
        return None, None
    streams = model.__dict__.get("_pipeline_streams")
    if streams is None or streams[0].device != device:
        streams = model.__dict__["_pipeline_streams"] = (torch.cuda.Stream(device), torch.cuda.Stream(device))
    return streams


def _batches(model, job):
    """Collated batches of one job, exactly as batch_extract would build them (same collator, same slices)."""
    from gliner2.training.trainer import ExtractorCollator

    texts, schemas, batch_size, max_len = job["texts"], job["schemas"], job["batch_size"], job.get("max_len")
    schema_list = schemas if isinstance(schemas, list) else [schemas] * len(texts)
    if isinstance(schemas, list) and len(schemas) != len(texts):
        raise ValueError(f"Schema count ({len(schemas)}) != text count ({len(texts)})")
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
    # batch_extract: one collate call for a dataset that fits a batch, else DataLoader(batch_size, no shuffle) slices
    step = len(dataset) if len(dataset) <= batch_size else batch_size
    for start in range(0, len(dataset), step):
        yield collator(dataset[start:start + step]), metadata_list[start:start + step]


@torch.inference_mode()
def extract_many(model, jobs):
    """[model.batch_extract(job...) for job in jobs], computed as one pipeline; jobs are dicts with texts, schemas,
    batch_size, threshold, include_confidence, include_spans and optionally max_len, overlap_policy."""
    _check_version()
    model.eval()
    model.processor.change_mode(is_training=False)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    encoder_stream, copy_stream = _streams(model, device)
    serving_stream = torch.cuda.current_stream(device) if encoder_stream is not None else None
    results = [[] for _ in jobs]

    def on(stream):
        return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()

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
        with on(copy_stream):
            batch = batch.to(device, dtype if dtype != torch.float32 else None)
        if encoder_stream is not None:
            encoder_stream.wait_stream(copy_stream)
        with on(encoder_stream):
            states = model.encode_tokens(batch, None)
        if encoder_stream is not None:
            # made on the copy and encoder streams, read on the serving stream: keep their memory until that is done
            for tensor in [*vars(batch).values(), states]:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    tensor.record_stream(serving_stream)
            done = torch.cuda.Event()
            done.record(encoder_stream)
        else:
            done = None
        return index, batch, metadata, states, done

    def finish(launched):
        index, batch, metadata, states, done = launched
        job = jobs[index]
        if done is not None:
            serving_stream.wait_event(done)
        model.encode_tokens = lambda batch_, hidden_states=None: states  # instance attribute, removed below
        try:
            batch_results = model._extract_from_batch(batch, job["threshold"], metadata, job["include_confidence"],
                                                      job["include_spans"], hidden_states=None)
        finally:
            del model.encode_tokens
        for i, result in enumerate(batch_results):
            meta = metadata[i]
            batch_results[i] = model.format_results(result, job["include_confidence"], meta.get("relation_order", []),
                                                    meta.get("classification_tasks", []))
        results[index].extend(batch_results)

    pending = None
    for item in work():  # collating the next batch here overlaps the encoder launched in the previous iteration
        launched = launch(item)
        if pending is not None:
            finish(pending)
        pending = launched
    if pending is not None:
        finish(pending)
    return results


def extract_entities_long_many(model, requests, batch_size=8, threshold=0.5, include_confidence=False,
                               include_spans=False, chunk_size=384, chunk_overlap=64, overlap_policy=None):
    """[model.batch_extract_entities_long(texts, entity_types, ...) for texts, entity_types in requests], as one
    pipeline. A request may be (texts, entity_types, threshold) to override the threshold for that request. A copy of gliner2 2.1.0's batch_extract_entities_long / batch_extract_long with the inner batch_extract
    calls of every request run through extract_many."""
    from gliner2.inference.chunking import merge_chunk_results, split_text_into_chunks
    from gliner2.processing.word_splitter import word_splitter_from

    prepared, jobs = [], []
    for request in requests:
        texts, entity_types = request[:2]
        request_threshold = request[2] if len(request) > 2 else threshold
        schema = model.create_schema().entities(entity_types)
        chunk_texts, chunk_schemas, doc_chunks, counts = [], [], [], []
        for text in texts:
            chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                            word_splitter=word_splitter_from(model))
            doc_chunks.append(chunks)
            counts.append(len(chunks))
            for chunk in chunks:
                chunk_texts.append(chunk.text)
                chunk_schemas.append(schema)
        prepared.append((texts, schema, doc_chunks, counts))
        jobs.append({"texts": chunk_texts, "schemas": chunk_schemas, "batch_size": batch_size,
                     "threshold": request_threshold,
                     "include_confidence": True, "include_spans": True, "max_len": chunk_size,
                     "overlap_policy": overlap_policy})
    chunk_results = extract_many(model, jobs)
    merged_all = []
    for (texts, schema, doc_chunks, counts), results in zip(prepared, chunk_results):
        merged, offset = [], 0
        for text, chunks, count in zip(texts, doc_chunks, counts):
            merged.append(merge_chunk_results(text, chunks, results[offset:offset + count],
                                              include_confidence=include_confidence, include_spans=include_spans,
                                              scalar_entity_labels=model._scalar_entity_labels(schema),
                                              overlap_policy=model._resolved_overlap_policy(overlap_policy)))
            offset += count
        merged_all.append(merged)
    return merged_all
