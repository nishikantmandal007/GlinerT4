# CPU/GPU overlap for gliner2 2.1.0 batch extraction, with no change to model math or results.
#
# gliner2's batch_extract collates batch i+1 (word splitting, schema prompt formatting, tokenization, padding) on the
# serving thread only after batch i is fully decoded, so the GPU sits idle while Python builds the next batch. Here a
# helper thread runs the same collator over the same batches in the same order, up to `depth` batches ahead, while the
# serving thread runs the model. The collator only reads the processor state the serving thread uses (the tokenizer,
# special-token ids, its thread-safe lru caches), so every batch is the batch the stock loop would build, and outputs
# are identical by construction. bench/pipeline_bench.py checks that against the stock path on the real corpus.
#
# The GIL is released while the serving thread waits on the GPU (the encoder's host syncs), which is when the helper
# runs; while the serving thread is busy in Python the two compete for the GIL, so depth stays small.
import queue
import threading

import torch

GLINER2_VERSION = "2.1.0"
_STOP = object()


class _Prefetch:
    """Iterate `batches` on a helper thread, at most `depth` collated batches ahead of the consumer."""

    def __init__(self, batches, depth):
        self._queue = queue.Queue(maxsize=depth)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._fill, args=(iter(batches),), name="gliner2-collate", daemon=True)
        self._thread.start()

    def _put(self, item):
        while not self._closed.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _fill(self, batches):
        try:
            for batch in batches:
                if not self._put(("batch", batch)):
                    return
        except BaseException as error:  # handed to the consumer, raised there
            self._put(("error", error))
            return
        self._put(("done", _STOP))

    def __iter__(self):
        try:
            while True:
                kind, item = self._queue.get()
                if kind == "error":
                    raise item
                if kind == "done":
                    return
                yield item
        finally:
            self.close()

    def close(self):
        self._closed.set()
        self._thread.join()


def make_batch_extract(depth=2):
    # A copy of gliner2 2.1.0's ExtractorRuntimeMixin.batch_extract; the only change is that the per-batch collation
    # is iterated through _Prefetch.
    @torch.inference_mode()
    def batch_extract(
        self,
        texts,
        schemas,
        batch_size=8,
        threshold=0.5,
        num_workers=0,
        format_results=True,
        include_confidence=False,
        include_spans=False,
        max_len=None,
        overlap_policy=None,
        hidden_states=None,
    ):
        from gliner2.inference.overlap import normalize_overlap_policy

        if not texts:
            return []
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if hidden_states is not None and len(hidden_states) != len(texts):
            raise ValueError(f"hidden_states count ({len(hidden_states)}) != text count ({len(texts)})")

        self.eval()
        self.processor.change_mode(is_training=False)

        if isinstance(schemas, list):
            if len(schemas) != len(texts):
                raise ValueError(f"Schema count ({len(schemas)}) != text count ({len(texts)})")
            schema_list = schemas
        else:
            schema_list = [schemas] * len(texts)

        schema_dicts, metadata_list = self._build_schema_dicts_and_metadata(schema_list)
        if overlap_policy is not None:
            overlap_policy = normalize_overlap_policy(overlap_policy)
            for metadata in metadata_list:
                metadata["_overlap_policy"] = overlap_policy

        dataset = list(zip(texts, schema_dicts))

        from torch.utils.data import DataLoader

        from gliner2.training.trainer import ExtractorCollator

        if max_len is None:
            if getattr(self, "_inference_collator", None) is None:
                self._inference_collator = ExtractorCollator(
                    self.processor, is_training=False, architecture=self.architecture
                )
            collator = self._inference_collator
        else:
            collator = ExtractorCollator(
                self.processor, is_training=False, max_len=max_len, architecture=self.architecture
            )

        if hidden_states is not None or (len(dataset) <= batch_size and num_workers == 0):
            batches = [collator(dataset)]
        else:
            batches = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True if torch.cuda.is_available() else False,
            )
            # one batch has nothing to overlap with; worker processes already collate off this thread
            if num_workers == 0 and depth > 0:
                batches = _Prefetch(batches, depth)

        all_results = []
        sample_idx = 0
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        for batch in batches:
            batch = batch.to(device, dtype if dtype != torch.float32 else None)
            batch_results = self._extract_from_batch(
                batch,
                threshold,
                metadata_list[sample_idx : sample_idx + len(batch)],
                include_confidence,
                include_spans,
                hidden_states=hidden_states,
            )

            if format_results:
                for i, result in enumerate(batch_results):
                    meta = metadata_list[sample_idx + i]
                    requested_relations = meta.get("relation_order", [])
                    classification_tasks = meta.get("classification_tasks", [])
                    batch_results[i] = self.format_results(
                        result, include_confidence, requested_relations, classification_tasks
                    )

            all_results.extend(batch_results)
            sample_idx += len(batch)

        return all_results

    batch_extract.prefetch_depth = depth
    return batch_extract


_original = None


def install(depth=2):
    """Patch gliner2's batch_extract (and so batch_extract_entities and the *_long helpers) to prefetch collation."""
    global _original
    import gliner2
    from gliner2.inference.runtime import ExtractorRuntimeMixin

    if gliner2.__version__ != GLINER2_VERSION:
        raise RuntimeError(f"gliner2_pipeline copies gliner2 {GLINER2_VERSION}'s batch_extract; found {gliner2.__version__}")
    if _original is None:
        _original = ExtractorRuntimeMixin.batch_extract
    ExtractorRuntimeMixin.batch_extract = make_batch_extract(depth)


def uninstall():
    global _original
    from gliner2.inference.runtime import ExtractorRuntimeMixin

    if _original is not None:
        ExtractorRuntimeMixin.batch_extract = _original
        _original = None
