# Further GLiNER2 optimization with exact-output gates — 2026-10-24

New experimental work: an Ampere/Ada asynchronous-copy attention variant plus a bulk vocabulary lookup. On the already prepared custom path, the combined new changes measured +3.8% on A100 and +3.2% on L4. Encoder gains are larger. These measurements do not credit the preparation optimizations already developed in the earlier work as new gains.

## Accuracy reference

Each candidate must preserve its own unchanged reference exactly: new CUDA is compared with original custom v4; native Triton plus preparation/lookup is compared with unchanged native Triton. Returned confidence scores, spans, values, and ordering are included. Raw attention and encoder comparisons use byte views, so no tolerance or signed-zero exception is allowed.

The existing custom implementation is still not identical to native Triton. This experiment preserves that custom implementation; it does not repair its earlier native-versus-custom differences or claim a new ground-truth accuracy improvement. For a requirement of exact native-Triton behavior, retain native Triton attention and apply only the exact preparation/lookup changes.

## Historical T4 comparison: different GPU counts and serving settings

The historical approximately 100 chunks/s HTTP result used **two T4 GPUs combined**, with two serving streams per GPU, fast-count enabled, and forced packing. The older 114–120 chunks/s direct result was also the aggregate of separate GPU processes. Current Colab results use **one GPU and one serving thread**, fast-count disabled, and the native automatic packing policy. They measure changes inside that setup; they are not matched hardware-capacity tests. The current single-GPU Colab figures must not be presented as the maximum serving throughput of L4 or A100 or as a controlled comparison with the aggregate T4 headline.

## Incremental extraction gains beyond existing preparation

Five rotating-order measurement rounds per configuration; medians. Same 290 unique approximately 1KB synthetic realistic text chunks, batch 16, FP16, unchanged three label passes. Word-split cache is cleared before each timed run. No entity-result cache is installed.

| GPU | Current custom + existing prep | + new bulk lookup | + new async copy and lookup | Combined new gain |
|---|---:|---:|---:|---:|
| A100 | 114.27 chunks/s | 116.29 chunks/s | 118.62 chunks/s | 3.81% |
| L4 | 78.17 chunks/s | 79.01 chunks/s | 80.68 chunks/s | 3.20% |

| GPU | Native Triton + existing prep | + new bulk lookup | Incremental gain |
|---|---:|---:|---:|
| A100 | 109.08 chunks/s | 111.03 chunks/s | 1.78% |
| L4 | 68.78 chunks/s | 69.52 chunks/s | 1.08% |

These are in-process detector measurements, not HTTP RPM, whole-document throughput, or a long-running load test. Only five measurements per path were made. Whole-extraction gains are modest; shared-host and clock variance remain relevant. The candidate was faster than the prepared custom reference in all five paired rounds on each GPU, but these samples do not establish a production SLA.

## Encoder results

Five rotating rounds with three forwards per timing, median CUDA event times. Batch 16; right-padded examples retain different real lengths per row. Packing policy is unchanged.

| GPU | Tokens | Padding | Original custom ms | Async-copy ms | Speedup |
|---|---:|---|---:|---:|---:|
| A100 | 256 | False | 7.725 | 7.367 | 1.049x |
| A100 | 256 | True | 6.736 | 6.700 | 1.005x |
| A100 | 512 | False | 18.915 | 16.993 | 1.113x |
| A100 | 512 | True | 10.688 | 10.070 | 1.061x |
| A100 | 1024 | False | 51.332 | 46.223 | 1.111x |
| A100 | 1024 | True | 25.930 | 23.724 | 1.093x |
| L4 | 256 | False | 25.113 | 23.246 | 1.080x |
| L4 | 256 | True | 21.840 | 21.053 | 1.037x |
| L4 | 512 | False | 59.040 | 53.751 | 1.098x |
| L4 | 512 | True | 34.504 | 32.564 | 1.060x |
| L4 | 1024 | False | 147.774 | 139.341 | 1.061x |
| L4 | 1024 | True | 81.747 | 79.102 | 1.033x |

The simple copy variant is preferred over the larger double-buffered prototype: it has a smaller shared-memory footprint and was faster on the complete A100 extraction comparison. The larger prototype is retained only in experiment archives. No precision, matrix-multiply order, softmax, score arithmetic, labels, thresholds, model weights, or passes were changed.

## Exactness checks

- A100: 279 byte-identical direct attention cases (three seeds, sequence boundaries, right padding, interior mask holes, all-padding input, packed batches); 16 byte-identical encoder shapes; complete returned results identical on 290 corpus chunks and 21 additional multilingual/edge cases.
- L4: 279 byte-identical direct attention cases (three seeds, sequence boundaries, right padding, interior mask holes, all-padding input, packed batches); 16 byte-identical encoder shapes; complete returned results identical on 290 corpus chunks and 21 additional multilingual/edge cases.

- Bulk lookup: every one of the 250,112 vocabulary entries matches the original lookup. Nine invalid/unknown/type/Unicode fallback cases match the original result or exception. Every timed bulk/preparation round asserts complete result equality against its frozen reference.
- The extra examples are synthetic regression cases across 15 languages plus repetition/empty/whitespace cases. They are not a repeat of the previous 895-example multilingual benchmark and are not a broad accuracy certification.
- Fast-count is excluded: earlier measurements showed it can change FP16 decisions. Rejected 8-warp and 128-register experiments were not repeated as candidates.

## Code and scope

- `async_simple.cu` / `async_simple.patch`: SM80+ experiment replaces register-held prefetch with exact 16-byte global-to-shared asynchronous copies, retaining the floating-point arithmetic verbatim. The tested variants use three-block launch bounds and a larger shared-memory carveout. Build separately for SM80 or SM89. The inherited source header refers to T4 v4; this asynchronous-copy variant itself requires SM80+ and must not replace the T4 build.
- `exact_prep.py`: reusable bulk token-ID installer for an immutable fast inference tokenizer. It exhaustively checks the vocabulary before installing, preserves the original path for unknown/invalid input and vocabulary/backend changes, and returns False for unsupported tokenizers. In-place same-size vocabulary replacement is outside its frozen-tokenizer contract.
- Existing preparation means the previously developed pure word-split cache and skipping repeated eval traversal on an already-evaluated inference model. These were enabled for both controls and new candidates in the incremental table.
- The new lookup batches a vocabulary-list lookup; it is not the earlier rejected per-token cache. Complete vocabulary and extraction results were compared to the original tokenizer.
- Tested software: GLiNER2 2.1.0; DisentangledFlash 1.1.1; PyTorch 2.11.0+cu130; Triton 3.6.0; Transformers 5.20.2; Python 3.13; four Torch CPU threads. Model fastino/gliner2-multi-v1, same cached checkpoint as the parent experiment.
- Original kernel sources, VM services, application code, current branch, and ai-safe are unchanged. No commit, publication, or deployment was made. New code and evidence live in this artifact directory.

## Reproduction

1. Allocate an A100 or L4 with `colab new -s NAME --gpu A100` (or L4). Execute the parent `../bootstrap.py` to install packages and stage the original source/corpus.
2. Execute `run_a100.py` or `run_l4.py`; these embed the asynchronous variants and run strict attention, encoder, and extraction comparisons.
3. Execute `bulk_lookup.py`, then `sanitize.py` and `validate_helper.py` in that same kernel. Save the generated ZIPs from `/content/gliner-ab/` and stop the session.
4. `make_async.py` regenerates the source variants from the saved parent `../payload.json`; its assertions verify that the score/softmax/accumulation/output block is unchanged.

## Limits before deployment

T4 and the actual VM .venv-perf environment have not been revalidated for these new changes. The CUDA variant is SM80+ only; retain the existing T4 path. Native-versus-custom differences remain visible, and there is no claim of universal accuracy from finite tests. New helpers and kernels must pass the same zero-tolerance gates in the intended serving environment before activation.

## Memory checks and exported helper validation

- A100: CUDA memcheck and racecheck passed with zero errors/warnings for the tested copy variants. The probe covers ordinary masks, interior holes, all-padding, and packed batches.
- L4: CUDA memcheck and racecheck passed with zero errors/warnings for the tested copy variants. The probe covers ordinary masks, interior holes, all-padding, and packed batches.

The exported helper itself was checked on the actual L4 tokenizer across all 250,112 vocabulary entries, unknown/invalid inputs, keyword calls, vocabulary growth, and a copied tokenizer. Its returned results also matched native Triton exactly on all 290 corpus chunks and 21 extra cases. These checks include confidence and offsets.

## Cleanup

Both Colab experiment sessions were stopped after downloading and verifying all result archives. No VM connection or production change was made.
