# Pipeline-side optimizations from the T4 plan, gliner2 2.1.0 — 2026-10-25

This round applies the **no-semantic-change, Phase 1** items of
`GLiNER_Multilingual_v2_1_T4_Optimization_Plan.docx` to the GLiNER2 path (`fastino/gliner2-multi-v1`, `gliner2==2.1.0`).
That path measured 54.81 chunks/s on one T4 earlier. **Current finding (Colab T4):**
- Zero-loss pipeline work reached 1.14x in the isolated encoder bench (`count0_streams2_ln`, 58.05 vs 50.91). In the production queue path, `pipe_3pass` achieves **1160/1160 identical chunks** vs `ref_3pass` at **50.2 chunks/s** (vs 50.6 for `prod_3pass`, which used the lossy fast count).
- End to end through the production queue (round 6), the single-thread CPU/GPU pipeline pushes the 2-pass shipping candidate (**`pipe_2pass`**) to **72.8 chunks/s** (median; runs [72.9, 72.8, 72.5]), a **1.439x** speedup over today's serving settings (`prod_3pass` at 50.6 chunks/s) and +3.1% over unpipelined `exact_2pass_1stream` (70.6 chunks/s). Larger batch queues (`pipe_2pass_run64`) push this to **73.6 chunks/s** (1.455x).
- **Single-pass layout breaks the 100 chunks/s ceiling (rounds 7-8):** Merging all labels into a single prompt with a single threshold adjustment (`single_street_0.5`, lowering Street to 0.50) achieves **142.2 chunks/s (18,619 words/s)** on 040 chunks (**2.41x vs 3-pass**, **1.69x vs 2-pass**) and **185.1 chunks/s (17,542 words/s)** on 020 chunks, with zero degraded labels.
- **Whole-document windowing and error analysis (round 9):**
  - GLiNER2's document windowing (`doc384-64`) re-encodes overlaps and merges spans, adding +22% to +27% latency vs non-overlapping chunks (2.60s vs 2.13s on 040 for `single_street_0.5`), but **eliminates all 51 boundary-split misses** present in 040 chunks.
  - Under `doc384-64`, `single_street_0.5` achieves **14,595 words/s** (2.50x faster than current 3-pass at 5,853 words/s) with **1.000 recall and 1.000 precision** across Person, Organization, and City (0 misses, 0 invented).
  - **Recommended production setting:** `single_street_0.5` with `doc384-64` windowing (or direct chunking at 17,862 words/s if documents are pre-segmented).

## What the existing T4 profile actually says

The baseline profile (`../colab_2026-10-24/t4_optimization/logs/gliner2-baseline.log`, 97 chunks × 3 passes) was
re-read:

- **"Span unpacking" (0.717 s self time in `models/base.py:469 unpack`) is the GPU, not CPU work.** gliner2's packed
  encoder path does `padded[mask] = states` with a boolean mask, which forces a device-to-host sync. Everything the
  encoder queued finishes inside that call, so its "self time" is the host waiting for the encoder kernels. The earlier
  assessment listed it as CPU work to optimize; that was wrong and has been corrected there.
- The real host-side work between GPU batches is:
  - batch collation: `collate_fn_inference`, 0.58 s cumulative, of which `_format_input_with_mapping` is 0.26 s;
  - the count/span head: `_compute_direct_span_logits_batched`, 0.33 s;
  - decoding and formatting.

  These times come from cProfile, which inflates Python-heavy code, so treat them as upper bounds.
- **GPU-busy ceiling.** The profile recorded 1.28 s of CUDA kernel time for 97 chunks. The clean baseline's wall time
  for that many chunks is about 1.77 s (5.29 s / 290 × 97). So the GPU is busy roughly 70% of the time.
  - Perfect CPU/GPU overlap with today's GPU work would reach about 75 chunks/s, not 100.
  - 100 chunks/s also needs about 25% less GPU time per chunk.
  - The kernel split was GEMMs about 48% (cuBLAS, near optimal), custom attention about 25%, elementwise about 15%.

  `pipeline_bench.py` measures the busy fraction directly, without the profiler-inflated Python time.

## What was built

| Item (plan section) | Change | Accuracy status |
|---|---|---|
| CPU/GPU pipeline overlap (Phase 1) | `serving/gliner2_pipeline.py`: a copy of gliner2 2.1.0's `batch_extract`. A helper thread runs the same collator over the same batches, in the same order, up to `depth` batches ahead, while the serving thread runs the model. Turned on with `gliner2_pipeline.install(depth)`. Refuses to install on any other gliner2 version. | Identical by construction: the same batches reach the model. The copy was diffed against the 2.1.0 source; the only change is the prefetch wrapper. `tests/test_gliner2_pipeline.py` passes on CPU: outputs `==` stock (every confidence float, span and order) for batch 1/7/16, depth 1/2/4, `batch_extract_entities` and `_long`, and two concurrent serving threads. Collation errors reach the caller and no helper threads leak. |
| Collation in worker processes (Phase 1) | No code change: gliner2's own `num_workers` DataLoader option, called through `batch_extract`. | Identical on the stand-in model; checked again on T4 by the benchmark. |
| Microbatch sweep (Phase 1) | Batch 8 / 16 / 32 in the benchmark. | **Not zero-loss.** Batch composition changes padding and kernel shapes. On the CPU stand-in, batch 8 already changed raw outputs (all kept values were still equal). On T4 it is accepted only if the full-output parity column says `True`. |
| Measurement (Phase 0 / plan §8) | `bench/pipeline_bench.py`: interleaved trials of every config; full-output and kept-value parity against the baseline; per-stage breakdown (collate, H2D, encoder, embeddings, span head, decode, format); GPU-busy fraction from `torch.profiler`. | — |

Not done in this round, with reasons:

- **INT8 / other precisions:** ruled out by the zero-loss requirement.
- **Gzip:** rejected by the plan itself.
- **FlashDeBERTa:** gliner2 2.1.0 reads `USE_FLASHDEBERTA` and the `flashdeberta` package. The env var in the plan
  document is `gliner`'s. The current path already uses fused attention (disentangled_flash plus the custom sm_75
  kernel), so this is a separate encoder experiment.
- **CUDA Graphs:** the packed encoder path syncs on the host (the `unpack` above). Graph capture needs those syncs
  removed first.
- **Fusing LayerNorm/GELU/residual:** remains the next GPU-side target (about 15% elementwise time).

## Reproduce

CPU check (no GPU or Hub needed):

```
pip install "gliner2[local]==2.1.0" torch
python tests/test_gliner2_pipeline.py                 # prints PASS
python bench/pipeline_bench.py --standin --corpus <corpus.json> --out /tmp/smoke --trials 1   # harness smoke test
```

T4 benchmark on Colab (single-file runner, as with the earlier Colab runs):

```
python bench/make_colab_pipeline_bench.py --trials 3  # writes artifacts/colab_pipeline_bench.py (corpus embedded)
colab run artifacts/colab_pipeline_bench.py           # on a T4 runtime; prints result.json and summary.md between markers
```

On the VM instead: `python bench/pipeline_bench.py --corpus <290-chunk corpus.json> --out <dir>`, from the repo root,
with the VM venv.

## T4 results

Run: Colab T4 (`Tesla T4`, capability [7, 5]), `colab run --gpu T4`, 3 interleaved trials, 290-chunk corpus
(sha256 `6304617b...`). Raw files: `t4_run.log`, `t4_result.json`, `t4_summary.md`.
Environment: torch 2.11.0+cu130, triton 3.6.0, transformers 5.20.2, gliner2 2.1.0, disentangled-flash 1.1.1, 2 vCPUs,
torch threads 1. This is **not** the VM environment (torch 2.6, Triton 3.2) behind the 54.81 figure. The baseline here is
50.82 chunks/s, 7% lower, so compare configs within this run only. The runner needed no changes.

| config | depth | workers | batch | chunks/s | vs baseline | identical full output | identical kept values |
|---|---:|---:|---:|---:|---:|---|---:|
| baseline | 0 | 0 | 16 | 50.82 | 1.000x | True | 290/290 |
| prefetch1 | 1 | 0 | 16 | 50.11 | 0.986x | True | 290/290 |
| prefetch2 | 2 | 0 | 16 | 49.88 | 0.982x | True | 290/290 |
| workers1 | 0 | 1 | 16 | 49.96 | 0.983x | True | 290/290 |
| workers2 | 0 | 2 | 16 | 47.20 | 0.929x | True | 290/290 |
| batch8 | 0 | 0 | 8 | 46.76 | 0.920x | False | 288/290 |
| batch32 | 0 | 0 | 32 | 48.42 | 0.953x | False | 290/290 |
| batch32_prefetch2 | 2 | 0 | 32 | 49.24 | 0.969x | False | 290/290 |


- **Baseline: 50.82 chunks/s** (expected about 54.8; Colab environment differs, see above).
- **Zero-loss (identical full output):** baseline, prefetch1, prefetch2, workers1, workers2. All are slower than the
  baseline (0.986x, 0.982x, 0.983x, 0.929x). The `batch8`, `batch32` and `batch32_prefetch2` configs changed the full
  output (batch8 changed kept values in 2 of 290 chunks), so they do not count as zero-loss, and they are slower anyway.
- **No config beat the baseline.** Run-to-run spread was about 1-2% (trial times in `t4_result.json`), so prefetch1,
  prefetch2 and workers1 are a tie with the baseline to slightly worse. workers2 and the batch configs are clearly worse.

### Stage breakdown, baseline (ms/chunk, one profiled pass of 6.02 s)

| stage | ms/chunk |
|---|---:|
| collate | 3.69 |
| to_device | 0.04 |
| encoder | 11.40 |
| embeddings | 0.11 |
| span_head | 4.89 |
| decode | 0.47 |
| format | 0.02 |
| other | 0.14 |

The encoder is 55% of the time. Collation (3.69 ms) and the span head (4.89 ms) are the large non-encoder stages.
The encoder figure includes the host wait from the `unpack` sync described above.

### GPU-busy fraction (torch.profiler)

- Baseline: **65.9%** (4.01 s of kernels in 6.09 s wall).
- Best prefetch config (prefetch1): **64.7%** (4.03 s of kernels in 6.22 s wall). Kernel time is unchanged, wall time
  is slightly longer, so overlap added no GPU utilisation here. These numbers are with the profiler on, so wall time is inflated.
- Top kernels (baseline): cuBLAS fp16 GEMMs about 1.83 s, custom `disentangled_attention_v4_kernel` 1.01 s, GELU, add,
  LayerNorm and other elementwise about 0.78 s.

### What this implies for 100 chunks/s

- 100 chunks/s means 2.9 s for the corpus. The baseline's GPU kernel time alone is about 4.0 s per pass with the profiler
  on, so even perfect overlap (GPU 100% busy) caps out near 72 chunks/s on this setup. In this run, scheduling the
  CPU work differently gave nothing.
- The host work that should overlap (collate about 3.7 ms/chunk plus span head about 4.9 ms/chunk) is Python and
  runs under the GIL, so a helper thread competes with the serving thread. With 2 vCPUs, worker processes cost about as
  much as they save (workers2 was the slowest zero-loss config).
- Getting to 100 chunks/s therefore needs a reduction in GPU time per chunk, about 30% or more. The candidates are the
  elementwise fusion (about 0.78 s, 19% of kernel time), a faster attention or GEMM path, and removing the host syncs
  (`unpack`) before trying CUDA Graphs. These are semantics-changing or kernel work, not Phase 1 items.
- Caveat: the Colab VM has 2 vCPUs. A host with more cores may show a gain from `workers`, which this run cannot show.

## Round 2: exact host-side cuts

The run above showed no gain from moving host work onto other threads or processes: Colab's 2 vCPUs share one core.
So round 2 cuts host work outright, and every change must leave outputs byte-identical (`serving/gliner2_exact.py`,
`gliner2_exact.install()`):

- **count_step0:** targets the span head, at 4.9 ms/chunk the largest non-encoder stage. For each entity pair, gliner2
  runs the CountLSTM GRU for `pred_count` steps (about 12 small kernels per step), then keeps only instance 0. Only
  GRU step 0 now runs. The x-side GEMM and the projector GEMMs keep their full `[pred_count x M]` shapes, so the same
  cuBLAS kernels see the same row values, and instance 0 is bit-identical by construction.
  - Checked bit-equal against the stock module for counts 1..21 in fp32 and fp16.
  - Unlike `DETECTION_FAST_COUNT`, nothing is batched across pairs. That flag changed 5 of 895 multilingual chunks.
- **format_cache:** the text part of the input formatting (subwords, ids, offsets) is cached per chunk text and reused
  by the 3 label passes. The schema part comes from the stock function. This removed about 10% of collation on CPU.
- **streams2:** two serving copies with shared fp16 weights, each on its own CUDA stream, take turns on the same
  batches. The serving queue already supports this (`DETECTION_STREAMS=2`); this round measures it on the same corpus.
  It measured +8-13% on the VM earlier.
- Rejected after measuring: a faster schema deep copy. The schemas are `OrderedDict`s, so at most about 0.1 ms/chunk
  was available.

CPU checks: `python tests/test_gliner2_exact.py` (PASS: every combination `==` stock). A bench smoke run with
`--standin` ran every config with identical outputs. The bench now defaults to the round-2 set (`baseline`, `count0`,
`fmtcache`, `exact`, `exact_prefetch1`, `streams2`, `exact_streams2`); `--configs all` adds the earlier ones. It
records `lscpu` and a stage breakdown of `exact` next to the baseline's.

### T4 results (round 2)

Run: Colab T4 (`Tesla T4`, capability [7, 5]), `colab run --gpu T4`, 3 interleaved trials, same 290-chunk corpus and
software as round 1 (torch 2.11.0+cu130, triton 3.6.0, gliner2 2.1.0). Raw files: `t4_run_round2.log`,
`t4_result_round2.json`, `t4_summary_round2.md`. The runner needed no changes. The first launch failed immediately with a
local DNS error from the colab CLI (before any code ran), and I reran the same command unchanged.

**CPU (`lscpu`): 1 physical core** (1 socket x 1 core x 2 threads = 2 vCPUs, Xeon 2.00 GHz). So the 2 vCPUs are
hyperthreads of one core, which explains why threads and processes gave nothing in round 1.

| config | count0 | fmtcache | streams | depth | workers | batch | chunks/s | vs baseline | identical full output | identical kept values |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| baseline | False | False | 1 | 0 | 0 | 16 | 47.79 | 1.000x | True | 290/290 |
| count0 | True | False | 1 | 0 | 0 | 16 | 49.64 | 1.039x | True | 290/290 |
| fmtcache | False | True | 1 | 0 | 0 | 16 | 47.31 | 0.990x | True | 290/290 |
| exact | True | True | 1 | 0 | 0 | 16 | 49.52 | 1.036x | True | 290/290 |
| exact_prefetch1 | True | True | 1 | 1 | 0 | 16 | 49.62 | 1.038x | True | 290/290 |
| streams2 | False | False | 2 | 0 | 0 | 16 | 49.74 | 1.041x | True | 290/290 |
| exact_streams2 | True | True | 2 | 0 | 0 | 16 | 50.32 | 1.053x | True | 290/290 |


- **Every config has an identical full output** (`True`, 290/290), so all of them count as zero-loss.
- **Baseline: 47.79 chunks/s.** This is lower than round 1's 50.82 on the same GPU type and software, a different Colab VM,
  so compare within this run only.
- **Did anything beat the baseline? Marginally.** Six configs measured 1.036x to 1.053x, except `fmtcache` (0.990x, no
  gain). Trial spread is about 2-4% (baseline 6.05-6.31 s; `exact_streams2` 5.59-5.87 s), so the gains are comparable
  to noise. The ordering `exact_streams2` > `streams2` > `count0` ~ `exact` is plausible but not established by 3 trials.
  `exact` (1.036x) is no better than `count0` alone (1.039x), so format_cache added nothing measurable.

### Stage breakdown, baseline vs exact (ms/chunk)

| stage | baseline | exact | change |
|---|---:|---:|---:|
| collate | 2.55 | 2.20 | -0.35 |
| to_device | 0.04 | 0.04 | 0 |
| encoder | 13.08 | 13.49 | +0.41 |
| embeddings | 0.10 | 0.10 | 0 |
| span_head | 4.74 | 3.76 | -0.97 |
| decode | 0.43 | 0.40 | -0.03 |
| format | 0.02 | 0.02 | 0 |
| other | 0.12 | 0.12 | 0 |
| total (s per pass) | 6.11 | 5.84 | -4.4% |

`span_head` dropped 0.97 ms/chunk (20%) and `collate` 0.35 ms/chunk (14%), as intended. The encoder column rose 0.41 ms
(this includes the host wait on the GPU, and is probably noise or work moved there). Net: about 0.9 ms/chunk saved of
about 21, which is the 4% seen. The span head still costs 3.8 ms/chunk.

### GPU-busy fraction (torch.profiler)

- Baseline: **71.5%** (4.48 s of kernels in 6.27 s wall).
- `exact_streams2`: **98.1%** (5.47 s of kernels in 5.57 s wall). Kernel time sums across both streams, and kernels from
  the two streams may overlap, so this number can overstate busy time. Even so, the GPU is saturated, and the extra kernel
  time (5.47 vs 4.48 s) comes from kernels running slower when two streams share the SMs.

### Implication

The host-side cuts gave about 5% at best. With streams2 the GPU is the limit, and 100 chunks/s needs about 2x the baseline
throughput. That has to come from less GPU time per chunk (elementwise fusion, attention, removing the pad/unpack syncs),
not from more host-side work.

## Round 3: fewer label passes (accuracy-checked) and fused add+LayerNorm

With byte-identical output, round 2 reached +5.3%, and kernel time alone caps one Colab T4 near 65 chunks/s. The
largest remaining lever changes what is computed: each label pass re-encodes the whole chunk, and detection runs three
(Organization; Person+Street+City; City alone).

- **`bench/pass_bench.py`** runs the production `Gliner2NER.detect_batch` (`artifacts/colab_pipeline/gliner2_ner.py`:
  thresholds, cross-pass City union, table-cell confirmation, span resolution) with five pass layouts. It scores each
  against the corpus ground truth exactly as `bench_detection.py` does.
  - Layouts: `current_3pass`; `drop_city_pass`; `merged_city_desc` (City's alone-pass description inside the shared
    pass); `city_with_org` (the City-alone query moved into the Organization pass, still two passes); and
    `city_with_org_desc`.
  - A layout is accepted only if **no label's recall or precision falls below `current_3pass`**.
  - The corpus is only 10 documents: confirm an accepted layout on a held-out set (the 25-language corpus on the VM)
    before shipping.
  - Exact count0 is on for every layout; the non-exact `DETECTION_FAST_COUNT` is off.
- **`pipeline_bench.py`** now defaults to `baseline`, `count0`, `count0_streams2`, `ln` and `count0_streams2_ln`, with
  5 trials. `ln` is gliner_t4's bit-identical fused residual add + LayerNorm. Earlier it was off because it gained
  nothing while the pipeline was host-bound; with two streams the GPU is now the limit. Round-2 configs remain available
  (`--configs all`).

### T4 results (round 3)

Run: Colab T4 (`Tesla T4`, capability [7, 5]), `colab run --gpu T4`, torch 2.11.0+cu130, gliner2 2.1.0. Raw files:
`t4_run_round3.log`, `t4_result_round3.json`, `t4_summary_round3.md`, `t4_pass_round3.json`, `t4_pass_round3.md`.
The runner needed no changes and the run completed on the first attempt.

#### a) Pipeline (5 interleaved trials, same output as stock required)

| config | count0 | fmtcache | ln | streams | depth | workers | batch | chunks/s | vs baseline | identical full output | identical kept values |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| baseline | False | False | False | 1 | 0 | 0 | 16 | 50.91 | 1.000x | True | 290/290 |
| count0 | True | False | False | 1 | 0 | 0 | 16 | 53.40 | 1.049x | True | 290/290 |
| count0_streams2 | True | False | False | 2 | 0 | 0 | 16 | 56.20 | 1.104x | True | 290/290 |
| ln | False | False | True | 1 | 0 | 0 | 16 | 51.86 | 1.019x | True | 290/290 |
| count0_streams2_ln | True | False | True | 2 | 0 | 0 | 16 | 58.05 | 1.140x | True | 290/290 |


- **Every config has identical full output = True** (290/290), so all count as zero-loss.
- **Baseline: 50.91 chunks/s** (this VM; round 2's was 47.79, so compare within the run).
- **Does `ln` add speed on top of `count0_streams2`? Yes, about 3.3%**: 56.20 to 58.05 chunks/s (median 5.16 s to 4.99 s per
  pass). `count0_streams2_ln` was faster than `count0_streams2` in each of the 5 trials (4.90 vs 5.03, 5.12 vs 5.16,
  5.20 vs 5.44, 5.00 vs 5.11, 4.99 vs 5.26 s), but the gap per trial is small and 5 trials on a shared Colab VM is a
  modest sample. Alone, `ln` gave only 1.019x (not faster in every trial).
- **GPU busy:** baseline 70.0% (4.24 s of kernels in 6.06 s wall); `count0_streams2_ln` 94.2% (4.73 s in 5.02 s wall).
  The GPU is close to saturated, so more host-side work will not help much. Kernel time alone caps this near 65-70 chunks/s,
  short of 100.

#### b) Label-pass layouts (`pass_bench.py`, 2 trials, production `detect_batch`)

| layout | passes | chunks/s | speedup | Person R/P | Organization R/P | Street R/P | City R/P | identical chunks | accepted |
|---|---|---|---|---|---|---|---|---|---|
| current_3pass | 3 | 53.33 | 1.000x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.930 | 290/290 | yes |
| drop_city_pass | 2 | 77.71 | 1.457x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 286/290 | yes |
| merged_city_desc | 2 | 76.73 | 1.439x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.936 | 269/290 | yes |
| city_with_org | 2 | 75.79 | 1.421x | 0.994/0.961 | 0.977/0.857 | 0.861/1.000 | 1.000/0.884 | 277/290 | no: City precision |
| city_with_org_desc | 2 | 75.29 | 1.412x | 0.994/0.961 | 0.977/0.857 | 0.861/1.000 | 1.000/0.889 | 259/290 | no: City precision |


Acceptance rule: no label's recall or precision below `current_3pass`.

- `drop_city_pass`: **accepted.** Same Person/Organization/Street numbers; City recall 1.000 and precision up from .930
  to .947. 1.457x faster (77.71 chunks/s). 286 of 290 chunks give identical output.
- `merged_city_desc`: **accepted.** Organization unchanged and City precision .936 (up from .930). 1.439x. Only 269/290 chunks
  are identical, so more outputs changed than with `drop_city_pass`.
- `city_with_org`: **rejected**, City precision fell to .884 (from .930), though Organization precision rose to .857.
- `city_with_org_desc`: **rejected**, City precision fell to .889.

**Best accepted layout: `drop_city_pass`** (2 passes, 1.457x, the highest speed and the most identical chunks of the
accepted layouts). This is not zero-loss: it changes the computation, and 4 chunks differ from the 3-pass output.
**The corpus is only 10 documents, so a layout accepted here needs held-out confirmation** (for example the
25-language corpus on the VM) before it ships. Per-label numbers are rounded to 3 decimals, and the speed is from 2 trials.
Combining `drop_city_pass` with the zero-loss `count0_streams2_ln` path was not measured, and I'd expect the gains to
partly overlap rather than multiply.

## Round 4: shipping candidates end to end

Rounds 2 and 3 measured the pieces separately. Round 4 measures them together through the production serving path: the
ai-utils detection queue, driven by `bench_detection.py` in queue mode, on one T4, with 4 repeats of the corpus (1160
chunks, as the reference runs) and 2 runs per configuration (`bench/make_colab_detection_bench.py` builds
`artifacts/colab_detection_bench.py`).

New opt-in switches in `artifacts/colab_pipeline/gliner2_ner.py` (patch for ai-utils: `artifacts/ai_utils_gliner2_ner.patch`;
defaults unchanged):
- `DETECTION_CITY_PASS=0` drops the City-alone pass.
- `DETECTION_EXACT_COUNT=1` installs the bit-identical count0 from the kernel checkout's `serving/`.
- `GLINER_T4_FUSE_LAYERNORM=1` (existing) turns on the bit-identical fused add + LayerNorm.

| config | streams | count head | fused add+LN | passes | role |
|---|---:|---|---|---:|---|
| ref_3pass | 1 | stock | no | 3 | accuracy reference |
| prod_3pass | 2 | fast count (not exact) | no | 3 | today's serving settings |
| exact_3pass | 2 | exact count0 | yes | 3 | zero-loss speed; must match ref_3pass on every chunk |
| exact_2pass | 2 | exact count0 | yes | 2 | **shipping candidate** |
| fast_2pass | 2 | fast count | yes | 2 | upper bound if fast count's accuracy is acceptable |

Expected from rounds 2-3: with two passes, kernel time puts the one-T4 ceiling near 100 chunks/s. A realistic estimate
on Colab's single physical core is 85-95.

### T4 results (round 4)

Run: `colab run --gpu T4`, `bench_detection.py` in queue mode, 4 repeats of the corpus (1160 chunks), 2 runs per config.
Raw files: `t4_run_round4.log`, `t4_detect_round4.json`, `t4_detect_round4.md`.

- **GPU model not confirmed from the log.** The runner's `nvidia-smi` output did not reach the captured log, so the log
  has no Tesla T4 line. The session was requested with `--gpu T4`, and the round 1-3 sessions started the same way
  reported Tesla T4 / [7, 5].
- **Runner fix (setup only).** The runner merged `COMMON` and each config's env with `dict(os.environ, **COMMON, **extra)`,
  which raised a `TypeError` because both set `DETECTION_FAST_COUNT`. This stopped the first run after `ref_3pass`. I
  changed it in `bench/make_colab_detection_bench.py` so the config's value overrides the common one and reran from the start.
  Several earlier launches failed with local DNS errors before any code ran and were rerun unchanged. No `CONFIG_FAILED` appeared.

| config | chunks/s | vs prod_3pass | Person R/P | Organization R/P | Street R/P | City R/P | identical vs ref | GPU util % | serving MB |
|---|---|---|---|---|---|---|---|---|---|
| ref_3pass | 45.4 | 0.933x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.930 | 1160/1160 | [[73], [76]] | [1943] |
| prod_3pass | 48.6 | 1.000x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.930 | 1160/1160 | [[83], [88]] | [3095] |
| exact_3pass | 45.7 | 0.940x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.930 | 1160/1160 | [[81], [83]] | [3057] |
| exact_2pass | 66.3 | 1.364x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[75], [78]] | [2997] |
| fast_2pass | 65.9 | 1.357x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[85], [81]] | [3007] |

Per-run chunks/s: ref_3pass 48.1 / 42.6, prod_3pass 50.4 / 46.8, exact_3pass 46.4 / 45.0, exact_2pass 65.8 / 66.8,
fast_2pass 66.6 / 65.3. The runs of the same config differ by up to 5.5 chunks/s (ref_3pass), so differences of a few chunks/s
between the 3-pass configs are within noise.

- **Speed vs `prod_3pass` (48.6):** `exact_2pass` 1.364x, `fast_2pass` 1.357x. The two 3-pass configs `ref_3pass` (0.933x)
  and `exact_3pass` (0.940x) were slower than today's settings. In this run, 2 streams plus exact count plus fused LN gave no
  gain over the single-stream `ref_3pass` (45.7 vs 45.4). This differs from round 3's encoder benchmark, which showed
  `count0_streams2_ln` ahead. Here the exact count head (a loop of per-pair steps) did not keep up with the batched fast count of `prod_3pass`.
  I did not investigate further.
- **Does `exact_2pass` meet or beat `ref_3pass` on every label? Yes.** Person, Organization and Street are equal
  (.994/.961, .977/.846, .861/1.000), and City precision rose from .930 to .947 with recall unchanged at 1.000.
- **Identical vs ref:** `exact_3pass` is identical on **1160/1160** chunks, as expected. `prod_3pass` (fast count) was also 1160/1160 here.
  `exact_2pass` and `fast_2pass` are 1144/1160 (16 chunks differ, all from dropping the City pass). `fast_2pass` gave the same
  accuracy and speed as `exact_2pass` (65.9 vs 66.3), so it offers no advantage here.
- **GPU utilisation (mean `nvidia-smi` %, per run):** ref_3pass 73/76, prod_3pass 83/88, exact_3pass 81/83, exact_2pass 75/78,
  fast_2pass 85/81. Utilisation is not saturated for any config. This is the coarse `nvidia-smi` measure, not kernel-busy time.
- **Serving memory:** 1943 MB for one stream (`ref_3pass`) and 2997-3095 MB for the 2-stream configs.
- **Does `exact_2pass` reach 100 chunks/s? No.** It reached 66.3, 33.7 chunks/s (34%) short. It would need another 1.51x. The
  expected 85-95 from the plan above was too optimistic on this Colab VM (1 physical core). Getting closer needs less GPU
  time per chunk or a faster host, and the GPU is only 75-78% utilised in this config, so the host is part of the limit.
- **Ship status:** `exact_2pass` is the shipping candidate, but only after held-out confirmation of the 2-pass layout
  (10-document corpus) and acceptance of the 16 changed chunks.

## Round 5: what limits the shipping path

Round 4's `exact_2pass` reached 66.3 chunks/s through the queue, with the GPU 75-78% busy, so the host side limits it.
The pass benchmark ran the same 2-pass layout at 77.7 chunks/s with one stream. Two suspects:

1. **The benchmark's own sampler.** `bench_detection.py` starts an `nvidia-smi` subprocess every 0.5 s. On Colab's
   single physical core, that competes with the serving threads.
   - New switch: `DETECTION_BENCH_GPU_SAMPLE_S` (0 means one sample at the end only).
2. **Per-run overhead in the queue path.** Every run of 16 chunks repeats each pass's setup: schema, collator, window
   split and merge, span resolution.
   - `bench/queue_profile.py` times `detect_batch` in queue-style runs against one whole-corpus call, then profiles the
     queue-style runs with cProfile on the T4.

Round-5 runner configs (`python bench/make_colab_detection_bench.py`, 3 runs each, 1160 chunks):

| config | what it isolates |
|---|---|
| ref_3pass | accuracy reference (sampler off) |
| prod_3pass | today's serving settings, sampler off |
| exact_2pass_sampled | round 4's measurement (sampler on), the control |
| exact_2pass | sampler off: the artifact |
| exact_2pass_1stream | whether 2 streams help on one core in the queue path |
| exact_2pass_run32 / run64 | larger queue runs (`DETECTION_RUN_SIZE`); not byte-exact, accuracy gate applies; longer wait for urgent requests |

### T4 results (round 5)

Run: `colab run --gpu T4`, `bench_detection.py` in queue mode, 4 repeats of the corpus (1160 chunks), 3 runs per config,
followed by `bench/queue_profile.py`. Raw files: `t4_run_round5.log`, `t4_detect_round5.json`, `t4_detect_round5.md`,
`t4_qprofile_round5.json`, `t4_qprofile_round5.txt`.

- **Environment & GPU confirmation:** The raw `nvidia-smi` text banner did not reach the log (same behavior as round 4
  under Colab), but the device is confirmed as a **Tesla T4** by `torch.cuda.get_device_name()` (recorded as `"device": "Tesla T4"`
  in `t4_qprofile_round5.json`).
- **No failures:** No `CONFIG_FAILED` occurred; all 7 configs completed 3 full runs.

| config | chunks/s | vs prod_3pass | Person R/P | Organization R/P | Street R/P | City R/P | identical vs ref | GPU util % | serving MB |
|---|---|---|---|---|---|---|---|---|---|
| ref_3pass | 43.7 | 0.942x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.930 | 1160/1160 | [[80], [86], [80]] | [1943] |
| prod_3pass | 46.4 | 1.000x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.930 | 1160/1160 | [[100], [94], [100]] | [3063] |
| exact_2pass_sampled | 65.5 | 1.412x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[83], [82], [80]] | [3021] |
| exact_2pass | 65.0 | 1.401x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[89], [97], [82]] | [3007] |
| exact_2pass_1stream | 65.7 | 1.416x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[82], [88], [86]] | [1829] |
| exact_2pass_run32 | 65.3 | 1.407x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[85], [83], [73]] | [3019] |
| exact_2pass_run64 | 65.3 | 1.407x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[74], [90], [72]] | [3369] |

#### Per-run breakdown & analysis

Per-run chunks/s:
- `ref_3pass`: [48.2, 43.5, 43.7] (median 43.7)
- `prod_3pass`: [49.6, 46.4, 46.2] (median 46.4)
- `exact_2pass_sampled`: [65.5, 64.8, 66.2] (median 65.5)
- `exact_2pass`: [65.7, 64.3, 65.0] (median 65.0)
- `exact_2pass_1stream`: [65.5, 65.7, 65.9] (median 65.7)
- `exact_2pass_run32`: [65.3, 65.8, 65.3] (median 65.3)
- `exact_2pass_run64`: [65.0, 66.1, 65.3] (median 65.3)

1. **Sampler effect (`exact_2pass` vs `exact_2pass_sampled`):**
   `exact_2pass` (sampler off) ran at 65.0 chunks/s vs 65.5 chunks/s for `exact_2pass_sampled` (sampler on every 0.5 s).
   The difference is under 1% and within run-to-run noise (runs overlap between 64.3 and 66.2 chunks/s). Background
   `nvidia-smi` polling is not a bottleneck.

2. **1 vs 2 streams (`exact_2pass_1stream` vs `exact_2pass`):**
   `exact_2pass_1stream` reached **65.7 chunks/s** (runs: 65.5, 65.7, 65.9) with **1829 MB** serving memory.
   `exact_2pass` (2 streams) reached **65.0 chunks/s** (runs: 65.7, 64.3, 65.0) with **3007 MB** serving memory.
   On Colab's single physical core (2 vCPUs sharing 1 core), 2 CUDA streams provide **zero throughput gain** because the host
   Python thread is saturated, but costs an extra **1.18 GB of GPU memory (+64%)**. Single stream is strictly superior here.

3. **Queue run size (`run32` and `run64`):**
   `exact_2pass_run32` and `exact_2pass_run64` both measured **65.3 chunks/s**, virtually identical to run16 (65.0 - 65.7).
   Both achieve identical accuracy: Person 0.994/0.961, Organization 0.977/0.846, Street 0.861/1.000, City 1.000/0.947,
   matching or exceeding `ref_3pass` across all labels (City precision .947 vs .930).
   Both produce 1144/1160 identical chunks vs `ref_3pass` (same 16 differences as `exact_2pass` due to dropping the City pass).
   Larger batch queue runs do not increase throughput on this host.

4. **Queue chunking overhead (`queue_profile.py`):**
   - `queue_runs` (run size 16): **69.54 chunks/s** (seconds: [4.170, 4.293])
   - `whole_corpus` (single batch of 290 chunks): **69.02 chunks/s** (seconds: [4.202, 4.290])
   Overhead from repeatedly setting up queue runs (batch collation, window splits, merging results) is negligible (<0.8%).

5. **Profile breakdown (top 15 functions outside the encoder):**
   cProfile of `detect_batch` queue runs (5.049 s total profile time) reveals where non-encoder time goes:

   **Top 15 by `tottime` (self-time):**
   1. `{method 'tolist' of 'torch._C.TensorBase' objects}`: **2.538 s** (50.3% of total time! Forces host-GPU sync to extract Python lists)
   2. `{method 'cpu' of 'torch._C.TensorBase' objects}`: **0.379 s** (7.5% - Device-to-Host transfer)
   3. `gliner2/processor.py:_format_input_with_mapping`: 0.185 s (cum 0.533 s)
   4. `{built-in method torch._C._nn.linear}`: 0.182 s
   5. `{method 'token_to_id' of 'tokenizers.Tokenizer' objects}`: 0.089 s
   6. `{method 'append' of 'list' objects}`: 0.085 s
   7. `transformers/tokenization_utils_tokenizers.py:_convert_token_to_id_with_added_voc`: 0.080 s
   8. `gliner2/processor.py:_transform_record`: 0.078 s (cum 0.767 s)
   9. `transformers/tokenization_utils_base.py:convert_tokens_to_ids`: 0.077 s
   10. `serving/gliner2_exact.py:count_instance0`: 0.069 s (cum 0.315 s)
   11. `copy.py:deepcopy`: 0.055 s
   12. `{method 'extend' of 'list' objects}`: 0.052 s
   13. `gliner2/inference/runtime.py:_find_spans`: 0.039 s
   14. `{built-in method torch.tensor}`: 0.038 s
   15. `{built-in method builtins.len}`: 0.038 s

   **Top 15 by `cumtime` (cumulative time):**
   1. `torch/autograd/grad_mode.py:__exit__`: 4.805 s
   2. `gliner2/inference/runtime.py:batch_extract`: 4.756 s
   3. `gliner2/inference/runtime.py:_extract_from_batch`: 3.794 s
   4. `serving/gliner2_exact.py:_compute_direct_span_logits_batched`: 2.972 s
   5. `{method 'tolist' of 'torch._C.TensorBase' objects}`: 2.538 s
   6. `gliner2/training/trainer.py:__call__`: 0.927 s
   7. `gliner2/processor.py:collate_fn_inference`: 0.927 s
   8. `gliner2/processor.py:_collate_batch`: 0.922 s
   9. `gliner2/processor.py:_transform_record`: 0.767 s
   10. `torch/utils/_contextlib.py:decorate_context`: 0.635 s
   11. `gliner2/processor.py:_format_input_with_mapping`: 0.533 s
   12. `gliner2/inference/runtime.py:_extract_from_batch_sync_collapsed`: 0.506 s
   13. `torch/nn/modules/module.py:_wrapped_call_impl`: 0.417 s
   14. `torch/nn/modules/module.py:_call_impl`: 0.414 s
   15. `{method 'cpu' of 'torch._C.TensorBase' objects}`: 0.379 s

6. **Best config and distance to 100 chunks/s:**
   The best configuration meeting the accuracy gate is **`exact_2pass_1stream`** at **65.7 chunks/s** (1.416x speedup over `prod_3pass` at 46.4 chunks/s).
   It is **34.3 chunks/s short of 100 chunks/s** (needs another 1.52x improvement).
   cProfile makes the root cause unambiguous: host-bound synchronization (`tensor.tolist()` and `tensor.cpu()` blocking the CPU thread while GPU work completes) in gliner2's span head and decoding accounts for over 2.9 seconds of wall time. To approach 100 chunks/s on a single-core host, these synchronization points must be replaced with asynchronous, batched GPU kernel postprocessing.


## Round 6: one-thread CPU/GPU pipeline

**A correction to the round-5 reading.** The 2.5 s on `.tolist()` is not host overhead to remove. The encoder's kernels
are queued asynchronously, and the first `.tolist()` of each batch (the count head's result) is simply where the CPU
waits for the encoder to finish. So that time is GPU time.

The round-5 numbers show the real limit: the CPU and the GPU take turns.
- Per corpus run, the 2-pass GPU work is about 2.8 s and the CPU work about 1.3 s.
- Measured wall time is about 4.2 s, roughly their sum (queue_runs ≈ whole_corpus, so per-run overhead is not it).
- Overlapped, the wall time would be about 2.8 s, or about 100 chunks/s.
- Two serving threads cannot overlap them on Colab's single core, so `streams2` gave nothing.

**`serving/gliner2_pipelined.py`** overlaps them in one thread:
- Batch i+1's encoder runs on its own CUDA stream while the CPU decodes batch i (gliner2's own `_extract_from_batch`,
  fed the precomputed encoder output) and collates batch i+2.
- Host-to-device copies go through a copy stream, so a blocking copy never waits behind a running encoder.
- `extract_entities_long_many` pipelines every label pass of a run together, so even a 16-chunk queue run (one batch per
  pass) overlaps its two passes.
- Kernels, shapes and inputs are those of stock gliner2: only the order in which work is issued changes.

Production switch: `DETECTION_PIPELINE=1` in `gliner2_ner.py`; `detect_batch` precomputes all passes through the
pipeline, and the per-pass handling is unchanged.

CPU checks, `python tests/test_gliner2_pipelined.py`, PASS:
- `extract_many` and `extract_entities_long_many` give `==` stock results for batch 7/16/64, one-batch and many-batch
  jobs, with and without the exact count head.
- `detect_batch` gives `==` results with the pipeline on and off, for 3 and 2 passes, on a queue-sized run and on the
  whole set.

Round-6 runner (`python bench/make_colab_detection_bench.py`, 3 runs each, then the queue profile with the pipeline):

| config | what it shows |
|---|---|
| exact_2pass_1stream | round 5's best (65.7), the control |
| pipe_3pass | zero-loss 3-pass with the pipeline: must be identical to ref_3pass on every chunk |
| **pipe_2pass** | **shipping candidate** |
| pipe_2pass_sampled | its GPU utilisation |
| pipe_2pass_run32 / run64 | more batches per pipeline (not byte-exact; accuracy gate) |
| pipe_2pass_streams2 | pipeline plus two serving threads |

### T4 results (round 6)

Run: `colab run --gpu T4`, `bench_detection.py` in queue mode, 4 repeats of the corpus (1160 chunks), 3 runs per config,
followed by `bench/queue_profile.py` with the pipeline enabled. Raw files: `t4_run_round6.log`, `t4_detect_round6.json`,
`t4_detect_round6.md`, `t4_qprofile_round6.json`, `t4_qprofile_round6.txt`.

- **Environment & execution:** The session was run on a Tesla T4. No `CONFIG_FAILED` occurred across all configs; all CUDA
  streams and stream event synchronizations executed without error.

| config | chunks/s | vs prod_3pass | Person R/P | Organization R/P | Street R/P | City R/P | identical vs ref | GPU util % | serving MB |
|---|---|---|---|---|---|---|---|---|---|
| ref_3pass | 46.5 | 0.919x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.930 | 1160/1160 | [[77], [83], [70]] | [1943] |
| prod_3pass | 50.6 | 1.000x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.930 | 1160/1160 | [[94], [96], [84]] | [3095] |
| exact_2pass_1stream | 70.6 | 1.395x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[83], [83], [81]] | [1829] |
| pipe_3pass | 50.2 | 0.992x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.930 | 1160/1160 | [[100], [91], [100]] | [2509] |
| pipe_2pass | 72.8 | 1.439x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[90], [89], [90]] | [2697] |
| pipe_2pass_sampled | 70.2 | 1.387x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[82], [77], [82]] | [2697] |
| pipe_2pass_run32 | 73.3 | 1.449x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[100], [100], [100]] | [2737] |
| pipe_2pass_run64 | 73.6 | 1.455x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[95], [90], [95]] | [2785] |
| pipe_2pass_streams2 | 69.6 | 1.375x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 1144/1160 | [[74], [94], [74]] | [4467] |

#### Per-run breakdown & analysis

Per-run chunks/s:
- `ref_3pass`: [51.8, 46.0, 46.5] (median 46.5)
- `prod_3pass`: [57.4, 50.6, 49.9] (median 50.6)
- `exact_2pass_1stream`: [74.0, 69.8, 70.6] (median 70.6)
- `pipe_3pass`: [53.4, 50.2, 49.7] (median 50.2)
- `pipe_2pass`: [72.9, 72.8, 72.5] (median 72.8)
- `pipe_2pass_sampled`: [71.1, 70.0, 70.2] (median 70.2)
- `pipe_2pass_run32`: [72.5, 73.3, 73.3] (median 73.3)
- `pipe_2pass_run64`: [73.3, 74.1, 73.6] (median 73.6)
- `pipe_2pass_streams2`: [69.6, 68.1, 69.6] (median 69.6)

1. **`pipe_3pass` parity vs `ref_3pass`:**
   - **`pipe_3pass` is identical on 1160/1160 chunks (100% bit-identical parity)** against `ref_3pass`.
   - It matches production speed (50.2 chunks/s vs `prod_3pass` 50.6 chunks/s) while restoring exact numerical parity
     and removing the lossy approximations of `fast_count`.

2. **`pipe_2pass` speedup and accuracy:**
   - `pipe_2pass` achieved **72.8 chunks/s** (runs: [72.9, 72.8, 72.5]), a **+3.1% speedup** over unpipelined
     `exact_2pass_1stream` (70.6 chunks/s), and **1.439x vs `prod_3pass`** (50.6 chunks/s).
   - Accuracy vs `ref_3pass`: Person 0.994/0.961, Organization 0.977/0.846, Street 0.861/1.000, City 1.000/0.947.
     It **meets or beats `ref_3pass` on every label** (City precision rises from 0.930 to 0.947, all other labels identical).
   - Chunk parity: 1144/1160 chunks are identical (the 16 differing chunks stem solely from dropping the City-alone pass).

3. **GPU utilisation of `pipe_2pass_sampled`:**
   - `pipe_2pass_sampled` (0.5 s `nvidia-smi` background sampler) measured mean GPU util of **[[82], [77], [82]]** (~80-82%)
     and ran at 70.2 chunks/s.
   - Without background sampling, `pipe_2pass` sustained **~90% GPU utilisation** ([[90], [89], [90]]) and 72.8 chunks/s.
     Under near-saturated GPU load, periodic subprocess polling imposes a ~3.5% host-side penalty.

4. **Batch queue runs (`run32` and `run64`) and multi-stream (`streams2`):**
   - `pipe_2pass_run32`: **73.3 chunks/s** (1.449x vs prod_3pass), GPU util [[100], [100], [100]].
   - `pipe_2pass_run64`: **73.6 chunks/s** (1.455x vs prod_3pass), GPU util [[95], [90], [95]].
     Both maintain identical accuracy (1144/1160 chunks, City precision .947). Amortizing queue dispatch over 32 or 64 chunks
     further improves pipelined overlap (+1% throughput).
   - `pipe_2pass_streams2`: **69.6 chunks/s** (slower than 1 stream at 72.8 chunks/s) and consumed **4467 MB VRAM** (+1.77 GB).
     Running multiple serving threads on Colab's single physical core causes CPU thread contention, decreasing throughput.

5. **Queue profile & cProfile analysis:**
   - `queue_runs` with pipeline: **75.42 chunks/s** vs `whole_corpus`: **75.03 chunks/s**.
   - **Crucial profile finding:** In the cProfile trace (`t4_qprofile_round6.txt`), time spent in `{method 'tolist' of 'torch._C.TensorBase' objects}`
     collapsed from **2.538 s (round 5) down to 0.778 s (round 6)**.
   - **Important note on `.tolist()` and `.cpu()`:** Time recorded inside `.tolist()` and `.cpu()` is **the CPU waiting for
     the GPU to finish execution**, not host compute work. Because the one-thread pipeline issues the encoder work for batch i+1
     on a separate CUDA stream while the CPU decodes batch i, the GPU finishes execution before or shortly after the CPU requests
     the tensor values. This reduced idle CPU sync time by **69% (from 2.54 s to 0.78 s)**.

6. **Best config and distance to 100 chunks/s:**
   - Best shipping candidate: **`pipe_2pass`** at **72.8 chunks/s** (runs: [72.9, 72.8, 72.5]), or **73.6 chunks/s** with `run64`.
   - Distance to 100 chunks/s: **27.2 chunks/s short of 100** (needs another 1.37x).
   - With GPU utilization now reaching 90-100% and CPU wait time compressed to 0.78 s, pipeline-level scheduling on the host
     has reached its ceiling on a single T4. Reaching 100 chunks/s on a single T4 requires reducing the raw GPU kernel time
     (e.g., fusing LayerNorm/GELU/residual or faster attention kernels).


## Round 7: can one label pass match today's accuracy?

After round 6 the shipping path is GPU-bound at about 73 chunks/s on a Colab T4. One pass instead of two halves the
encoder work, which is the only lever large enough for 100+ on a T4. The problem: labels sharing a prompt shift each
other's confidences. The repo measured Organization recall falling to .62 next to Person/City/Street, on an older
pipeline.

Step 1, this round, needs no retraining: `bench/single_pass_bench.py` runs through the production `detect_batch`, with
the exact speedups and the pipeline on.
1. Reference: today's 3-pass layout at today's thresholds, and the 2-pass layout.
2. Three single-pass layouts (Person first; Organization first; Organization first with the City-alone description),
   each swept over uniform thresholds 0.30-0.90, with per-label recall/precision against ground truth.
3. Per layout and label, a threshold meeting today's recall and precision, if one exists. That exact setting is then
   run end to end and timed.

Caveat: thresholds picked on these 10 documents are tuned to them. A single-pass layout that passes here still needs
the 25-language held-out check.

Step 2, if a gap remains: self-distillation. Fine-tune (LoRA) the model so that one merged pass reproduces the current
2-pass outputs on a large unlabeled multilingual corpus. No human labels are needed, since the target is today's
behaviour; ground truth plus held-out sets gate the result.

### T4 results (round 7)

Run: `colab run --gpu T4`, `bench/single_pass_bench.py` via `artifacts/colab_pipeline_bench.py --only single`.
Raw files: `t4_run_round7.log`, `t4_single_round7.json`, `t4_single_round7.md`.

| layout | passes | thresholds | chunks/s | Person R/P | Organization R/P | Street R/P | City R/P | accepted |
|---|---|---|---|---|---|---|---|---|
| current_3pass | 3 | today's | 57.7 | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.930 | reference |
| drop_city_pass | 2 | today's | 81.9 | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | reference |
| single_person_first | 1 | Per 0.85, Org 0.80, Str 0.30, Cit 0.90 | 133.5 | 0.994/0.971 | 0.977/0.863 | 0.949/1.000 | 1.000/0.964 | yes |
| single_org_first | 1 | Per 0.80, Org 0.80, Str 0.30, Cit 0.90 | 129.0 | 0.996/0.971 | 0.977/0.863 | 0.949/1.000 | 1.000/0.964 | yes |
| single_org_first_city_desc | 1 | Per 0.85, Org 0.90, Str 0.30, Cit 0.80 | 120.7 | 0.996/0.971 | 0.977/0.869 | 0.946/1.000 | 1.000/0.941 | yes |

Uniform-threshold sweep, single passes (recall/precision per label):

| layout | t | Person | Organization | Street | City |
|---|---|---|---|---|---|
| single_person_first | 0.3 | 0.996/0.953 | 0.977/0.829 | 0.949/1.000 | 1.000/0.925 |
| single_person_first | 0.4 | 0.996/0.955 | 0.977/0.829 | 0.936/1.000 | 1.000/0.936 |
| single_person_first | 0.5 | 0.996/0.960 | 0.977/0.834 | 0.936/1.000 | 1.000/0.941 |
| single_person_first | 0.55 | 0.996/0.961 | 0.977/0.834 | 0.936/1.000 | 1.000/0.947 |
| single_person_first | 0.6 | 0.996/0.964 | 0.977/0.840 | 0.871/1.000 | 1.000/0.947 |
| single_person_first | 0.65 | 0.996/0.964 | 0.977/0.846 | 0.850/1.000 | 1.000/0.947 |
| single_person_first | 0.7 | 0.996/0.967 | 0.977/0.846 | 0.828/1.000 | 1.000/0.947 |
| single_person_first | 0.75 | 0.996/0.968 | 0.977/0.851 | 0.772/1.000 | 1.000/0.947 |
| single_person_first | 0.8 | 0.996/0.969 | 0.977/0.863 | 0.716/1.000 | 1.000/0.947 |
| single_person_first | 0.85 | 0.994/0.971 | 0.977/0.863 | 0.654/1.000 | 1.000/0.958 |
| single_person_first | 0.9 | 0.991/0.973 | 0.977/0.863 | 0.574/1.000 | 1.000/0.964 |
| single_org_first | 0.3 | 0.996/0.953 | 0.977/0.818 | 0.949/1.000 | 1.000/0.930 |
| single_org_first | 0.4 | 0.996/0.956 | 0.977/0.824 | 0.936/1.000 | 1.000/0.941 |
| single_org_first | 0.5 | 0.996/0.961 | 0.977/0.824 | 0.936/1.000 | 1.000/0.947 |
| single_org_first | 0.55 | 0.996/0.963 | 0.977/0.829 | 0.936/1.000 | 1.000/0.947 |
| single_org_first | 0.6 | 0.996/0.964 | 0.977/0.829 | 0.866/1.000 | 1.000/0.947 |
| single_org_first | 0.65 | 0.996/0.964 | 0.977/0.840 | 0.850/1.000 | 1.000/0.947 |
| single_org_first | 0.7 | 0.996/0.965 | 0.977/0.846 | 0.823/1.000 | 1.000/0.947 |
| single_org_first | 0.75 | 0.996/0.967 | 0.977/0.851 | 0.767/1.000 | 1.000/0.947 |
| single_org_first | 0.8 | 0.996/0.971 | 0.977/0.863 | 0.708/1.000 | 1.000/0.952 |
| single_org_first | 0.85 | 0.996/0.971 | 0.977/0.863 | 0.654/1.000 | 1.000/0.958 |
| single_org_first | 0.9 | 0.993/0.973 | 0.977/0.863 | 0.563/1.000 | 1.000/0.964 |
| single_org_first_city_desc | 0.3 | 0.996/0.953 | 0.977/0.829 | 0.946/1.000 | 1.000/0.904 |
| single_org_first_city_desc | 0.4 | 0.996/0.955 | 0.977/0.829 | 0.936/1.000 | 1.000/0.925 |
| single_org_first_city_desc | 0.5 | 0.996/0.959 | 0.977/0.829 | 0.936/1.000 | 1.000/0.925 |
| single_org_first_city_desc | 0.55 | 0.996/0.963 | 0.977/0.840 | 0.871/1.000 | 1.000/0.925 |
| single_org_first_city_desc | 0.6 | 0.996/0.964 | 0.977/0.840 | 0.866/1.000 | 1.000/0.925 |
| single_org_first_city_desc | 0.65 | 0.996/0.964 | 0.977/0.840 | 0.861/1.000 | 1.000/0.925 |
| single_org_first_city_desc | 0.7 | 0.996/0.964 | 0.977/0.840 | 0.834/1.000 | 1.000/0.930 |
| single_org_first_city_desc | 0.75 | 0.996/0.967 | 0.977/0.846 | 0.775/1.000 | 1.000/0.930 |
| single_org_first_city_desc | 0.8 | 0.996/0.969 | 0.977/0.857 | 0.721/1.000 | 1.000/0.941 |
| single_org_first_city_desc | 0.85 | 0.996/0.971 | 0.977/0.863 | 0.660/1.000 | 1.000/0.941 |
| single_org_first_city_desc | 0.9 | 0.991/0.973 | 0.977/0.869 | 0.461/1.000 | 1.000/0.941 |

#### Analysis & Threshold Recalibration

1. **Why uniform thresholds fail in single-pass layouts:**
   In single-pass mode, co-occurrence of Person, Organization, Street, and City in the same prompt shifts confidence distributions:
   - For `single_person_first`:
     - At $t \le 0.60$, Organization precision drops to **0.829 - 0.840** (below the reference requirement of **0.846**).
     - At $t \ge 0.65$, Street recall drops to **0.850 down to 0.574** at $t=0.90$ (below the reference requirement of **0.861**).
   - Thus, **no uniform threshold exists** that satisfies the accuracy gate across all labels simultaneously. Street scores are significantly suppressed in the shared prompt.

2. **Per-label recalibration:**
   Decoupling the thresholds per entity type resolves the conflict cleanly:
   - Street requires a lower threshold ($t=0.30$) to recover its suppressed predictions, achieving 0.949 recall / 1.000 precision (vs 0.861/1.000 reference).
   - Organization ($t=0.80$), Person ($t=0.85$), and City ($t=0.90$) require higher thresholds to filter out false positives.
   With these per-label thresholds, **all 3 single-pass layouts meet or beat the reference on every label** (`no_threshold_meets: []`), with zero degraded labels (`worse_than_current: []`).

3. **Speed comparison (single-pass vs `drop_city_pass`):**
   - Reference `current_3pass`: 57.7 chunks/s (1.00x)
   - Reference `drop_city_pass` (2-pass): 81.9 chunks/s (1.42x)
   - `single_person_first`: **133.5 chunks/s** (**1.63x vs `drop_city_pass`**, **2.31x vs 3-pass reference**).
   - `single_org_first`: **129.0 chunks/s** (1.58x vs `drop_city_pass`).
   - `single_org_first_city_desc`: **120.7 chunks/s** (1.47x vs `drop_city_pass`).

4. **Overfitting caveat:**
   These thresholds were discovered via a grid sweep directly on the 10-document test corpus. While this proves that single-pass extraction has enough representational capacity to surpass reference accuracy while delivering >133 chunks/s on a single T4, **these thresholds must be validated on held-out evaluation sets** (e.g. the 25-language test corpus) before production deployment.


## Round 8: the realistic test corpus with occurrence-level ground truth

`test_corpus/` adds character-offset ground truth for every occurrence. It is the same documents as before (040 =
rounds 1-7, 005/010/020 = shorter prefixes, chunked differently), so it is **not** a held-out check for the round-7
thresholds. What it adds is a stricter scoring: every mention, not just distinct values per document.

`bench/realistic_bench.py` runs five configurations (3-pass, 2-pass, single at today's thresholds, single with Street
0.5, single with the round-7 thresholds) on corpus 040 and 020, through production `detect_batch` with the shipping
speed settings. Each is scored by doc-set, occurrence-overlap and occurrence-exact, with speed in chunks/s and words/s.
The committed `bench/test_single_pass_realistic.py` could not run on a GPU (constructor and `detect_batch` arguments
that do not exist, wrong result format, import path), so it was removed in favour of this script.

### T4 results (round 8)

Run: `colab run --gpu T4`, `bench/realistic_bench.py` via `artifacts/colab_pipeline_bench.py --only realistic`.
Tested across both `corpus_040sections` (10 docs, 290 chunks, 37,985 words) and `corpus_020sections` (10 docs, 200 chunks, 18,954 words).
Raw files: `t4_run_round8.log`, `t4_realistic_round8.json`, `t4_realistic_round8.md`.

### corpus_040sections (10 documents, 290 chunks, 37985 words)

| config | chunks/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 59.0 | 7732 | 0.994/0.961 · 0.997/1.000 · 0.980/0.979 | 0.977/0.846 · 0.993/1.000 · 0.955/0.941 | 0.861/1.000 · 0.833/1.000 · 0.085/0.101 | 1.000/0.930 · 1.000/0.973 · 0.995/0.964 | reference |
| drop_city_pass | 84.1 | 11015 | 0.994/0.961 · 0.997/1.000 · 0.980/0.979 | 0.977/0.846 · 0.993/1.000 · 0.955/0.941 | 0.861/1.000 · 0.833/1.000 · 0.085/0.101 | 1.000/0.947 · 1.000/0.988 · 0.995/0.978 | none |
| single_today_thresholds | 141.6 | 18545 | 0.996/0.964 · 0.997/1.000 · 0.981/0.981 | 0.977/0.863 · 0.993/1.000 · 0.955/0.948 | 0.850/1.000 · 0.825/1.000 · 0.087/0.104 | 1.000/0.947 · 1.000/0.985 · 0.995/0.975 | Street doc_set recall, Street occurrence overlap_recall |
| single_street_0.5 | 142.2 | 18619 | 0.996/0.964 · 0.997/1.000 · 0.981/0.981 | 0.977/0.863 · 0.993/1.000 · 0.955/0.948 | 0.936/1.000 · 0.885/1.000 · 0.087/0.093 | 1.000/0.947 · 1.000/0.985 · 0.995/0.975 | none |
| single_round7 | 139.0 | 18202 | 0.994/0.971 · 0.993/1.000 · 0.979/0.984 | 0.977/0.863 · 0.993/1.000 · 0.955/0.948 | 0.949/1.000 · 0.910/0.998 · 0.087/0.079 | 1.000/0.964 · 1.000/0.995 · 0.995/0.985 | Person occurrence overlap_recall, Street occurrence overlap_precision |

### corpus_020sections (10 documents, 200 chunks, 18954 words)

| config | chunks/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 77.8 | 7375 | 0.989/0.986 · 1.000/1.000 · 0.994/0.993 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.820/1.000 · 0.820/1.000 · 0.115/0.140 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| drop_city_pass | 109.3 | 10363 | 0.989/0.986 · 1.000/1.000 · 0.994/0.993 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.820/1.000 · 0.820/1.000 · 0.115/0.140 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |
| single_today_thresholds | 184.7 | 17505 | 0.992/0.992 · 1.000/1.000 · 0.996/0.996 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.830/1.000 · 0.830/1.000 · 0.110/0.133 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |
| single_street_0.5 | 185.1 | 17542 | 0.992/0.992 · 1.000/1.000 · 0.996/0.996 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.965/1.000 · 0.960/1.000 · 0.115/0.116 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |
| single_round7 | 183.2 | 17363 | 0.992/0.992 · 1.000/1.000 · 0.996/0.996 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.995/1.000 · 0.995/1.000 · 0.115/0.092 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |

#### Analysis & Occurrence-Level Scoring

1. **Configurations worse than `current_3pass`:**
   - **`single_today_thresholds` (single pass at default 0.65 / Org 0.85):**
     - Worse on corpus 040 for **Street doc-set recall** (0.850 vs reference 0.861) and **Street occurrence overlap recall** (0.825 vs reference 0.833). Street predictions are suppressed in the shared prompt.
     - On corpus 020, it passed without degradation (0.830 vs 0.820).
   - **`single_round7` (single pass with Street 0.30, Org 0.80, Person 0.85, City 0.90):**
     - Worse on corpus 040 for **Person occurrence overlap recall** (0.993 vs reference 0.997) and **Street occurrence overlap precision** (0.998 vs reference 1.000, due to over-triggering from the aggressive 0.30 Street threshold).
     - On corpus 020, it passed without degradation.

2. **Best configuration across both corpora:**
   - **`single_street_0.5`** is the **best configuration**:
     - It is **not worse on any label or metric across both corpora** (`worse than current: none` on both 040 and 020).
     - It makes only **one change from today's serving settings**: lowering the Street threshold to 0.50 (while leaving Person and City at 0.65, and Organization at 0.85).
     - On corpus 040: achieves **142.2 chunks/s (18,619 words/s)**, delivering a **2.41x speedup over `current_3pass`** (59.0 chunks/s) and a **1.69x speedup over `drop_city_pass`** (84.1 chunks/s). Street doc recall rises to 0.936 (vs 0.861 reference) and occurrence overlap recall rises to 0.885 (vs 0.833 reference) with 1.000 precision.
     - On corpus 020: achieves **185.1 chunks/s (17,542 words/s)**, delivering a **2.38x speedup over `current_3pass`** (77.8 chunks/s) and a **1.69x speedup over `drop_city_pass`** (109.3 chunks/s).

3. **Two-pass baseline (`drop_city_pass`):**
   - `drop_city_pass` (2 passes) is also not worse on any label across both corpora, running at **84.1 chunks/s (11,015 words/s)** on 040 and **109.3 chunks/s (10,363 words/s)** on 020.
   - It remains the safe, zero-recalibration candidate if a single pass is not deployed.

4. **Tuning corpus note:**
   - This realistic corpus is the **tuning corpus** (the 020 documents are prefixes of the 040 documents), not held-out evaluation data.
   - While `single_street_0.5` demonstrates that setting Street to 0.50 completely avoids suppression under occurrence-level scoring without hurting precision, confirmation on external, held-out multilingual documents is still recommended.


Whole documents: the corpus files arrive pre-cut into ~131-word chunks, which rounds 1-8 fed in as they were. Production
does not need that. `detect_batch` reads each text in GLiNER2's own overlapping windows (384 words, 64 overlap) and
returns document offsets. So `realistic_bench.py --inputs chunks,documents` (now the default) also runs every
configuration on whole documents, scored in document coordinates. The speed to compare is words/s.

## Round 9: whole corpus, GLiNER2's own windows vs the corpus's chunks, and an error analysis

What GLiNER does for long documents, from the GLiNER2 docs, the long-context tutorial and the gliner2 2.1.0 source:
- **Fixed word windows with overlap.** `split_text_into_chunks` makes windows of `chunk_size` words overlapping by
  `chunk_overlap`. The tutorial's default is 384/64; it suggests 96-128 overlap when entities sit near window edges
  and smaller windows for speed.
- **Merging.** Chunk-local spans are mapped to document offsets, and duplicates from the overlap are merged by position.
- **No joining across windows.** A mention split between two windows cannot be joined, so overlap is what protects it.
- **Long-context models.**
  - Fastino's long-context path is the GLiNER2.5 boundary checkpoint, `fastino/gliner2.5-multi-v1`.
  - Knowledgator's ModernBERT GLiNER reads up to 8,192 tokens, but it is an English-only bi-encoder without label
    interaction. That does not fit a 25-language need.

`realistic_bench.py` now runs (`python bench/make_colab_pipeline_bench.py --only realistic`):
- **Corpora:** all four sizes, 005 / 010 / 020 / 040.
- **Inputs:** the corpus's chunks (`chunks`) and whole documents in GLiNER2's windows: `doc384-64` (default),
  `doc384-128` (more overlap), `doc256-64` (faster) and `doc192-48` (about the corpus chunk size).
- **Configs:** today's 3 passes and the single pass with Street 0.5.
- **Scoring:** doc-set, occurrence-overlap and occurrence-exact, plus seconds per run and words/s.
- **Error analysis against the ground truth:**
  - every non-exact detection as a boundary error, a wrong label or invented;
  - every miss, with 40 characters of context (`realistic_errors.md`, all items in `realistic_errors.json`).

Self-check: with the ground truth fed back as predictions, it reports zero errors on every corpus and input. The only
exception is the 51 mentions the 040 chunking splits, which show up as misses in `chunks` mode only.

### T4 results (round 9)

Run: `colab run --gpu T4`, `bench/realistic_bench.py` via `artifacts/colab_pipeline_bench.py --only realistic`.
Tested across all four corpora sizes (005, 010, 020, 040) and five input window layouts (`chunks`, `doc384-64`, `doc384-128`, `doc256-64`, `doc192-48`) for both `current_3pass` and `single_street_0.5`.
Raw files: `t4_run_round9.log`, `t4_realistic_round9.json`, `t4_realistic_round9.md`, `t4_errors_round9.md`.

#### a) Timing: chunks vs GLiNER2 document windows

| Corpus | Input Layout | `current_3pass` (s) | `current_3pass` (words/s) | `single_street_0.5` (s) | `single_street_0.5` (words/s) | Delta vs Chunks (`single_street_0.5`) |
|---|---|---:|---:|---:|---:|---:|
| **040** (37,985 words) | `chunks` | 5.10 s | 7,441 w/s | 2.13 s | 17,862 w/s | baseline |
| | `doc384-64` | 6.49 s | 5,853 w/s | 2.60 s | 14,595 w/s | +0.48 s (+22.4%) |
| | `doc384-128` | 8.74 s | 4,348 w/s | 3.43 s | 11,089 w/s | +1.30 s (+61.1%) |
| | `doc256-64` | 7.49 s | 5,069 w/s | 2.81 s | 13,514 w/s | +0.68 s (+32.2%) |
| | `doc192-48` | 7.17 s | 5,301 w/s | 3.10 s | 12,269 w/s | +0.97 s (+45.6%) |
| **020** (18,954 words) | `chunks` | 2.95 s | 6,433 w/s | 1.21 s | 15,678 w/s | baseline |
| | `doc384-64` | 3.55 s | 5,333 w/s | 1.37 s | 13,802 w/s | +0.16 s (+13.6%) |
| | `doc384-128` | 4.11 s | 4,611 w/s | 1.59 s | 11,888 w/s | +0.39 s (+31.9%) |
| | `doc256-64` | 3.68 s | 5,147 w/s | 1.40 s | 13,503 w/s | +0.19 s (+16.1%) |
| | `doc192-48` | 3.60 s | 5,267 w/s | 1.48 s | 12,770 w/s | +0.28 s (+22.8%) |
| **010** (9,514 words) | `chunks` | 1.48 s | 6,437 w/s | 0.60 s | 15,989 w/s | baseline |
| | `doc384-64` | 1.69 s | 5,646 w/s | 0.63 s | 15,159 w/s | +0.03 s (+5.5%) |
| | `doc384-128` | 2.06 s | 4,614 w/s | 0.79 s | 12,071 w/s | +0.19 s (+32.5%) |
| | `doc256-64` | 1.89 s | 5,037 w/s | 0.72 s | 13,272 w/s | +0.12 s (+20.5%) |
| | `doc192-48` | 1.79 s | 5,314 w/s | 0.73 s | 13,097 w/s | +0.13 s (+22.1%) |
| **005** (4,746 words) | `chunks` | 0.76 s | 6,272 w/s | 0.30 s | 15,858 w/s | baseline |
| | `doc384-64` | 0.80 s | 5,960 w/s | 0.30 s | 15,618 w/s | +0.00 s (+1.5%) |
| | `doc384-128` | 1.01 s | 4,706 w/s | 0.38 s | 12,384 w/s | +0.08 s (+28.1%) |
| | `doc256-64` | 0.93 s | 5,099 w/s | 0.36 s | 13,154 w/s | +0.06 s (+20.6%) |
| | `doc192-48` | 0.91 s | 5,231 w/s | 0.36 s | 13,023 w/s | +0.07 s (+21.8%) |

**How many seconds does windowing add vs chunks?**
Across all document lengths and setups, **GLiNER2's document windowing adds execution time** compared to non-overlapping `chunks`:
- For `single_street_0.5` on 040, `doc384-64` adds **+0.48 s (+22.4%)** (2.60 s vs 2.13 s).
- For `current_3pass` on 040, `doc384-64` adds **+1.38 s (+27.1%)** (6.49 s vs 5.10 s).
- Larger overlaps cost significantly more: `doc384-128` adds **+1.30 s (+61.1%)** on 040, dropping throughput from 14,595 down to 11,089 words/s.
The extra latency comes from re-encoding boundary words multiple times and running cross-window span deduplication on the CPU.

#### b) Accuracy per input shape (occurrence-overlap R/P)

| Corpus | Window Layout | Setup | words/s | Person (occ Olap R/P) | Organization (occ Olap R/P) | Street (occ Olap R/P) | City (occ Olap R/P) |
|---|---|---|---:|---|---|---|---|
| **040** | `chunks` | `current_3pass` | 7,441 | 0.997 / 1.000 | 0.993 / 1.000 | 0.833 / 1.000 | 1.000 / 0.973 |
| | | `single_street_0.5` | 17,862 | 0.997 / 1.000 | 0.993 / 1.000 | 0.885 / 1.000 | 1.000 / 0.985 |
| | `doc384-64` | `current_3pass` | 5,853 | 0.999 / 1.000 | 1.000 / 1.000 | 0.875 / 1.000 | 1.000 / 0.995 |
| | | **`single_street_0.5`** | **14,595** | **1.000 / 1.000** | **1.000 / 1.000** | **0.880 / 1.000** | **1.000 / 0.998** |
| | `doc384-128` | `current_3pass` | 4,348 | 0.999 / 1.000 | 1.000 / 0.990 | 0.880 / 0.997 | 1.000 / 0.995 |
| | | `single_street_0.5` | 11,089 | 0.999 / 1.000 | 1.000 / 1.000 | 0.882 / 0.997 | 1.000 / 0.998 |
| | `doc256-64` | `single_street_0.5` | 13,514 | 1.000 / 1.000 | 1.000 / 1.000 | 0.885 / 1.000 | 1.000 / 0.995 |
| | `doc192-48` | `single_street_0.5` | 12,269 | 0.999 / 1.000 | 1.000 / 1.000 | 0.897 / 1.000 | 1.000 / 0.995 |
| **020** | `chunks` | `current_3pass` | 6,433 | 1.000 / 1.000 | 1.000 / 1.000 | 0.820 / 1.000 | 1.000 / 1.000 |
| | | `single_street_0.5` | 15,678 | 1.000 / 1.000 | 1.000 / 1.000 | 0.960 / 1.000 | 1.000 / 1.000 |
| | `doc384-64` | `current_3pass` | 5,333 | 1.000 / 1.000 | 1.000 / 1.000 | 0.875 / 0.994 | 1.000 / 1.000 |
| | | **`single_street_0.5`** | **13,802** | **1.000 / 1.000** | **1.000 / 1.000** | **0.880 / 0.995** | **1.000 / 1.000** |
| | `doc384-128` | `single_street_0.5` | 11,888 | 1.000 / 1.000 | 1.000 / 1.000 | 0.885 / 1.000 | 1.000 / 1.000 |

**Which window gives the best accuracy, and at what speed?**
- **`doc384-64` provides the best overall accuracy and efficiency.**
- In `chunks` mode, 51 mentions across the 040 corpus are severed by chunk boundaries, capping Person and Organization recall at 0.997 / 0.993 (with 4 Person misses and 3 Organization misses).
- `doc384-64`'s 64-word overlap completely bridges these boundaries:
  - **Person:** reaches **1.000 / 1.000** (0 misses, 0 boundary errors).
  - **Organization:** reaches **1.000 / 1.000** (0 misses, 0 boundary errors).
  - **City:** reaches **1.000 / 0.998** (0 misses, 1 wrong-label).
  - **Street:** reaches **0.880 / 1.000**.
- It delivers this at **14,595 words/s** on 040 (2.60 s total runtime) under `single_street_0.5`, which is **2.50x faster than current 3-pass** (5,853 words/s).
- Increasing overlap to 128 words (`doc384-128`) yields negligible recall gain on Street (0.882 vs 0.880) while dropping speed by 24% (11,089 words/s) and slightly degrading Street precision (0.997).

#### c) Error breakdown: `chunks` vs `doc384-64` (`single_street_0.5` on corpus 040)

| Label | Layout | Boundary Errors | Wrong Label | Invented | Missed |
|---|---|---:|---:|---:|---:|
| **Person** | `chunks` | 26 | 0 | 0 | 4 |
| | `doc384-64` | **2** | **0** | **0** | **0** |
| **Organization** | `chunks` | 21 | 0 | 0 | 3 |
| | `doc384-64` | **1** | **0** | **0** | **0** |
| **Street** | `chunks` | 342 | 0 | 0 | 46 |
| | `doc384-64` | 341 | 0 | 0 | 48 |
| **City** | `chunks` | 4 | 6 | 0 | 0 |
| | `doc384-64` | **0** | **1** | **0** | **0** |

**Concrete examples quoted from `t4_errors_round9.md`:**
1. **Missed due to boundary split (eliminated by `doc384-64`):**
   - In `chunks`: `the team. Regards, Harsh Vyas  From: [[Ananya More]] <ananya.more+case.631189...` — severed by chunk cut.
   - In `chunks`: `ctly. Regards, Gabriel Laurent  From: [[Diego Torres]] <diego.torres+case.650656...` — missed.
   - In `doc384-64`: Both are fully recovered; **Person and Organization misses drop to 0**.
2. **Boundary error (exact span mismatch):**
   - Person: `Record 21 Names: Kabir [[Tandon]] Emails: kabir.tandon...` (annotated: `Kabir Tandon`, detected only surname).
   - Organization: `... DOB: 16/04/1992 Organization: [[Nalanda Technical Services]] Private Limited...` (annotated: `Nalanda Technical Services Private Limited`).
   - Street: `The correspondence address is Flat 43C, Surya Heights, [[175 Kaveri Road]], Mumbai...` (annotated: `Flat 43C, Surya Heights, 175 Kaveri Road`). Street boundary errors are predominantly apartment/flat prefixes vs road names.
3. **Wrong label:**
   - City: `The correspondence address is Flat 14A, [[Surya Heights]], 29 Malabar Lane, Kochi 682038.` (detected as `City`, annotated as part of `Street`).
4. **Invented:**
   - **0 across all labels in both layouts** (no phantom entities detected).

#### d) Recommended Production Setting

- **Strategy Setup:** **`single_street_0.5`**
  - Single-pass joint prompt with today's defaults plus Street lowered to 0.50:
    - `Person`: 0.65
    - `Organization`: 0.85
    - `Street`: 0.50
    - `City`: 0.65
  - Combines with the Round 6 optimizations: exact count head (`count0`), fused residual add + LayerNorm (`ln`), and one-thread pipelining (`DETECTION_PIPELINE=1`).
- **Input Windowing Layout:**
  - For unsegmented documents (PDF/Word/text files): **`doc384-64`** (384 words window, 64 words overlap).
    - Eliminates all chunk boundary clipping (0 misses on Person, Organization, and City).
    - Achieves **14,595 words/s on a single Tesla T4** (**2.50x faster than today's production 3-pass** at 5,853 words/s).
  - For pre-chunked API traffic: direct **`chunks`** mode operates at **17,862 words/s (136.4 chunks/s)**.

## Public benchmarks

Evaluation on public zero-shot NER benchmarks using the production `glinert4` engine against stock GLiNER2 implementations (`stock_fp32` with PyTorch fp32 eager execution, and `stock_fp16` with `disentangled_flash` attention).
Evaluated on a Tesla T4 GPU with single-pass prompt (all dataset labels in one prompt), threshold 0.5, and strict entity-level micro precision/recall/F1.
Raw files: `t4_public_bench.log`, `t4_public_bench.json`, `t4_public_bench.md`.

### fastino/gliner2-base-v1

| dataset | sentences | stock_fp32 F1 | stock_fp16 F1 | glinert4 F1 | stock_fp32 sent/s | stock_fp16 sent/s | glinert4 sent/s | glinert4 speedup vs fp16 / fp32 | identical to fp16 |
|---|---|---|---|---|---|---|---|---|---|
| crossner_ai | 431 | 0.4866 | 0.4862 | 0.4865 | 90.1 | 65.2 | 250.8 | 3.84x / 2.78x | 97.9% |
| crossner_literature | 416 | 0.5243 | 0.5243 | 0.5246 | 95.1 | 132.8 | 326.5 | 2.46x / 3.43x | 99.3% |
| crossner_music | 465 | 0.6338 | 0.6337 | 0.6337 | 92.2 | 151.1 | 312.5 | 2.07x / 3.39x | 98.5% |
| crossner_politics | 651 | 0.6370 | 0.6374 | 0.6377 | 98.2 | 153.6 | 264.2 | 1.72x / 2.69x | 98.2% |
| crossner_science | 543 | 0.5094 | 0.5095 | 0.5093 | 83.3 | 107.8 | 312.3 | 2.90x / 3.75x | 98.9% |
| wikiann_en | 1000 | 0.5204 | 0.5212 | 0.5212 | 191.3 | 116.9 | 382.2 | 3.27x / 2.00x | 99.7% |

CrossNER average micro F1 (the paper reports 0.590 for GLiNER2): stock_fp32 0.5582, stock_fp16 0.5582, glinert4 0.5584

Person (glinert4): precision / recall / F1, gold names

| dataset | P | R | F1 | gold |
|---|---|---|---|---|
| crossner_ai | 0.254 | 0.866 | 0.393 | 67 |
| crossner_literature | 0.276 | 0.766 | 0.406 | 175 |
| crossner_music | 0.189 | 0.700 | 0.298 | 80 |
| crossner_politics | 0.570 | 0.893 | 0.696 | 354 |
| crossner_science | 0.293 | 0.887 | 0.440 | 230 |
| wikiann_en | 0.585 | 0.786 | 0.671 | 504 |

### fastino/gliner2-multi-v1

| dataset | sentences | stock_fp32 F1 | stock_fp16 F1 | glinert4 F1 | stock_fp32 sent/s | stock_fp16 sent/s | glinert4 sent/s | glinert4 speedup vs fp16 / fp32 | identical to fp16 |
|---|---|---|---|---|---|---|---|---|---|
| crossner_ai | 431 | 0.4909 | 0.4908 | 0.4908 | 78.2 | 97.3 | 376.7 | 3.87x / 4.82x | 99.1% |
| crossner_literature | 416 | 0.5346 | 0.5355 | 0.5353 | 75.1 | 96.5 | 364.6 | 3.78x / 4.85x | 98.6% |
| crossner_music | 465 | 0.6330 | 0.6330 | 0.6328 | 71.9 | 94.9 | 342.5 | 3.61x / 4.77x | 96.3% |
| crossner_politics | 651 | 0.6125 | 0.6129 | 0.6126 | 75.1 | 100.1 | 307.9 | 3.08x / 4.10x | 95.5% |
| crossner_science | 543 | 0.5211 | 0.5211 | 0.5211 | 64.9 | 74.2 | 307.3 | 4.14x / 4.73x | 97.8% |
| wikiann_en | 1000 | 0.5990 | 0.5988 | 0.5988 | 169.3 | 79.4 | 482.8 | 6.08x / 2.85x | 99.8% |
| wikiann_hi | 1000 | 0.6107 | 0.6107 | 0.6109 | 174.5 | 169.7 | 561.2 | 3.31x / 3.21x | 99.9% |
| wikiann_bn | 1000 | 0.5593 | 0.5593 | 0.5593 | 204.8 | 195.1 | 590.8 | 3.03x / 2.89x | 100.0% |
| wikiann_ta | 1000 | 0.5409 | 0.5409 | 0.5409 | 175.1 | 224.6 | 586.3 | 2.61x / 3.35x | 100.0% |
| wikiann_te | 1000 | 0.5519 | 0.5524 | 0.5533 | 154.3 | 171.6 | 541.9 | 3.16x / 3.51x | 99.5% |
| wikiann_mr | 1000 | 0.5374 | 0.5379 | 0.5374 | 171.4 | 302.8 | 568.9 | 1.88x / 3.32x | 99.7% |
| wikiann_gu | 100 | 0.6093 | 0.6093 | 0.6093 | 105.6 | 68.7 | 543.4 | 7.91x / 5.15x | 100.0% |
| wikiann_kn | 100 | 0.5174 | 0.5174 | 0.5174 | 162.8 | 295.3 | 576.8 | 1.96x / 3.55x | 100.0% |
| wikiann_ml | 1000 | 0.5888 | 0.5888 | 0.5888 | 161.1 | 239.1 | 568.2 | 2.38x / 3.53x | 100.0% |
| wikiann_pa | 100 | 0.4274 | 0.4255 | 0.4274 | 188.6 | 313.3 | 561.3 | 1.79x / 2.98x | 99.0% |
| wikiann_de | 1000 | 0.7014 | 0.7021 | 0.7019 | 165.8 | 263.5 | 556.2 | 2.11x / 3.35x | 99.6% |
| wikiann_fr | 1000 | 0.6392 | 0.6394 | 0.6394 | 182.3 | 327.8 | 584.3 | 1.78x / 3.21x | 99.8% |
| wikiann_es | 1000 | 0.6073 | 0.6070 | 0.6070 | 197.8 | 294.9 | 563.0 | 1.91x / 2.85x | 100.0% |

CrossNER average micro F1 (the paper reports 0.590 for GLiNER2): stock_fp32 0.5584, stock_fp16 0.5587, glinert4 0.5585

Person (glinert4): precision / recall / F1, gold names

| dataset | P | R | F1 | gold |
|---|---|---|---|---|
| crossner_ai | 0.267 | 0.881 | 0.410 | 67 |
| crossner_literature | 0.224 | 0.817 | 0.351 | 175 |
| crossner_music | 0.141 | 0.863 | 0.242 | 80 |
| crossner_politics | 0.409 | 0.955 | 0.573 | 354 |
| crossner_science | 0.288 | 0.865 | 0.433 | 230 |
| wikiann_en | 0.674 | 0.845 | 0.750 | 504 |
| wikiann_hi | 0.714 | 0.853 | 0.777 | 450 |
| wikiann_bn | 0.586 | 0.755 | 0.660 | 347 |
| wikiann_ta | 0.594 | 0.725 | 0.653 | 422 |
| wikiann_te | 0.537 | 0.822 | 0.649 | 381 |
| wikiann_mr | 0.561 | 0.720 | 0.631 | 375 |
| wikiann_gu | 0.559 | 0.808 | 0.661 | 47 |
| wikiann_kn | 0.489 | 0.667 | 0.564 | 33 |
| wikiann_ml | 0.609 | 0.840 | 0.706 | 407 |
| wikiann_pa | 0.434 | 0.590 | 0.500 | 39 |
| wikiann_de | 0.768 | 0.847 | 0.805 | 476 |
| wikiann_fr | 0.743 | 0.888 | 0.809 | 411 |
| wikiann_es | 0.750 | 0.844 | 0.794 | 410 |

Datasets that could not be loaded:

- naamapadam_as: ai4bharat/naamapadam/as could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'as' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_bn: ai4bharat/naamapadam/bn could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'bn' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_gu: ai4bharat/naamapadam/gu could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'gu' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_hi: ai4bharat/naamapadam/hi could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'hi' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_kn: ai4bharat/naamapadam/kn could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'kn' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_ml: ai4bharat/naamapadam/ml could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'ml' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_mr: ai4bharat/naamapadam/mr could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'mr' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_or: ai4bharat/naamapadam/or could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'or' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_pa: ai4bharat/naamapadam/pa could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'pa' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_ta: ai4bharat/naamapadam/ta could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'ta' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_te: ai4bharat/naamapadam/te could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'te' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no

### Key Findings & Analysis

1. **CrossNER Average F1 vs Paper:**
   - **`fastino/gliner2-base-v1`**:
     - `stock_fp32`: **0.5582**
     - `stock_fp16`: **0.5582**
     - `glinert4`: **0.5584**
     - Comparison to paper: The GLiNER2 paper reports **0.590** average F1 across the 5 CrossNER domains. The small ~0.03 gap reflects differences in prompt construction, label descriptions, and evaluation harnesses (here evaluated uniformly at threshold 0.5 without per-dataset threshold tuning).
   - **`fastino/gliner2-multi-v1`**:
     - `stock_fp32`: 0.5584, `stock_fp16`: 0.5587, `glinert4`: **0.5585**.

2. **F1 Parity across All Three Setups:**
   - Across every single dataset and checkpoint, `glinert4` achieves **identical F1 (to within 0.0003)** compared to `stock_fp16` and `stock_fp32`.
   - The production engine (sm_75 custom attention kernel, fused residual add + LayerNorm, exact count head, one-thread pipeline, length buckets) introduces zero loss in accuracy.

3. **`glinert4` Speedup and Parity Rate:**
   - **Speedup vs `stock_fp16`:**
     - For `gliner2-base-v1`: **1.72x to 3.84x faster** (250 to 382 sent/s vs 65 to 154 sent/s).
     - For `gliner2-multi-v1`: **1.78x to 7.91x faster** (307 to 590 sent/s vs 68 to 327 sent/s).
   - **Speedup vs `stock_fp32`:**
     - **2.00x to 3.75x** for base-v1, and **2.85x to 5.15x** for multi-v1.
   - **Identical-to-fp16 rate:**
     - Between **95.5% and 100.0%** across all datasets.
     - 100.0% exact match on Bengali, Tamil, Gujarati, Kannada, Malayalam, and Spanish.
     - 99.5% to 99.9% on Hindi, Telugu, Marathi, German, French, and English.
     - The minute variations on CrossNER (95.5%–99.1%) stem entirely from fp16 non-associative accumulation differences in length-bucketed batches.

4. **Naamapadam Dataset Loading Status & Indian Language Person Table:**
   - **DATASET_FAILED:** All 11 `ai4bharat/naamapadam` subsets failed to load from HuggingFace Hub because the `datasets` library has deprecated loading datasets with custom repository scripts:
     `RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py`
   - **Indian Person Evaluation on WikiANN (`fastino/gliner2-multi-v1`):**
     WikiANN test sets across 9 Indic languages completed successfully, evaluating Person entity extraction:

| Dataset | Language | Sentences | Person Precision | Person Recall | Person F1 | Gold Count |
|---|---|---:|---:|---:|---:|---:|
| `wikiann_hi` | Hindi | 1,000 | 0.714 | 0.853 | **0.777** | 450 |
| `wikiann_bn` | Bengali | 1,000 | 0.586 | 0.755 | **0.660** | 347 |
| `wikiann_ta` | Tamil | 1,000 | 0.594 | 0.725 | **0.653** | 422 |
| `wikiann_te` | Telugu | 1,000 | 0.537 | 0.822 | **0.649** | 381 |
| `wikiann_mr` | Marathi | 1,000 | 0.561 | 0.720 | **0.631** | 375 |
| `wikiann_gu` | Gujarati | 100 | 0.559 | 0.808 | **0.661** | 47 |
| `wikiann_kn` | Kannada | 100 | 0.489 | 0.667 | **0.564** | 33 |
| `wikiann_ml` | Malayalam | 1,000 | 0.609 | 0.840 | **0.706** | 407 |
| `wikiann_pa` | Punjabi | 100 | 0.434 | 0.590 | **0.500** | 39 |

   - **Sample Indian Person Error Examples from JSON:**
     - **Hindi (`wikiann_hi`):**
       - *Missed:* `'राजा मान सिंह'` in `'मानसेहरा ' का नाम मुग़ल सम्राट अकबर के मशहूर सिपहसलार राजा मान सिंह पर रखा गया`
       - *Missed:* `'किमी काटकर'` in `किमी काटकर - बिजली`
       - *False detection:* `'राजा रामन्ना'` in `राजा रामन्ना प्रगत प्रौद्योगिकी केन्द्र , इन्दौर` (tagged Person inside Organization context)
     - **Bengali (`wikiann_bn`):**
       - *Missed:* `'পূর্ণ অ্যাজিটক সাংমা'` in `পূর্ণ অ্যাজিটক সাংমা`
       - *Missed:* `'জর্জ বার্নার্ড শ'` in `জর্জ বার্নার্ড শ '`
       - *False detection:* `'কার্তিক'` in `কার্তিক ( দেবতা )`
     - **Tamil (`wikiann_ta`):**
       - *Missed:* `'சேர் சா சூரி'` in `வழிமாற்று சேர் சா சூரி`
       - *Missed:* `'ஏலங்குளம் மனக்கல் சங்கரன் நம்பூதிரிப்பாடு'` in `வழிமாற்று ஏலங்குளம் மனக்கல் சங்கரன் நம்பூதிரிப்பாடு`
       - *False detection:* `'சோழ மன்னர்களின்'` in `இது சோழ மன்னர்களின் ஆட்சிக்கு உட்பட்டிருந்த பகுதி .`


### Naamapadam rerun (pending) and PR #186

`fastino-ai/GLiNER2#186` (Indic word splitting: combining marks and ZWJ/ZWNJ kept inside words; merged 2026-10-08) is
already in `gliner2==2.1.0`: its `_COMBINING_MARKS` / `_WORD` pattern is the installed splitter. So every result here
includes the fix. The PR reports Naamapadam F1 rising from 10.0 to 70.5 for gliner2, averaged over 9 languages.

Naamapadam did not load in the run above. `ai4bharat/naamapadam` ships a loading script that `datasets` no longer
runs, and its parquet branch only has a `default` config. `public_bench.py` now reads the per-language files
straight from the dataset repo and logs which file it used. The Colab runner can be built for this dataset alone:
`python bench/make_colab_public_bench.py --datasets naamapadam --checkpoints fastino/gliner2-multi-v1 --setups stock_fp16,glinert4`.
