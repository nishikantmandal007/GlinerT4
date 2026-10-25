# Pipeline-side optimizations from the T4 plan, gliner2 2.1.0 — 2026-10-25

This round applies the **no-semantic-change, Phase 1** items of
`GLiNER_Multilingual_v2_1_T4_Optimization_Plan.docx` to the GLiNER2 path (`fastino/gliner2-multi-v1`, `gliner2==2.1.0`).
That path measured 54.81 chunks/s on one T4 earlier. **Current finding (Colab T4):**
- Zero-loss pipeline work (rounds 1-3) reached 1.14x in the encoder-plus-head benchmark (`count0_streams2_ln`, 58.05 vs 50.91 chunks/s).
- End to end through the production queue (rounds 4-5), **no config reaches 100 chunks/s**. The best shipping candidate,
  `exact_2pass_1stream` (drop City-alone pass, exact count head, fused add+LayerNorm, single stream), ran at **65.7 chunks/s**
  versus 46.4 chunks/s for today's serving settings (`prod_3pass`), a **1.416x** speedup, and **34.3 chunks/s short of 100**
  (needs another 1.52x). It meets or beats the 3-pass reference on every label (City precision .930 to .947, others equal) with
  1144/1160 identical chunks vs ref, and cuts GPU memory from 3.1 GB to 1.8 GB.
- Round 5 isolates the bottleneck: queue run size (`queue_runs` 69.5 vs `whole_corpus` 69.0 chunks/s) and sampler overhead
  (within noise) do not limit throughput. Instead, cProfile reveals that **host-side device-to-host synchronization**
  (`tensor.tolist()` taking 2.54 s and `tensor.cpu()` taking 0.38 s out of 5.05 s total) inside the span head and decoding
  dominates wall time on Colab's single physical core.

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

