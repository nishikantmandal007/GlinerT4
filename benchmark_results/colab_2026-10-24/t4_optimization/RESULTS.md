# One-T4 GLiNER2 throughput and kernel tests — 2026-10-25

## Result

The matched single-T4 GLiNER2 2.1.0 run measured **54.81 chunks/s** over the 290-chunk synthetic corpus (three trials; median 5.291 s). The current target is 100 chunks/s, or 1.82x this measured rate. The two T4-specific memory-scheduling variants preserved outputs but did not improve throughput. Neither reached 100 chunks/s.

This is one T4, one serving thread, batch size 16, FP16, the same three label passes and no fast-count or result cache. The run used the existing custom attention kernel, word-split caching, one-time `eval()` and the exact bulk-token lookup helper. The benchmark checkpoint is `fastino/gliner2-multi-v1`; it is the same model as the earlier GLiNER2 2.1.0 Colab comparisons, not the separate `urchade/gliner_multi-v2.1` checkpoint named in the supplied optimization-plan document.

| Trial | Seconds | Chunks/s |
|---:|---:|---:|
| 1 | 5.240 | 55.34 |
| 2 | 5.291 | 54.81 |
| 3 | 5.327 | 54.44 |

The first run compiled the original T4 v4 kernel for SM75 on Colab's PyTorch 2.11.0+cu130, Triton 3.6.0, Transformers 5.20.2, Python 3.13 stack. Do not compare its absolute rate directly with the VM's older software stack or with earlier two-T4 aggregate rates.

## What the profile says

A PyTorch profile over 97 representative chunks and the same three extraction passes recorded 1.280 seconds of CUDA kernel time. The custom disentangled-attention kernel accounted for 326.8 ms (25.5%). Turing FP16 GEMM kernels accounted for about 48% of CUDA time. The CPU profile also shows material post-encoder work: span unpacking took 0.757 seconds of profiled self time over those 97 chunks; batch collation and formatting were also visible. Profilers add overhead, so these timings locate work and are not throughput numbers.

This does not establish a hard T4 capacity limit. It does show why attention tuning alone is unlikely to produce the roughly 1.82x needed: even halving the attention share would yield at most about 1.15x at the GPU-kernel level, before CPU work and other extraction stages.

## T4 kernel variants tested

Both candidates leave the MMA, score, masking, softmax and output arithmetic unchanged:

- `pos_sync.cu` removes register-held prefetch for relative-position key/query data and copies those tiles to shared memory immediately before use.
- `all_sync.cu` also removes next-tile K/V register prefetch.

The original build reported 255 registers/thread. Candidate builds reported 200–212 registers/thread for `pos_sync` and 190–207 for `all_sync`, with no compiler spill loads/stores. Lower register use did not improve these timings.

All three kernels were byte-identical on 279 raw attention cases: sequence boundaries, three random seeds, right padding, interior mask holes, all-masked rows and packed layouts. Full encoder outputs were byte-identical for batch-16 lengths 96, 256, 512 and 1024, both padded and unpadded (eight shapes). The complete JSON extraction outputs, including scores, spans and order, matched the original kernel on each of four completed paired corpus rounds.

| Path | Completed corpus samples | Median rate | Relative to original in same run |
|---|---:|---:|---:|
| Original v4 | 4 | 45.22 chunks/s | reference |
| `pos_sync` | 4 | 44.75 chunks/s | -1.0% |
| `all_sync` | 5 | 44.70 chunks/s | -1.1% |

These samples were slower overall than the clean baseline above, likely due to run/clock variation across the longer compile and test sequence. The paired measurements still show no candidate win. A repeated validation run reproduced the 54.81 baseline and reconfirmed byte equality through the 256-token shapes, then Colab pruned the T4 assignment before the remaining shapes and full-corpus round could finish. That interrupted repeat is not included in the table.

## Conclusion and next target

No correctness flaw was found in the tested v4 attention path. Register pressure is visible, but reducing it with synchronous shared-memory copies did not help on T4. T4 does not support the Ampere `cp.async` optimization tested on A100/L4, so that candidate cannot transfer to SM75.

The strongest next investigation is the non-attention path: profile span unpacking/count and token-to-span preparation, batch collation, and CPU/GPU overlap. Any fusion or batching change must retain the same labels, thresholds, chunk boundaries and model math, then pass full score/span parity. A 100 chunks/s single-T4 result remains unproven; these tests do not show it is impossible, but the two attention-prefetch changes do not move it closer.

## Artifacts and reproduction

- `baseline.py`: original-kernel build, 290-chunk baseline, CPU/GPU profiles and captured attention plan.
- `make_variants.py`, `variants_run.py`, `pos_sync.cu`, `all_sync.cu`: source generation, strict gates and paired measurements.
- `bootstrap.py`, `transfer.py`: disposable Colab environment setup and source transfer.
- `logs/`: raw Colab client outputs, including the interrupted repeat.

The Colab assignment is stopped. No production source or VM was changed.
