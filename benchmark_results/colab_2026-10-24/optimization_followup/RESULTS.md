# A100 bottleneck and optimization follow-up — 2026-10-24

Confirmed: high register use restricts the attention kernel, while input preparation and count processing restrict whole GLiNER2 extraction. A one-line launch-bounds experiment improves longer encoder batches about 6%, with only about 1% extraction gain in this comparison. No production or original kernel source was modified.

## What was measured

- Same model, 290 unique synthetic corpus chunks, FP16, batch 16, three grouped passes, single GPU/thread. PyTorch 2.11+cu130, Triton 3.6, GLiNER2 2.1.0, DisentangledFlash 1.1.1, Transformers 5.20.2. Initial setup reuses the parent experiment bootstrap and corpus.
- Seven rotating-order encoder rounds (five forwards each), five rotating-order extraction rounds; compilation/warmup excluded.
- Existing 8-warp variant: no measurable overall gain (88.89 to 89.00 chunks/s in its paired block). All six encoder shapes were bit-identical and all 290 entity-value sets matched the 4-warp custom path.
- Two separate compiled copies change only launch bounds from `8 / WARPS` to `12 / WARPS` or `16 / WARPS`. Unique library/operator/launch names prevent symbol collisions. Four warps are used in these register experiments.

## Confirmed register bottleneck

Nsight Compute collected one real first-layer attention launch from a 16 x 512 batch. Tensor values and strides were preserved. These are single-launch hardware measurements, separate from the multi-round encoder benchmarks.

| Metric | Original | Three-block target |
|---|---:|---:|
| Registers/thread | 255 | 168 |
| Register-limited blocks/SM | 2 | 3 |
| Shared-memory-limited blocks/SM | 3 | 3 |
| Achieved active-warps occupancy | 12.33% | 17.85% |
| Attention launch | 0.988 ms | 0.864 ms |

The sampled attention launch has 1.144x throughput. The source holds Q/K/V and position-window prefetch data plus matrix accumulators in registers. Reducing that live state is a stronger next kernel candidate than merely forcing still lower register limits.

PTXAS output: original 255 registers, 20-byte spill stores/24-byte loads; three-block target 168 registers, 136-byte spill stores/132-byte loads; four-block target 128 registers, 492-byte spill stores/420-byte loads. The aggressive 128-register setting creates more spills and is slower. Static spill-byte counts are compiler resource reports, not total runtime traffic.

## Encoder and extraction results

| Tokens | Padding | Original ms | 168-register ms | 128-register ms | 168-register speedup |
|---|---|---:|---:|---:|---:|
| 256 | False | 8.481 | 8.477 | 8.494 | 1.001x |
| 256 | True | 8.519 | 8.523 | 8.489 | 0.999x |
| 512 | False | 18.891 | 17.727 | 19.550 | 1.066x |
| 512 | True | 10.766 | 10.636 | 11.532 | 1.012x |
| 1024 | False | 51.134 | 48.110 | 53.676 | 1.063x |
| 1024 | True | 26.028 | 25.300 | 27.748 | 1.029x |

Full extraction, paired block: original 81.28 chunks/s; 168-register candidate 82.12 chunks/s (1.04% gain); 128-register candidate 80.67 chunks/s.

Absolute rates drift between phases: the original custom path measured 88.89 chunks/s earlier in this session, and 81.28 in the later register comparison. Use the paired within-block comparisons. The approximately 1% whole-extraction gain is small and is not a strong standalone deployment reason without a fresh-process confirmation. The repeated encoder improvement and hardware occupancy improvement are clearer.

All six tested encoder shapes remained bit-identical to the original custom path. All 290 chunks retained the same returned values and spans (confidence/score fields excluded from structural comparison). This does not resolve previously measured custom-versus-Triton or multilingual differences.

## Why whole extraction gains less

On a representative 97-chunk subset, the custom path used approximately 109.75 ms in attention kernels, 137.46 ms in other matrix-multiplication kernels, and 352.87 ms total GPU kernel time. Profiled extraction wall time was 1.477 s. Profiling changes timing; these numbers are diagnostic rather than an exact production CPU/GPU utilization split. Attention is a small part of the complete critical path.

CPU profiling of the same subset: word splitter 48,761 calls and 0.305 s self time; input transformation 0.757 s cumulative; count/span-logit processing 0.447 s cumulative. Cumulative times overlap, and Torch calls can include launch or GPU wait time, so do not add these times or interpret them all as pure CPU computation.

The earlier isolation benchmark deliberately excluded the fast-count, word-split cache, skipped repeated eval, and multi-stream serving optimizations already developed for T4. Those existing changes should be evaluated together on A100 before expecting a large whole-service gain from further attention work.

## Next work justified by evidence

1. Kernel: reduce live prefetch state and spills; test global-to-shared asynchronous copy on Ampere+ while retaining the current T4 implementation. Shared-memory carveout and tile shape must be considered with register limits; the current launch is also limited to three blocks by shared memory.
2. Whole extraction: profile the already optimized serving path on A100. Reuse the existing fast-count and word-split/eval changes rather than reimplementing them. Validate occurrence accuracy and corpus generality.
3. Treat the 168-register patch as an A100 experiment, not a global default. T4/L4 and broader multilingual/occurrence validation remain required.

NVIDIA documents the register and occupancy limits and that Ampere asynchronous global-to-shared copies avoid intermediate copy registers: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html . The proposed async-copy implementation has not been written or benchmarked.

## Artifacts and reproduction

- `a100_candidate.patch`: the exact one-line launch-bounds change; not applied to original source.
- `profile-export.zip`: CPU summaries, GPU-family totals, and Chrome traces.
- `register-export.zip`: variant sources, build logs, timings, and raw detections.
- `ncu-export.zip`: Nsight Compute reports and complete raw metric tables.
- Start from the parent `bootstrap.py` on an A100, execute `profile_tune.py`, then `register_tune.py`, then `ncu_check.py` and `ncu_summary.py` in the same Colab kernel. Download exports and stop the session.

## Cleanup

The A100 profiling session was stopped after all three export archives were downloaded and verified.
