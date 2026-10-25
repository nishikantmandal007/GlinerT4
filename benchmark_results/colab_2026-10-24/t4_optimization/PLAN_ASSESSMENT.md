# Assessment of `GLiNER_Multilingual_v2_1_T4_Optimization_Plan.docx`

The plan is technically reasonable as a profiling roadmap, but its named checkpoint and the tested checkpoint are different.

The document targets `urchade/gliner_multi-v2.1` through the `gliner` package. The GPU experiment requested afterward uses `gliner2==2.1.0` and `fastino/gliner2-multi-v1`. These GLiNER runtimes do not share the same integration API. Both checkpoints use the mDeBERTa-v3 family, and the T4 kernel's 12-head, head-dimension-64 attention math is relevant, so an adapter at the encoder attention boundary is feasible. It still needs its own implementation and accuracy validation; the current GLiNER2 monkeypatch cannot be copied unchanged.

The plan's package version and model version are distinct: the latest `gliner` library on PyPI at the time of this check is 0.2.29, while `gliner_multi-v2.1` is a model checkpoint identifier. `gliner2==2.1.0` is a separate package/version chosen for the actual T4 throughput comparison.

## Feasibility by plan section

| Proposal | Assessment for one T4 |
|---|---|
| Fixed corpus, warm-up, repeats, GPU-only and full-path timings | Feasible. A one-GPU, 290-chunk, three-pass baseline was reproduced at 54.81 chunks/s. The current corpus is synthetic and English-oriented, so it does not satisfy the plan's multilingual production-corpus goal. |
| Batch-size sweep and length bucketing | Feasible without truncation. Must preserve masks and exact full outputs. Not completed in this run. |
| Buffer reuse and fewer host copies/synchronizations | Feasible after profiling. Existing work already has word-split caching and exact bulk token lookup. The profile shows CPU span unpacking/collation work worth prioritizing. |
| CPU/GPU overlap and multiple streams | Feasible, but each GLiNER2 inference worker needs independent model execution state. Needs a matched load test and VRAM tracking. Not completed in this run. |
| FlashDeBERTa | Worth testing separately on the target model. Upstream documents `transformers>=4.51.3`; its published validation GPUs begin at SM86, so T4/SM75 compatibility and speed must be verified locally rather than assumed. |
| Custom relative-position attention on T4 | Feasible and already integrated for the GLiNER2 path. Two register-prefetch-removal variants passed exact gates but were neutral/slower. Attention was about 25.5% of profiled CUDA time, so other pipeline work matters for a 1.82x target. |
| Span-scoring kernel or fusion | Feasible only after isolating it. The profiler shows substantial CPU-side unpack and count/preparation work; a CUDA kernel is not automatically the right fix. Preserve label decisions and output scores exactly. |
| FP16 or INT8 changes | FP16 is already used in the GLiNER2 T4 run. Do not change precision or quantize under a zero-accuracy-loss requirement without separate exact output checks. INT8 is not included in the result. |
| CUDA Graphs, `torch.compile`, TensorRT | Possible exploratory work, but not the first target: verify operator support and complete output parity after the CPU/preparation and GPU profile is improved. |
| Gzip activation compression | Not recommended; the next model operations need decoded tensors and no measured traffic bottleneck supports adding compression. |

## Sources

- GLiNER installation and FlashDeBERTa instructions: https://github.com/urchade/GLiNER/blob/main/docs/instalation.md
- `urchade/gliner_multi-v2.1` model configuration: https://huggingface.co/urchade/gliner_multi-v2.1/blob/main/gliner_config.json
- DisentangledFlash supported/validated GPU notes: https://github.com/delyan-boychev/disentangled-flash
- PyPI `gliner` releases: https://pypi.org/project/gliner/
