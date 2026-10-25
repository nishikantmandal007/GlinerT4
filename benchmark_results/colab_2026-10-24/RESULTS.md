# GLiNER2 custom CUDA kernel versus Triton: Colab, 2026-10-24

Our original T4 v4 kernel source also runs after recompilation for L4 (SM89) and A100 (SM80). It improved extraction throughput by 12.9% on L4 and 4.8% on A100 in this workload. This is an experimental portability/performance result, not a production enablement change.

## Extraction throughput

| GPU | Triton chunks/s | Custom chunks/s | Gain | Triton input tokens/s | Custom input tokens/s |
|---|---:|---:|---:|---:|---:|
| NVIDIA L4 | 62.49 | 70.56 | 12.91% | 16,599 | 18,742 |
| NVIDIA A100-SXM4-40GB | 84.63 | 88.69 | 4.80% | 22,478 | 23,558 |

Three alternating A/B rounds per GPU; median full grouped extraction time. Each round processes all 290 chunks through three unchanged label passes, batch 16. Input token counts exclude repeated schema prompts and the additional model passes. Chunks/min and input TPM are in throughput.csv. This is model extraction in one process and one serving thread, not HTTP RPM or document-processing throughput.

## Encoder timings

Batch 16. Padded rows retain between approximately 25% and 100% of tokens. Native packed="auto" policy is used on both paths. Seven alternating rounds per shape, five forward passes per timing; medians reported.

| GPU | Tokens | Right padding | Triton ms | Custom ms | Speedup |
|---|---:|---|---:|---:|---:|
| NVIDIA L4 | 256 | False | 28.03 | 23.60 | 1.188x |
| NVIDIA L4 | 256 | True | 27.55 | 20.46 | 1.347x |
| NVIDIA L4 | 384 | False | 44.79 | 38.27 | 1.170x |
| NVIDIA L4 | 384 | True | 29.42 | 23.21 | 1.267x |
| NVIDIA L4 | 512 | False | 65.66 | 55.19 | 1.190x |
| NVIDIA L4 | 512 | True | 41.27 | 32.79 | 1.258x |
| NVIDIA L4 | 1024 | False | 155.60 | 139.71 | 1.114x |
| NVIDIA L4 | 1024 | True | 98.93 | 77.27 | 1.280x |
| NVIDIA A100-SXM4-40GB | 256 | False | 9.52 | 7.58 | 1.255x |
| NVIDIA A100-SXM4-40GB | 256 | True | 9.57 | 6.63 | 1.443x |
| NVIDIA A100-SXM4-40GB | 384 | False | 13.60 | 12.77 | 1.064x |
| NVIDIA A100-SXM4-40GB | 384 | True | 10.82 | 7.90 | 1.370x |
| NVIDIA A100-SXM4-40GB | 512 | False | 19.24 | 18.77 | 1.025x |
| NVIDIA A100-SXM4-40GB | 512 | True | 12.81 | 10.62 | 1.207x |
| NVIDIA A100-SXM4-40GB | 1024 | False | 50.49 | 51.05 | 0.989x |
| NVIDIA A100-SXM4-40GB | 1024 | True | 30.54 | 25.74 | 1.186x |

A100 at 1024 tokens without padding is about 1% slower with our kernel; performance is shape-dependent. No CUDA tile tuning was performed for these newer GPUs.

## Accuracy and actual dispatch

Profiler traces confirmed `_deberta_attention_forward_kernel` for native Triton and `disentangled_attention_v4_kernel<4>` for custom CUDA on both GPUs. Independent call checks verified all 12 attention layers use the custom implementation for every tested encoder shape.

- NVIDIA L4: exact entity-value sets match on 288/290 chunks. The strict requirement that BOTH mean and maximum encoder error relative to FP32 never exceed native Triton is satisfied on 2/8 shapes, so that strict numerical gate does not pass across the full shape set.
- NVIDIA A100-SXM4-40GB: exact entity-value sets match on 286/290 chunks. The strict requirement that BOTH mean and maximum encoder error relative to FP32 never exceed native Triton is satisfied on 1/8 shapes, so that strict numerical gate does not pass across the full shape set.

L4 differences: two street values include/exclude house numbers. A100 differences: two house-number span changes, one added street, and one removed street occurrence. Per-document unique-value recall is unchanged for all four labels on both GPUs; this can hide an occurrence change when a value is still found elsewhere in the same document. Detailed differences and raw span/confidence outputs are retained in result JSONs and export ZIPs.

Ground-truth scoring uses Unicode/case/space normalization; Street scoring ignores house numbers and allows containment. These are raw GLiNER2 grouped outputs without the ai-utils span resolver; these metrics are not strict occurrence/span F1 and should not be compared directly with the earlier serving accuracy report.

| GPU | Label | Triton recall | Custom recall | Triton precision | Custom precision |
|---|---|---:|---:|---:|---:|
| NVIDIA L4 | Person | 0.9928 | 0.9928 | 0.9597 | 0.9597 |
| NVIDIA L4 | Organization | 0.9767 | 0.9767 | 0.8456 | 0.8456 |
| NVIDIA L4 | Street | 0.8700 | 0.8700 | 0.9933 | 0.9933 |
| NVIDIA L4 | City | 1.0000 | 1.0000 | 0.9040 | 0.9040 |
| NVIDIA A100-SXM4-40GB | Person | 0.9928 | 0.9928 | 0.9597 | 0.9597 |
| NVIDIA A100-SXM4-40GB | Organization | 0.9767 | 0.9767 | 0.8456 | 0.8456 |
| NVIDIA A100-SXM4-40GB | Street | 0.8700 | 0.8700 | 0.9933 | 0.9933 |
| NVIDIA A100-SXM4-40GB | City | 1.0000 | 1.0000 | 0.9040 | 0.9040 |

## Reproduction and limitations

- GLiNER2 2.1.0; DisentangledFlash 1.1.1; PyTorch 2.11.0+cu130; Triton 3.6.0; Transformers 5.20.2; Python 3.13 on Colab; four Torch CPU threads.
- Model fastino/gliner2-multi-v1, FP16, checkpoint snapshot ce747d79a8e362d3dee0b0b26d1201f7f1a8615a. Same fused model instance toggled between native and custom attention; standalone FP32 model provides encoder reference.
- Original v4 CUDA source SHA256: b594066b4995835fd5dead31b1f419305652d224bec4146700ebbd3044ab3035. The CUDA algorithm is unchanged. Experimental CMake build targets the actual GPU architecture; the binding only registers v4, and the loader uses the resulting prebuilt library.
- Corpus: 290 approximately 1KB chunks from ten synthetic realistic TXT documents, 40 sections each; 77,027 input tokens. Original data is embedded in payload.json. No PDF extraction, anonymization writer, ai-safe endpoint, Whisper/Flair/T5, multi-stream queue, fast-count optimization, or result cache is involved.
- Downloads, compilation, and warmup are excluded from timed runs. Three extraction samples and seven encoder samples do not constitute a long-running load test. Colab CPUs differ, so use within-GPU A/B ratios rather than attributing absolute cross-GPU extraction rates solely to GPU hardware.
- Saved scripts: bootstrap.py, benchmark.py, verify_export.py. bootstrap.py embeds the sources and synthetic corpus. colab_dns.py is a local CLI-only DNS workaround; it leaves system DNS and TLS verification unchanged.
- No production files, branch settings, original kernel sources, or VM services were changed. The existing ai-utils T4 dispatch guard is not expanded by this experiment.
- Broader multilingual/occurrence accuracy, BF16, other shapes/models, tuned Triton profiles, other GPUs, and production deployment remain untested.

Upstream sources checked: [GLiNER2 optional backend](https://github.com/fastino-ai/GLiNER2#disentangledflash), [merged integration PR #182](https://github.com/fastino-ai/GLiNER2/pull/182), [DisentangledFlash](https://github.com/delyan-boychev/disentangled-flash).

## Cleanup

Both experiment sessions were stopped. Final Colab usage check reported zero active assignments and 0.00 units/hour; balance decreased from 180.33 to 179.19 units during the experiments.
