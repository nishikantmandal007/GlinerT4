# GLiNER2 T4 custom attention kernel

Experimental CUDA implementation of DeBERTa disentangled attention for GLiNER2's mDeBERTa encoder, with the original v4 kernel, earlier kernel iterations, tests, benchmark harnesses, and Colab evidence.

## Current finding

The single-T4 GLiNER2 2.1.0 path measured 54.81 chunks/s on a 290-chunk synthetic corpus. This is well below the 100 chunks/s goal. Two T4-compatible changes reduced register use and preserved exact outputs, but did not improve throughput. The new candidates are retained as experiments and are not enabled by the original loader.

See [the T4 profile and optimization report](benchmark_results/colab_2026-10-24/t4_optimization/RESULTS.md) and the [plan assessment](benchmark_results/colab_2026-10-24/t4_optimization/PLAN_ASSESSMENT.md).

## Important benchmark context

The earlier approximately 100 chunks/s serving result was the combined throughput of **two T4 GPUs**, with two streams per GPU and fast-count enabled. The older 114–120 chunks/s direct result also summed two T4s. Those numbers are not one-T4 capacity results and are not directly comparable to the strict single-T4 GLiNER2 2.1.0 run.

The 2026-10-24 Colab artifacts include the A100/L4 comparison, exact-output optimization tests, profiler data, and the original experiment inputs. The later T4 report records both successful checks and the Colab session that expired during a repeat.

## Layout

- `csrc/`, `torch-ext/`: CUDA sources and experimental GLiNER2 attention adapter.
- `tests/`, `bench/`: existing correctness tests and benchmark/profiling scripts.
- `artifacts/`: earlier source references, Colab runs, and serving-path experiments.
- `benchmark_results/colab_2026-10-24/`: reproducible A100/L4 results and T4 optimization experiments.
- `kernel_log.xlsx`: historical kernel iteration notes and timings.

The work targets `fastino/gliner2-multi-v1` through `gliner2==2.1.0`. The supplied plan document names the separate `urchade/gliner_multi-v2.1` checkpoint; the existing adapter is not a drop-in integration for that runtime.

## Development

The original kernel folder's `AGENTS.md` records the VM build environment and accuracy gates. For the isolated Colab runs, use the saved `bootstrap.py` and benchmark scripts under `benchmark_results/colab_2026-10-24/`. Keep the original v4 source as the T4 control; the `async_simple` experiment under `exact_optimization/` requires SM80+ and is not a T4 kernel.

All results here are experimental. They do not establish zero accuracy loss for every input or a production throughput SLA.
