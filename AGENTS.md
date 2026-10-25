# Custom CUDA kernel for GLiNER2's mDeBERTa encoder on Tesla T4

## Goal
Make GLiNER2's encoder (`fastino/gliner2-multi-v1`, mDeBERTa-v3-base) faster on a Tesla T4 than the current
best path, the Triton kernel `disentangled_flash` (fused disentangled attention), with no accuracy loss.
Report honestly if it cannot be beaten. A negative result with measurements is an acceptable outcome.

Use the `cuda-kernels` skill (HuggingFace kernels, `~/.codex/skills/cuda-kernels`), especially
`references/t4-optimization-guide.md` and `references/kernel-templates.md`.

## Where things run
- Edit code ONLY in this local folder: `/Users/nishikantmandal007/MDP/Maya_Work/gliner-t4-kernel`.
- Build and run on the GPU VM over ssh. Sync first, every time:
  `rsync -az --delete --exclude .git ./ aitesting:nishikant/gliner-t4-kernel/`
  then `ssh aitesting 'cd ~/nishikant/gliner-t4-kernel && ...'`.
- Python on the VM: `~/nishikant/ai-utils/.venv-t32/bin/python` (torch 2.6.0+cu124, Triton 3.2.0,
  transformers 5.19, gliner2 2.1.0, disentangled-flash 1.1.1). Set `HF_HUB_OFFLINE=1`. nvcc is `/usr/bin/nvcc` (CUDA 12.0).
  No pip in that venv and do NOT install anything into it; if you need a package, use
  `~/.local/bin/uv pip install --python <venv python> --target ~/nishikant/gliner-t4-kernel/.deps --no-deps <pkg>` and PYTHONPATH.
- Do not touch `~/soumik/` or `~/nishikant/ai-utils/` files on the VM (read-only reference). Write only under
  `~/nishikant/gliner-t4-kernel/` on the VM.
- The VM is shared: 2x T4 (cuda:0, cuda:1). Before a GPU run, check `nvidia-smi`; if someone else is using a GPU,
  use the other one or wait. Never kill processes you did not start. Use one GPU at a time for benchmarks.
- Nix / kernel-builder are not available on the VM. Keep the kernel sources in the skill's layout where it is cheap
  (`csrc/`, `torch-ext/`), but build for experiments with `torch.utils.cpp_extension.load` (dev only; Hub packaging is out
  of scope for this task). Compile for `-gencode arch=compute_75,code=sm_75`.

## Model facts
hidden 768, 12 layers, 12 heads, head dim 64, intermediate 3072, GELU, LayerNorm, relative attention with
`pos_att_type: [p2c, c2p]`, `position_buckets: 256`, `max_relative_positions: -1` (log-bucketed relative positions),
`share_att_key` per config. Config: `~/.cache/huggingface/hub/models--fastino--gliner2-multi-v1/snapshots/*/encoder_config/config.json`.
Reference implementations on the VM: transformers `models/deberta_v2/modeling_deberta_v2.py` (DisentangledSelfAttention),
and `disentangled_flash/` (kernel.py, deberta.py, _reference.py) in the venv's site-packages.
Workload: batches of 16 texts, sequence lengths 96..1024 tokens (median 480, usually padded to a multiple of 32), fp16
activations/weights, fp32 accumulation. Inference only (no backward).

## Baseline (measured on this T4, encoder only, `bench/encoder_bench.py`)
| 16 x tokens | disentangled_flash | plain fp16 PyTorch | TensorRT 11.4 (no DeBERTa plugin) |
|---|---|---|---|
| 256 | 64.0 ms | 107.4 ms | 102.7 ms |
| 384 | 103.0 ms | 199.6 ms | 185.4 ms |
| 512 | 162.6 ms | 334.4 ms | 284.2 ms |
Earlier profile with disentangled_flash: kernel time = fused attention 44%, GEMM (cuBLAS, tensor cores) 43%, elementwise 10%.
So the realistic targets are (1) the attention kernel itself and (2) fusing the elementwise work (bias, residual, LayerNorm, GELU)
into neighbouring kernels. cuBLAS GEMMs are close to optimal already; do not rewrite them unless you can measure a win.
T4 facts that matter: sm_75, 320 GB/s, 40 SMs, 64 KB smem/SM, fp16 tensor cores via mma.sync m16n8k8 / WMMA, no bf16,
no cp.async (that is sm_80+), Triton >= 3.3 loses tensor cores on sm_75 (so stay with CUDA C++ for the kernel).

## Accuracy gate (must pass, otherwise the kernel does not count)
1. Encoder output vs an fp32 PyTorch reference on real token batches: mean and max |diff| no worse than
   disentangled_flash's own distance to the fp32 reference (measure both, report both).
2. End to end: replace the encoder in GLiNER2 and run detection on the corpus `~/nishikant/detbench/corpus.json`
   (see `~/nishikant/ai-utils/scripts/bench_detection.py` for how texts, labels and scoring work; import from
   `~/nishikant/ai-utils` read-only via sys.path, monkeypatch in your own harness, do not edit that repo). Ground-truth
   recall/precision per label must equal the disentangled_flash run (`~/nishikant/detbench/bucket_gpu0.json`:
   Person .994/.961, Organization .977/.846, Street .861/1.0, City 1.0/.93), and report per-chunk identical rate vs that file.

## Deliverables (write to this folder)
- `RESULTS.md`: what you built, encoder ms at 16x256/384/512 vs the baseline table, end-to-end chunks/s on one T4 vs 31.3
  (bucketed disentangled_flash, `queue` mode, one GPU), the accuracy gate numbers, what limited further speed (with profiler
  evidence: nsys/ncu or torch.profiler), and your estimate of what the same kernel would give on an A10 (sm_86, 600 GB/s) and why.
- Kernel sources, the test/bench scripts, and exact commands to reproduce.
Work in steps: profile first, pick the biggest target, build, verify correctness, measure, iterate. Stop when further work
is not paying off and write RESULTS.md.
