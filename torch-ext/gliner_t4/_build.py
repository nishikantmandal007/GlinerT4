# sources and flags shared by the JIT loader and the ahead-of-time build (build.py)
from pathlib import Path

CSRC = Path(__file__).resolve().parents[2] / "csrc"
SOURCES = [str(CSRC / name) for name in ("torch_binding.cpp", "disentangled_attention.cu", "disentangled_attention_v2.cu",
                                         "disentangled_attention_v3.cu", "disentangled_attention_v4.cu",
                                         "add_layernorm.cu")]
CUDA_FLAGS = ["-O3", "-gencode=arch=compute_75,code=sm_75", "--use_fast_math", "-lineinfo"]
