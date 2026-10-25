from pathlib import Path

import torch
from torch.utils.cpp_extension import load

PACKAGE_DIR = Path(__file__).resolve().parent
SOURCES = [str(PACKAGE_DIR / "csrc" / name) for name in ("ops.cpp", "attention.cu", "add_layernorm.cu")]
CUDA_FLAGS = ["-O3", "-gencode=arch=compute_75,code=sm_75", "--use_fast_math"]
PREBUILT = PACKAGE_DIR / "_glinert4.so"

_loaded = False


def build(build_directory=None, verbose=False):
    load(name="glinert4", sources=SOURCES, extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"],
         is_python_module=False, build_directory=build_directory, verbose=verbose)


def ops():
    global _loaded
    if not _loaded:
        if PREBUILT.exists():
            torch.ops.load_library(str(PREBUILT))
        else:
            build()
        _loaded = True
    return torch.ops.glinert4
