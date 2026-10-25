# Build gliner_t4.so ahead of time (needs nvcc and the same torch as the runtime) and place it next to the loader,
# for images without a compiler. Run it in a CUDA devel builder stage, then copy torch-ext/gliner_t4/gliner_t4.so.
#   python build.py
import shutil
import sys
import tempfile
from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "torch-ext" / "gliner_t4"))
from _build import CUDA_FLAGS, SOURCES  # noqa: E402  (settings only; importing the package would load the op)

target = ROOT / "torch-ext" / "gliner_t4" / "gliner_t4.so"

with tempfile.TemporaryDirectory() as build_dir:
    load(name="gliner_t4", sources=SOURCES, extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"], is_python_module=False,
         build_directory=build_dir)
    shutil.copy(Path(build_dir) / "gliner_t4.so", target)
print(f"built {target}")
