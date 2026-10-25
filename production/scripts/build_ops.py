import shutil
import tempfile
from pathlib import Path

from glinert4.ops import PREBUILT, build

with tempfile.TemporaryDirectory() as build_directory:
    build(build_directory=build_directory, verbose=True)
    shutil.copy(Path(build_directory) / "glinert4.so", PREBUILT)
print(PREBUILT)
