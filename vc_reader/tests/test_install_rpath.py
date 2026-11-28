"""The installed _reader module must find libvc_reader_native through its own rpath, wherever
CMAKE_INSTALL_LIBDIR put the library (lib64 on Red Hat and SUSE), without help from LD_LIBRARY_PATH.

Run against an installed (or scikit-build-core editable) build: python -m pytest tests/test_install_rpath.py
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


def rpaths(module):
    if sys.platform == "darwin":
        out = subprocess.run(["otool", "-l", str(module)], capture_output=True, text=True, check=True).stdout
        return re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path (\S+)", out)
    out = subprocess.run(["readelf", "-d", str(module)], capture_output=True, text=True, check=True).stdout
    found = re.findall(r"\((?:RPATH|RUNPATH)\)\s+Library r(?:un)?path: \[([^\]]*)\]", out)
    return [p for entry in found for p in entry.split(":") if p]


def test_reader_rpath_reaches_native_library():
    if sys.platform not in ("darwin", "linux"):
        pytest.skip("rpaths are a macOS and Linux concern")
    tool = "otool" if sys.platform == "darwin" else "readelf"
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} is not installed")
    import importlib.util
    spec = importlib.util.find_spec("vc_reader")
    package = Path(spec.submodule_search_locations[0])
    modules = sorted(package.glob("_reader.*"))
    if not modules:
        pytest.skip("built without the reader module")
    name = "libvc_reader_native.dylib" if sys.platform == "darwin" else "libvc_reader_native.so"
    origin = "@loader_path" if sys.platform == "darwin" else "$ORIGIN"
    dirs = [Path(p.replace(origin, str(modules[0].parent))) for p in rpaths(modules[0])]
    assert any((d / name).exists() for d in dirs), f"{name} is in none of the rpath directories {dirs}"
