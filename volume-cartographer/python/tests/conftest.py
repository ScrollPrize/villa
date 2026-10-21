"""Explicit build selection, including when an editable install redirects vc."""
import importlib.abc
import importlib.machinery
from pathlib import Path
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption("--vc-build-python", help="CMake build/python directory to test")
    parser.addoption("--vc-real-volume", help="Optional local volume for exact-path performance validation")
    parser.addoption("--vc-origin-xyz", default="0,0,0", help="Base XYZ origin for the real-volume crop")
    parser.addoption("--vc-level", type=int, default=0, help="Pyramid level for real-volume validation")


def pytest_configure(config):
    directory = config.getoption("--vc-build-python")
    if not directory:
        return
    root = Path(directory).resolve()
    if not (root / "vc" / "__init__.py").is_file():
        raise pytest.UsageError(f"No built vc package in {root}")
    if any(name == "vc" or name.startswith("vc.") for name in sys.modules):
        raise pytest.UsageError("vc was imported before selecting the test build")

    class BuildFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "vc":
                return importlib.machinery.PathFinder.find_spec(fullname, [str(root)])
            if fullname.startswith("vc."):
                return importlib.machinery.PathFinder.find_spec(fullname, path)
            return None

    finder = BuildFinder()
    sys.meta_path.insert(0, finder)
    config.add_cleanup(lambda: sys.meta_path.remove(finder))
