"""Isolated native-binding selection for the cooperative test launcher only.

An installed scikit-build editable finder precedes PYTHONPATH, so simply adding
the build directory does not select its extension. Never mutate the installed
finder or venv; override only vc and vc.volume in these test child processes.
"""
import importlib.machinery
import os
from pathlib import Path
import sys


class NativeVolumeFinder:
    def __init__(self, root):
        self.root = Path(root)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ("vc", "vc.volume"):
            return None
        directory = self.root if fullname == "vc" else self.root / "vc"
        spec = importlib.machinery.PathFinder.find_spec(fullname, [str(directory)])
        if spec is None:
            raise ImportError(f"requested native test binding missing: {fullname} in {directory}")
        return spec


if os.environ.get("FIBER_COOPERATIVE_NATIVE_BUILD"):
    sys.meta_path.insert(0, NativeVolumeFinder(os.environ["FIBER_COOPERATIVE_NATIVE_BUILD"]))
