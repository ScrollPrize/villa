"""Optional build-tree override for the volume-cartographer bindings.

Set ``VC_PYTHON_BUILD_DIR`` to a volume-cartographer CMake build tree that was
configured with ``VC_BUILD_PYTHON=ON`` to test freshly built ``vc`` modules
without reinstalling the editable package. The scikit-build-core editable
finder always redirects ``vc`` to the installed copy, so it is removed from
``sys.meta_path`` while the override is active. Mirrors
``volume-cartographer/python/tests/conftest.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_build_dir = os.environ.get("VC_PYTHON_BUILD_DIR")
if _build_dir:
    _python_dir = Path(_build_dir) / "python"
    if not (_python_dir / "vc" / "__init__.py").exists():
        raise RuntimeError(f"VC_PYTHON_BUILD_DIR={_build_dir} has no python/vc package")
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if "ScikitBuildRedirectingFinder" not in type(finder).__name__
    ]
    for name in [m for m in sys.modules if m == "vc" or m.startswith("vc.")]:
        del sys.modules[name]
    sys.path.insert(0, str(_python_dir))
