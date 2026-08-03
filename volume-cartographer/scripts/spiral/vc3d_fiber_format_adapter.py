from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _candidate_module_roots() -> list[Path]:
    here = Path(__file__).resolve().parent
    roots = [here]
    for parent in here.parents:
        roots.append(parent / "vesuvius" / "src")
    return roots


def _import_vc3d_fiber_format():
    try:
        return importlib.import_module("vc3d_fiber_format")
    except ModuleNotFoundError:
        pass

    attempted: list[str] = []
    for root in _candidate_module_roots():
        attempted.append(str(root))
        if not (root / "vc3d_fiber_format" / "__init__.py").is_file():
            continue
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        return importlib.import_module("vc3d_fiber_format")
    searched = "\n  ".join(attempted)
    raise ModuleNotFoundError(
        "No module named 'vc3d_fiber_format'. Searched:\n  " + searched)


parse_vc3d_fiber_format = _import_vc3d_fiber_format().parse_vc3d_fiber_format

__all__ = ["parse_vc3d_fiber_format"]
