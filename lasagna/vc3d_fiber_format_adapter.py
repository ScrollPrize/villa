from __future__ import annotations

import sys
from pathlib import Path


_VESUVIUS_SRC = Path(__file__).resolve().parents[1] / "vesuvius" / "src"
if str(_VESUVIUS_SRC) not in sys.path:
	sys.path.insert(0, str(_VESUVIUS_SRC))

from vc3d_fiber_format import parse_vc3d_fiber_format  # noqa: E402

__all__ = ["parse_vc3d_fiber_format"]
