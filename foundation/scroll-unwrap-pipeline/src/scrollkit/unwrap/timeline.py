"""Animation timeline: frame index → interpolation parameter t with eased morph and holds."""

from __future__ import annotations

import numpy as np


def quintic(s: np.ndarray) -> np.ndarray:
    return s * s * s * (s * (6.0 * s - 15.0) + 10.0)


def timeline_t(n_frames: int = 240, hold_start: int = 20, hold_end: int = 25, easing: str = "quintic") -> np.ndarray:
    """t per frame: 0 during the start hold, eased 0→1, 1 during the end hold."""
    morph = n_frames - hold_start - hold_end
    if morph < 2:
        raise ValueError("timeline too short for the holds")
    s = np.linspace(0.0, 1.0, morph)
    e = quintic(s) if easing == "quintic" else s
    return np.concatenate([np.zeros(hold_start), e, np.ones(hold_end)])
