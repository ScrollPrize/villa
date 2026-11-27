"""Which end of a layer window is air, from texture alone.

Used by ``FORCE_REVERSE=auto``. Exposed fragment surfaces are often stored with the air side first
(low layer numbers), the opposite of the order the 3D-decoder checkpoints read best. The rule follows
AtomCrtr's in villa #2627: in a few fixed patches inside the data, the air end of the window has much
less texture than the papyrus end. It only decides when the two ends differ clearly. The rule in #2627
used 8 patches; on real fragment stacks 8 random patches can point either way, so 256 are used here.
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def end_texture(source, n_end: int = 13, patch: int = 64, n_patches: int = 256, seed: int = 0,
                max_tries: int = 20000) -> Optional[Tuple[float, float]]:
    """Mean texture of the first and the last ``n_end`` layers of ``source``.

    ``source`` has ``shape`` (H, W, C) and ``read_roi(y1, y2, x1, x2) -> (h, w, C)`` (an
    ``inference.LayersSource``). Texture of a layer = median over ``n_patches`` patches of the
    standard deviation of that layer in the patch. Patches are drawn from a fixed seed and kept only
    where every pixel has data in some layer. Returns None if too few patches have data.
    """
    H, W, C = source.shape
    if C < 2 * n_end or H < patch or W < patch:
        return None
    rng = np.random.default_rng(seed)
    stds = []
    for _ in range(max_tries):
        y = int(rng.integers(0, H - patch + 1))
        x = int(rng.integers(0, W - patch + 1))
        roi = np.asarray(source.read_roi(y, y + patch, x, x + patch))
        if not np.any(roi != 0, axis=-1).all():
            continue
        stds.append(roi.reshape(-1, C).astype(np.float32).std(axis=0))
        if len(stds) == n_patches:
            break
    if len(stds) < n_patches:
        return None
    per_layer = np.median(np.stack(stds), axis=0)
    return float(per_layer[:n_end].mean()), float(per_layer[-n_end:].mean())


def air_end(source, ratio: float = 2.0, **kw) -> Optional[str]:
    """``"low"`` or ``"high"``: the end of the window that looks like air; None when unclear."""
    t = end_texture(source, **kw)
    if t is None:
        logger.warning("Orientation check: not enough patches with data; keeping the stored layer order")
        return None
    low, high = t
    logger.info(f"Orientation check: texture first layers {low:.2f}, last layers {high:.2f}")
    if max(low, high) < ratio * max(min(low, high), 1e-6):
        logger.warning(f"Orientation check: the two ends differ by less than {ratio:g}x; keeping the stored layer order")
        return None
    return "low" if low < high else "high"
