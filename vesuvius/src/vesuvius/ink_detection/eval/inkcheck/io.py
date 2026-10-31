"""Loading 2D ink maps and masks from TIFF / PNG / JPEG / NPY / zarr-free inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # published ds8 previews are large but trusted local files


def load_map(path: str | Path) -> np.ndarray:
    """Load a 2D map as float32. Multi-channel images are averaged; 3D stacks
    with a leading singleton axis are squeezed."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".npy":
        a = np.load(p)
    elif suf in (".tif", ".tiff"):
        import tifffile

        a = tifffile.imread(p)
    else:
        a = np.asarray(Image.open(p))
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.ndim == 3:  # HxWxC image
        if a.shape[-1] in (3, 4):
            a = a[..., :3].mean(-1)
        else:
            raise ValueError(f"{p}: expected a 2D map, got shape {a.shape}")
    if a.ndim != 2:
        raise ValueError(f"{p}: expected a 2D map, got shape {a.shape}")
    return a.astype(np.float32)


def load_mask(path: str | Path) -> np.ndarray:
    """Load a mask (any nonzero pixel = valid)."""
    return load_map(path) > 0
