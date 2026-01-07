import numpy as np
from pathlib import Path

import tifffile
import torch

def load_image(path: str, device: torch.device) -> torch.Tensor:
    p = Path(path)
    img = tifffile.imread(str(p))

    # Reduce to a single channel while preserving spatial resolution.
    if img.ndim == 3:
        # Heuristic:
        # - if first dim is small (<=4) and last dim is large, treat as (C,H,W)
        # - if last dim is small (<=4) and first dim is large, treat as (H,W,C)
        # - otherwise, fall back to taking the first slice along the last axis.
        if img.shape[0] <= 4 and img.shape[-1] > 4:
            # (C,H,W) -> take first channel -> (H,W)
            img = img[0]
        elif img.shape[-1] <= 4 and img.shape[0] > 4:
            # (H,W,C) -> take first channel -> (H,W)
            img = img[..., 0]
        else:
            img = img[..., 0]
    elif img.ndim != 2:
        raise ValueError(f"Unsupported image ndim={img.ndim} for {path}")

    img = torch.from_numpy(img.astype("float32"))
    max_val = float(img.max())
    if max_val > 0.0:
        img = img / max_val
    img = img.unsqueeze(0).unsqueeze(0)
    return img.to(device)


def load_tiff_layer(path: str, device: torch.device, layer: int | None = None) -> torch.Tensor:
    """
    Load a single layer from a (possibly multi-layer) TIFF as (1,1,H,W) in [0,1].

    For intensity handling we mirror the training dataset:
    - if uint16: downscale to uint8 via division by 257, then normalize to [0,1]
    - if uint8: normalize to [0,1].
    """
    p = Path(path)
    with tifffile.TiffFile(str(p)) as tif:
        series = tif.series[0]
        shape = series.shape
        if len(shape) == 2:
            img = series.asarray()
        elif len(shape) == 3:
            idx = 0 if layer is None else int(layer)
            img = series.asarray(key=idx)
        else:
            raise ValueError(f"Unsupported TIFF shape {shape} for {path}")

    if img.dtype == np.uint16:
        img = (img // 257).astype(np.uint8)

    img = torch.from_numpy(img.astype("float32"))
    max_val = float(img.max())
    if max_val > 0.0:
        img = img / max_val
    img = img.unsqueeze(0).unsqueeze(0)
    return img.to(device)
