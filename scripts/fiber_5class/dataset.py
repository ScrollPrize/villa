"""Random-crop dataset for self-distillation training (no ground-truth labels).

Each ``__getitem__`` returns a uniformly-random 256^3 crop of the source
volume (after rejection-sampling against mostly-air crops), plus the raw uint8
image needed by the label generator's dark-voxel guard.

Normalization: per-crop min-max scaled to [0, 1] after clipping to the
1st-99th percentile.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import s3fs
import torch
import zarr
from torch.utils.data import IterableDataset


def open_volume_zarr(
    *,
    volume_url: str,
    storage_options: Mapping | None = None,
    scale: int = 0,
):
    """Open a (possibly remote) zarr volume at the requested scale level."""
    storage_options = dict(storage_options or {"anon": True})
    if volume_url.startswith("s3://"):
        fs = s3fs.S3FileSystem(**storage_options)
        store = s3fs.S3Map(volume_url.replace("s3://", "", 1), s3=fs, check=False)
    else:
        store = volume_url
    root = zarr.open(store, mode="r")
    return root[str(scale)]


def percentile_minmax_normalize(crop: np.ndarray, *, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Clip to [p_lo, p_hi]-percentile of the crop and rescale to [0, 1]."""
    flat = crop.reshape(-1)
    lo, hi = np.percentile(flat, [p_lo, p_hi])
    out = np.clip(crop.astype(np.float32), lo, hi)
    span = max(hi - lo, 1e-6)
    return (out - lo) / span


def _crop_is_valid(raw_crop: np.ndarray, *, min_nonempty_frac: float, dark_threshold: int) -> bool:
    """Reject crops that are mostly air."""
    nonempty = (raw_crop > dark_threshold).mean()
    return float(nonempty) >= min_nonempty_frac


class RandomFiberCropDataset(IterableDataset):
    """Iterable dataset yielding random 256^3 crops with rejection sampling.

    Each yielded dict has::

        image           (1, 256, 256, 256) float32, normalized to [0, 1]
        raw_image       (1, 256, 256, 256) uint8,   raw (for dark-voxel guard)
        crop_origin     (3,)               int64,   ZYX corner in volume coords
    """

    def __init__(
        self,
        *,
        volume_url: str,
        crop_size: tuple[int, int, int] = (256, 256, 256),
        storage_options: Mapping | None = None,
        scale: int = 0,
        min_nonempty_frac: float = 0.10,
        dark_threshold: int = 50,
        max_rejection_retries: int = 32,
        seed: int | None = None,
    ):
        super().__init__()
        self.volume_url = volume_url
        self.crop_size = tuple(int(s) for s in crop_size)
        self.storage_options = dict(storage_options or {"anon": True})
        self.scale = int(scale)
        self.min_nonempty_frac = float(min_nonempty_frac)
        self.dark_threshold = int(dark_threshold)
        self.max_rejection_retries = int(max_rejection_retries)
        self.seed = seed
        self._volume = None
        self._volume_shape: tuple[int, int, int] | None = None

    def _ensure_volume(self):
        if self._volume is not None:
            return
        self._volume = open_volume_zarr(
            volume_url=self.volume_url,
            storage_options=self.storage_options,
            scale=self.scale,
        )
        self._volume_shape = tuple(self._volume.shape)

    def _sample_one(self, rng: np.random.Generator) -> dict:
        self._ensure_volume()
        Z, Y, X = self._volume_shape  # type: ignore[misc]
        cz, cy, cx = self.crop_size
        z0 = y0 = x0 = 0
        for _ in range(self.max_rejection_retries):
            z0 = int(rng.integers(0, Z - cz + 1))
            y0 = int(rng.integers(0, Y - cy + 1))
            x0 = int(rng.integers(0, X - cx + 1))
            raw = np.asarray(self._volume[z0:z0 + cz, y0:y0 + cy, x0:x0 + cx])
            if raw.dtype != np.uint8:
                raw = raw.astype(np.uint8)
            if _crop_is_valid(raw, min_nonempty_frac=self.min_nonempty_frac,
                              dark_threshold=self.dark_threshold):
                break
        else:
            # Last resort: keep the most recent crop even if mostly air (rare).
            pass
        norm = percentile_minmax_normalize(raw)
        return {
            "image": torch.from_numpy(norm[None].astype(np.float32, copy=False)),
            "raw_image": torch.from_numpy(raw[None]),
            "crop_origin": torch.tensor([z0, y0, x0], dtype=torch.int64),
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        base_seed = self.seed if self.seed is not None else 0
        rng = np.random.default_rng(base_seed * 1_000 + worker_id + 1)
        while True:
            yield self._sample_one(rng)


def collate_random_crops(batch: Sequence[dict]) -> dict:
    """Stack the per-sample tensors into batched tensors."""
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "raw_image": torch.stack([b["raw_image"] for b in batch], dim=0),
        "crop_origin": torch.stack([b["crop_origin"] for b in batch], dim=0),
    }
