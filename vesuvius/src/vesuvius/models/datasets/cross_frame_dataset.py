"""Dataset that samples patches from two zarr volumes in different frames.

The label volume is authored in a canonical "fixed" frame; the image volume
lives in its own "moving" frame. An affine transform in ``transform.json``
relates the two (``p_fixed = M @ p_moving`` in XYZ order). We enumerate
foreground patches natively in label-voxel coords, then resample the
matching image region through ``M^{-1}`` with trilinear interpolation.

All S3/HTTPS/local zarr I/O goes through :func:`vesuvius.data.utils.open_zarr`.
The patch index is built once per unique (labels URL, patch size, thresholds,
transform checksum) tuple and cached on disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from vesuvius.data import affine
from vesuvius.data.utils import open_zarr
from vesuvius.utils.utils import pad_or_crop_3d

from ..augmentation.pipelines import create_training_transforms
from ..training.normalization import get_normalization


logger = logging.getLogger(__name__)


def _resolve_zarr_array(obj) -> zarr.Array:
    """Return the underlying ``zarr.Array`` for a group-or-array object.

    ``open_zarr`` may return a Group when the URL points at an OME-Zarr
    container. Prefer the ``"0"`` subpath, then a handful of common fallbacks.
    """
    if hasattr(obj, "shape") and hasattr(obj, "dtype") and not hasattr(obj, "keys"):
        return obj
    for key in ("0", "data", "arr_0"):
        if key in obj:
            candidate = obj[key]
            if hasattr(candidate, "shape"):
                return candidate
    raise ValueError("Could not resolve a zarr Array inside the given store")


class CrossFrameZarrDataset(Dataset):
    """3D patch dataset with affine cross-frame image/label sampling."""

    def __init__(self, mgr, *, is_training: bool = True) -> None:
        super().__init__()
        self.mgr = mgr
        self.is_training = is_training

        ds_cfg = getattr(mgr, "dataset_config", {}) or {}
        self.image_zarr_url = _require_str(ds_cfg, "image_zarr_url")
        self.labels_zarr_url = _require_str(ds_cfg, "labels_zarr_url")
        self.transform_json_url = _require_str(ds_cfg, "transform_json_url")

        shared_storage = dict(ds_cfg.get("storage_options") or {})
        self.storage_options_image = dict(
            ds_cfg.get("storage_options_image") or shared_storage
        )
        self.storage_options_labels = dict(
            ds_cfg.get("storage_options_labels") or shared_storage
        )

        self.patch_size: Tuple[int, int, int] = tuple(int(v) for v in mgr.train_patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(
                f"CrossFrameZarrDataset requires a 3D patch size, got {self.patch_size}"
            )

        self.targets = getattr(mgr, "targets", {}) or {}
        self.target_names = [
            name for name, info in self.targets.items()
            if not info.get("auxiliary_task", False)
        ] or ["fibers"]
        if len(self.target_names) != 1:
            raise ValueError(
                "CrossFrameZarrDataset expects exactly one target "
                f"but got {self.target_names}"
            )
        self.target_name = self.target_names[0]

        raw_valid_patch_value = ds_cfg.get("valid_patch_value")
        self.valid_patch_value: Optional[float] = (
            None if raw_valid_patch_value is None else float(raw_valid_patch_value)
        )
        self.min_labeled_ratio = float(getattr(mgr, "min_labeled_ratio", 0.01))
        self.min_bbox_percent = float(getattr(mgr, "min_bbox_percent", 0.15))
        self.stride: Tuple[int, int, int] = tuple(
            int(v) for v in ds_cfg.get("patch_stride", self.patch_size)
        )

        self._image_array = _resolve_zarr_array(
            open_zarr(self.image_zarr_url, mode="r", storage_options=self.storage_options_image)
        )
        self._labels_array = _resolve_zarr_array(
            open_zarr(self.labels_zarr_url, mode="r", storage_options=self.storage_options_labels)
        )
        self._image_shape = tuple(int(v) for v in self._image_array.shape[-3:])
        self._labels_shape = tuple(int(v) for v in self._labels_array.shape[-3:])

        self._transform_doc = affine.read_transform_json(self.transform_json_url)
        self._matrix_xyz = self._transform_doc.matrix_xyz
        self._matrix_zyx_label_to_image = affine.label_to_image_zyx_matrix(
            self._matrix_xyz, invert=True
        )

        self.normalization_scheme = getattr(mgr, "normalization_scheme", "zscore")
        self.intensity_properties = getattr(mgr, "intensity_properties", None) or {}
        self.normalizer = get_normalization(self.normalization_scheme, self.intensity_properties)

        self.cache_dir = self._resolve_cache_dir(ds_cfg)
        self._patches: List[Tuple[int, int, int]] = []
        self._build_patch_index()

        self.transforms = None
        if self.is_training:
            self.transforms = create_training_transforms(
                patch_size=self.patch_size,
                no_spatial=bool(getattr(mgr, "no_spatial_augmentation", False)),
                no_scaling=bool(getattr(mgr, "no_scaling_augmentation", False)),
            )

        logger.info(
            "CrossFrameZarrDataset: image=%s (shape=%s) labels=%s (shape=%s) patches=%d",
            self.image_zarr_url, self._image_shape,
            self.labels_zarr_url, self._labels_shape,
            len(self._patches),
        )

    # -------------------------------------------------------------- cache --

    def _resolve_cache_dir(self, ds_cfg: dict) -> Optional[Path]:
        explicit = ds_cfg.get("cache_dir")
        if explicit:
            return Path(explicit).expanduser().resolve()
        data_path = getattr(self.mgr, "data_path", None)
        if data_path is not None:
            return Path(data_path) / ".cross_frame_cache"
        return None

    def _cache_key(self) -> str:
        payload = json.dumps(
            {
                "labels_url": self.labels_zarr_url,
                "image_url": self.image_zarr_url,
                "patch_size": list(self.patch_size),
                "stride": list(self.stride),
                "min_labeled_ratio": self.min_labeled_ratio,
                "min_bbox_percent": self.min_bbox_percent,
                "valid_patch_value": self.valid_patch_value,
                "transform_checksum": affine.matrix_checksum(self._matrix_xyz),
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"fibers_patches_{self._cache_key()}.npz"

    def _load_cache(self) -> Optional[np.ndarray]:
        cp = self._cache_path()
        if cp is None or not cp.exists():
            return None
        try:
            blob = np.load(cp, allow_pickle=False)
        except Exception:  # corrupted/partial cache
            logger.warning("Ignoring unreadable patch cache at %s", cp)
            return None
        positions = blob.get("positions") if hasattr(blob, "get") else blob["positions"]
        if positions is None or positions.ndim != 2 or positions.shape[1] != 3:
            return None
        return positions.astype(np.int64)

    def _save_cache(self, positions: np.ndarray) -> None:
        cp = self._cache_path()
        if cp is None:
            return
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cp,
            positions=np.asarray(positions, dtype=np.int64),
            patch_size=np.asarray(self.patch_size, dtype=np.int64),
        )

    # ---------------------------------------------------------- indexing --

    def _build_patch_index(self) -> None:
        cached = self._load_cache()
        if cached is not None and len(cached) > 0:
            self._patches = [tuple(int(v) for v in row) for row in cached]
            logger.info("Loaded %d patches from cache", len(self._patches))
            return

        positions = self._scan_foreground_positions()
        if not positions:
            raise RuntimeError(
                "CrossFrameZarrDataset found 0 foreground patches. "
                "Check labels volume, patch size, and thresholds."
            )
        self._patches = positions
        self._save_cache(np.asarray(positions, dtype=np.int64))

    def _scan_foreground_positions(self) -> List[Tuple[int, int, int]]:
        dz, dy, dx = self.patch_size
        sz, sy, sx = self.stride
        lz, ly, lx = self._labels_shape
        patch_vol = float(dz * dy * dx)

        positions: List[Tuple[int, int, int]] = []
        for z in range(0, max(1, lz - dz + 1), sz):
            for y in range(0, max(1, ly - dy + 1), sy):
                for x in range(0, max(1, lx - dx + 1), sx):
                    slab = np.asarray(
                        self._labels_array[z:z + dz, y:y + dy, x:x + dx]
                    )
                    if slab.shape != (dz, dy, dx):
                        continue
                    if self.valid_patch_value is None:
                        mask = slab > 0
                    else:
                        mask = slab == self.valid_patch_value
                    n_fg = int(mask.sum())
                    if n_fg == 0:
                        continue
                    if n_fg / patch_vol < self.min_labeled_ratio:
                        continue
                    if self._bbox_ratio(mask) < self.min_bbox_percent:
                        continue
                    if not self._image_aabb_in_bounds((z, y, x)):
                        continue
                    positions.append((z, y, x))
        logger.info("Foreground scan found %d patches in labels volume", len(positions))
        return positions

    @staticmethod
    def _bbox_ratio(mask: np.ndarray) -> float:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return 0.0
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        bbox_vol = float(np.prod(maxs - mins + 1))
        return bbox_vol / float(mask.size) if mask.size else 0.0

    def _image_aabb_in_bounds(self, position_zyx: Tuple[int, int, int]) -> bool:
        start, stop = affine.label_patch_image_aabb(
            self._matrix_zyx_label_to_image,
            position_zyx,
            self.patch_size,
            image_shape_zyx=None,
            margin=0,
        )
        for s, e, limit in zip(start, stop, self._image_shape):
            if s < 0 or e > limit:
                return False
        return True

    # --------------------------------------------------- dataset interface --

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pos = self._patches[index]
        ps = self.patch_size

        slab = np.asarray(
            self._labels_array[pos[0]:pos[0] + ps[0], pos[1]:pos[1] + ps[1], pos[2]:pos[2] + ps[2]]
        )
        label_patch = pad_or_crop_3d(slab.astype(np.float32), ps)
        label_bin = (label_patch > 0).astype(np.float32)

        image_patch = affine.resample_image_to_label_grid(
            self._image_array,
            self._matrix_zyx_label_to_image,
            pos,
            ps,
        )
        if self.normalizer is not None:
            image_patch = self.normalizer.run(image_patch)
        image_patch = image_patch.astype(np.float32, copy=False)

        padding_mask = np.ones(ps, dtype=np.float32)

        result: Dict[str, torch.Tensor] = {
            "image": torch.from_numpy(image_patch[np.newaxis, ...]),
            "padding_mask": torch.from_numpy(padding_mask[np.newaxis, ...]),
            self.target_name: torch.from_numpy(label_bin[np.newaxis, ...]),
            "is_unlabeled": bool(label_bin.sum() == 0),
            "patch_info": {
                "volume_name": "fibers",
                "position": pos,
            },
        }

        if self.transforms is not None:
            result = self.transforms(**result)
        return result

    # helpers preserved for compatibility with BaseTrainer ------------------

    @property
    def valid_patches(self):
        return [
            {"position": pos, "patch_size": self.patch_size, "volume_name": "fibers"}
            for pos in self._patches
        ]

    @property
    def n_fg(self) -> int:
        return len(self._patches)

    @property
    def n_unlabeled_fg(self) -> int:
        return 0

    def get_labeled_unlabeled_patch_indices(self) -> Tuple[List[int], List[int]]:
        return list(range(len(self._patches))), []


def _require_str(mapping: dict, key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"dataset_config.{key} is required for CrossFrameZarrDataset "
            "(provide a local path, https://, or s3:// URL)"
        )
    return value


__all__ = ["CrossFrameZarrDataset"]
