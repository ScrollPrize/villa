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

import atexit
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
from .zarr_dataset import PatchInfo


logger = logging.getLogger(__name__)


def _stage(msg: str) -> None:
    """Print a dataset-stage line to stdout with immediate flush.

    Used so operators can see exactly where __init__ is blocked (remote zarr
    probe vs. transform.json fetch vs. coarse scan vs. cache load).
    """
    print(f"[CrossFrameZarrDataset] {msg}", flush=True)


def _open_zarr_any(url: str, storage_options: Optional[dict]):
    """Open a zarr at ``url``. Uses ``open_zarr`` for http(s)/s3 URLs and
    ``zarr.open`` for local paths, because zarr rejects ``storage_options``
    on non-fsspec paths even when the mapping is empty.
    """
    if url.startswith(("http://", "https://", "s3://")):
        return open_zarr(url, mode="r", storage_options=storage_options or {})
    return zarr.open(url, mode="r")


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
        _stage(f"init: image={self.image_zarr_url}")
        _stage(f"init: labels={self.labels_zarr_url}")
        _stage(f"init: transform={self.transform_json_url}")

        shared_storage = dict(ds_cfg.get("storage_options") or {})
        self.storage_options_image = _storage_options_for(
            self.image_zarr_url,
            ds_cfg.get("storage_options_image"),
            shared_storage,
        )
        self.storage_options_labels = _storage_options_for(
            self.labels_zarr_url,
            ds_cfg.get("storage_options_labels"),
            shared_storage,
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

        # Zarr handles are opened lazily per-process (see
        # ``_ensure_process_local_handles``). An fsspec/s3fs session pickled
        # into a DataLoader worker does not survive fork/spawn cleanly, which
        # is why dinovol uses the same per-PID reopen pattern.
        self._image_array: Optional[zarr.Array] = None
        self._labels_array: Optional[zarr.Array] = None
        self._scan_array: Optional[zarr.Array] = None
        self._handle_pid: Optional[int] = None
        self._atexit_pid: Optional[int] = None
        self._debug_calls_remaining: int = self._DEBUG_CALLS_PER_PID

        # Shape metadata is read once in the parent process so the patch index
        # build doesn't have to hit the network (for S3 URLs) or disk (for local).
        _stage("probing image zarr shape ...")
        image_probe = _resolve_zarr_array(
            _open_zarr_any(self.image_zarr_url, self.storage_options_image)
        )
        self._image_shape = tuple(int(v) for v in image_probe.shape[-3:])
        _stage(f"image shape = {self._image_shape}")
        del image_probe

        _stage("probing labels zarr shape ...")
        labels_probe = _resolve_zarr_array(
            _open_zarr_any(self.labels_zarr_url, self.storage_options_labels)
        )
        self._labels_shape = tuple(int(v) for v in labels_probe.shape[-3:])
        _stage(f"labels shape = {self._labels_shape}")
        del labels_probe

        raw_scan_level = ds_cfg.get("labels_scan_level")
        self._scan_level: Optional[int] = (
            int(raw_scan_level) if raw_scan_level is not None else None
        )
        self._scan_factor: Tuple[int, int, int] = (1, 1, 1)
        if self._scan_level is not None:
            _stage(f"probing scan level {self._scan_level} ...")
            self._scan_factor = self._compute_scan_factor(self._scan_level)
            _stage(f"scan factor = {self._scan_factor}")

        _stage(f"reading transform.json ...")
        self._transform_doc = affine.read_transform_json(self.transform_json_url)
        self._matrix_xyz = self._transform_doc.matrix_xyz
        self._matrix_zyx_label_to_image = affine.label_to_image_zyx_matrix(
            self._matrix_xyz, invert=True
        )
        self._matrix_zyx_image_to_label = affine.image_to_label_zyx_matrix(
            self._matrix_xyz
        )
        _stage("transform loaded")

        self.normalization_scheme = getattr(mgr, "normalization_scheme", "zscore")
        self.intensity_properties = getattr(mgr, "intensity_properties", None) or {}
        self.normalizer = get_normalization(self.normalization_scheme, self.intensity_properties)

        self.cache_dir = self._resolve_cache_dir(ds_cfg)
        # data_path is read by BaseTrainer._configure_dataloaders to decide
        # whether train/val datasets share a source. Setting it to the mgr's
        # data_path keeps the random-split path (no cross-dataset leakage scan)
        # when both train and val datasets are built from the same config.
        self.data_path = getattr(mgr, "data_path", None)
        self._patches: List[Tuple[int, int, int]] = []
        _stage("building patch index ...")
        self._build_patch_index()
        _stage(f"patch index built: {len(self._patches)} patches")

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
                "sampling_frame": "image",  # v2: patches are image-space starts
                "labels_url": self.labels_zarr_url,
                "image_url": self.image_zarr_url,
                "patch_size": list(self.patch_size),
                "stride": list(self.stride),
                "min_labeled_ratio": self.min_labeled_ratio,
                "min_bbox_percent": self.min_bbox_percent,
                "valid_patch_value": self.valid_patch_value,
                "transform_checksum": affine.matrix_checksum(self._matrix_xyz),
                "scan_level": self._scan_level,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _scan_url(self, level: int) -> str:
        """URL of the OME-Zarr level used for the FG scan."""
        base_url = self.labels_zarr_url.rstrip("/")
        # Strip trailing ``/<digit>`` so ``<base>/0`` -> ``<base>``.
        last = base_url.rsplit("/", 1)[-1]
        parent = base_url.rsplit("/", 1)[0] if last.isdigit() else base_url
        return f"{parent}/{level}"

    def _compute_scan_factor(self, level: int) -> Tuple[int, int, int]:
        """Probe the scan level once to compute the downsample factor."""
        scan = _resolve_zarr_array(
            _open_zarr_any(self._scan_url(level), self.storage_options_labels)
        )
        scan_shape = tuple(int(v) for v in scan.shape[-3:])
        factor = tuple(
            max(1, int(round(full / s)))
            for full, s in zip(self._labels_shape, scan_shape)
        )
        logger.info(
            "CrossFrameZarrDataset: will scan FG at level %d (%s, factor=%s)",
            level, self._scan_url(level), factor,
        )
        return factor

    # ------------------------------------------------- per-process handles --

    def __getstate__(self) -> dict:
        """Strip live zarr handles before pickling into DataLoader workers.

        fsspec-backed stores (S3, HTTPS) keep asyncio loops and aiohttp
        sessions that do not survive fork/spawn. Re-open them in the worker.
        """
        state = self.__dict__.copy()
        state["_image_array"] = None
        state["_labels_array"] = None
        state["_scan_array"] = None
        state["_handle_pid"] = None
        state["_atexit_pid"] = None
        state["_debug_calls_remaining"] = self._DEBUG_CALLS_PER_PID
        return state

    def _ensure_process_local_handles(self) -> None:
        pid = os.getpid()
        if self._handle_pid == pid and self._image_array is not None:
            return
        self._image_array = _resolve_zarr_array(
            _open_zarr_any(self.image_zarr_url, self.storage_options_image)
        )
        self._labels_array = _resolve_zarr_array(
            _open_zarr_any(self.labels_zarr_url, self.storage_options_labels)
        )
        if self._scan_level is not None:
            self._scan_array = _resolve_zarr_array(
                _open_zarr_any(self._scan_url(self._scan_level), self.storage_options_labels)
            )
        self._handle_pid = pid
        if self._atexit_pid != pid:
            atexit.register(self._close_handles)
            self._atexit_pid = pid

    def _close_handles(self) -> None:
        self._image_array = None
        self._labels_array = None
        self._scan_array = None

    # Per-worker __getitem__ diagnostics. These counters are reset by
    # __getstate__ so each DataLoader worker reports a few timings from
    # its own first calls and then stays quiet.
    _DEBUG_CALLS_PER_PID: int = 2

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
        cp = self._cache_path()
        _stage(f"cache path: {cp}")
        cached = self._load_cache()
        if cached is not None and len(cached) > 0:
            self._patches = [tuple(int(v) for v in row) for row in cached]
            _stage(f"loaded {len(self._patches)} patches from cache")
            return

        _stage("no cache -- scanning foreground positions")
        self._ensure_process_local_handles()
        _stage("handles opened for scan")
        positions = self._scan_foreground_positions()
        if not positions:
            raise RuntimeError(
                "CrossFrameZarrDataset found 0 foreground patches. "
                "Check labels volume, patch size, and thresholds."
            )
        self._patches = positions
        self._save_cache(np.asarray(positions, dtype=np.int64))
        _stage(f"scan complete, cache saved to {cp}")
        # Release handles after the main-process scan so the DataLoader
        # workers don't inherit a live fsspec session via spawn pickling.
        self._close_handles()

    def _scan_foreground_positions(self) -> List[Tuple[int, int, int]]:
        if self._scan_level is not None:
            return self._scan_foreground_positions_coarse()
        return self._scan_foreground_positions_full()

    def _scan_foreground_positions_coarse(self) -> List[Tuple[int, int, int]]:
        """Enumerate image-space patch positions whose corresponding label
        region (after applying the forward transform) contains foreground.

        Strategy: read the coarse label level in one go, find label voxels
        with FG, map each to an image-space position via
        ``label_to_image_zyx``, snap the candidate image position to a
        patch-stride grid so the mapped image patch contains the FG label
        voxel, dedupe, and filter by image bounds.
        """
        assert self._scan_array is not None
        fz, fy, fx = self._scan_factor
        ps = np.asarray(self.patch_size, dtype=np.int64)
        st = np.asarray(self.stride, dtype=np.int64)

        _stage(f"coarse scan: reading {self._scan_array.shape} into memory")
        coarse = np.asarray(self._scan_array[...])
        if self.valid_patch_value is None:
            coarse_mask = coarse > 0
        else:
            coarse_mask = coarse == self.valid_patch_value
        _stage(
            f"coarse scan: {int(coarse_mask.sum())} FG voxels "
            f"/ {coarse_mask.size} total ({100.0 * coarse_mask.mean():.2f}%)"
        )

        fg_coarse = np.argwhere(coarse_mask)
        if fg_coarse.size == 0:
            return []

        # Upscale the coarse FG voxel centers into label-scale-0 voxel centers.
        factor = np.asarray(self._scan_factor, dtype=np.float64)
        fg_label = fg_coarse.astype(np.float64) * factor + 0.5 * factor
        # Map each label center to its image-space voxel.
        fg_image = affine.apply_affine_zyx(
            self._matrix_zyx_label_to_image, fg_label
        )

        # Snap each image center to the stride cell that CONTAINS it, so the
        # resulting patch [start, start + ps) actually covers the image voxel
        # that the label FG mapped to. (The old ``floor(c - ps/2) // st * st``
        # snap could push the start so far that the patch no longer contained
        # the FG voxel, producing empty label AABBs in __getitem__.)
        image_centers = np.floor(fg_image).astype(np.int64)
        image_starts = (image_centers // st) * st

        image_shape = np.asarray(self._image_shape, dtype=np.int64)
        keep = np.all(
            (image_starts >= 0) & (image_starts + ps <= image_shape), axis=1
        )
        image_starts = image_starts[keep]
        if image_starts.size == 0:
            return []

        # Dedupe.
        unique = np.unique(image_starts, axis=0)
        _stage(
            f"coarse scan: {fg_coarse.shape[0]} coarse FG -> "
            f"{unique.shape[0]} unique image-space patch starts"
        )

        # Verify each candidate by forward-mapping the patch corners back to
        # label space and checking that the AABB is non-degenerate and inside
        # the label volume. This catches positions where the coarse upscale +
        # snap combination pushes the mapped label region out of bounds under
        # rotation/shear (the Z-extent-zero case we saw in debug runs).
        ps_c = np.asarray(self.patch_size, dtype=np.int64)
        label_shape = np.asarray(self._labels_shape, dtype=np.int64)
        corners_offsets = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=np.float64)
        keep_verified = np.ones(unique.shape[0], dtype=bool)
        for i, start in enumerate(unique):
            corners = corners_offsets * ps_c + start
            mapped = affine.apply_affine_zyx(
                self._matrix_zyx_image_to_label, corners.astype(np.float64)
            )
            lo = np.floor(mapped.min(axis=0)).astype(np.int64)
            hi = np.ceil(mapped.max(axis=0)).astype(np.int64)
            if np.any(hi <= lo):
                keep_verified[i] = False
                continue
            lo = np.clip(lo, 0, label_shape)
            hi = np.clip(hi, 0, label_shape)
            if np.any(hi <= lo):
                keep_verified[i] = False
        unique = unique[keep_verified]
        _stage(
            f"coarse scan: {keep_verified.sum()} / {len(keep_verified)} "
            f"patches survive label-AABB verification"
        )

        candidates = [tuple(int(v) for v in row) for row in unique]
        _stage(f"coarse scan kept {len(candidates)} candidate patches")
        return candidates

    def _scan_foreground_positions_full(self) -> List[Tuple[int, int, int]]:
        """Full-resolution scan of the labels volume, emitting image-space
        patch starts (one per label FG patch).

        Kept as a fallback when ``labels_scan_level`` is not set. The coarse
        path is strictly preferred for realistic volumes.
        """
        dz, dy, dx = self.patch_size
        sz, sy, sx = self.stride
        lz, ly, lx = self._labels_shape

        image_shape = np.asarray(self._image_shape, dtype=np.int64)
        ps_arr = np.asarray(self.patch_size, dtype=np.int64)
        st_arr = np.asarray(self.stride, dtype=np.int64)

        seen: set = set()
        for z in range(0, max(1, lz - dz + 1), sz):
            for y in range(0, max(1, ly - dy + 1), sy):
                for x in range(0, max(1, lx - dx + 1), sx):
                    slab = np.asarray(
                        self._labels_array[z:z + dz, y:y + dy, x:x + dx]
                    )
                    if slab.shape != (dz, dy, dx):
                        continue
                    mask = (
                        slab > 0 if self.valid_patch_value is None
                        else slab == self.valid_patch_value
                    )
                    if not mask.any():
                        continue
                    # Center of this label patch -> image voxel -> snap.
                    center_label = np.array(
                        [z + dz / 2.0, y + dy / 2.0, x + dx / 2.0], dtype=np.float64
                    )[None, :]
                    center_image = affine.apply_affine_zyx(
                        self._matrix_zyx_label_to_image, center_label
                    )[0]
                    start = np.floor(center_image - ps_arr / 2.0).astype(np.int64)
                    start = (start // st_arr) * st_arr
                    if np.any(start < 0) or np.any(start + ps_arr > image_shape):
                        continue
                    seen.add(tuple(int(v) for v in start))
        _stage(f"full scan kept {len(seen)} image-space patches")
        return sorted(seen)

    @staticmethod
    def _bbox_ratio(mask: np.ndarray) -> float:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return 0.0
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        bbox_vol = float(np.prod(maxs - mins + 1))
        return bbox_vol / float(mask.size) if mask.size else 0.0

    def _image_patch_has_label_fg(self, position_image_zyx: Tuple[int, int, int]) -> bool:
        """Check that the forward-mapped label AABB of an image patch has FG."""
        start, stop = affine.image_patch_label_aabb(
            self._matrix_zyx_image_to_label,
            position_image_zyx,
            self.patch_size,
            label_shape_zyx=self._labels_shape,
            margin=0,
        )
        if any(st >= sp for st, sp in zip(start, stop)):
            return False
        slab = np.asarray(
            self._labels_array[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
        )
        if self.valid_patch_value is None:
            return bool((slab > 0).any())
        return bool((slab == self.valid_patch_value).any())

    # --------------------------------------------------- dataset interface --

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        import time

        debug = self._debug_calls_remaining > 0
        if debug:
            self._debug_calls_remaining -= 1
            tag = f"[pid={os.getpid()} idx={index}]"
            t_open_s = time.perf_counter()

        self._ensure_process_local_handles()
        if debug:
            t_open_e = time.perf_counter()
            _stage(f"__getitem__ {tag}: handles ready ({1e3*(t_open_e-t_open_s):.1f} ms)")

        pos = self._patches[index]  # image-space patch start (ZYX)
        ps = self.patch_size

        # Read the IMAGE patch natively -- it's already grid-aligned with the
        # patch coords, so no interpolation needed. This is the fast path.
        if debug:
            t_img_s = time.perf_counter()
        slab = np.asarray(
            self._image_array[pos[0]:pos[0] + ps[0], pos[1]:pos[1] + ps[1], pos[2]:pos[2] + ps[2]]
        )
        image_patch = pad_or_crop_3d(slab.astype(np.float32), ps)
        if debug:
            t_img_e = time.perf_counter()
            _stage(
                f"__getitem__ {tag}: image read {1e3*(t_img_e-t_img_s):.1f} ms "
                f"(pos={pos} shape={image_patch.shape} "
                f"min={image_patch.min():.1f} max={image_patch.max():.1f})"
            )

        if self.normalizer is not None:
            image_patch = self.normalizer.run(image_patch)
        image_patch = image_patch.astype(np.float32, copy=False)

        # Resample labels through the forward affine into the image grid.
        # The label AABB fetched from disk is ~det(linear)^(1/3) smaller than
        # the image patch, so this is tens of milliseconds, not seconds.
        if debug:
            lbl_start, lbl_stop = affine.image_patch_label_aabb(
                self._matrix_zyx_image_to_label, pos, ps,
                label_shape_zyx=self._labels_shape, margin=1,
            )
            _stage(
                f"__getitem__ {tag}: label AABB start={lbl_start} stop={lbl_stop} "
                f"size={tuple(e - s for s, e in zip(lbl_start, lbl_stop))}"
            )
            t_lbl_s = time.perf_counter()

        label_patch = affine.resample_label_to_image_grid(
            self._labels_array,
            self._matrix_zyx_image_to_label,
            pos,
            ps,
            order=0,  # nearest neighbor preserves binary labels
        )
        label_bin = (label_patch > 0).astype(np.float32)
        if debug:
            t_lbl_e = time.perf_counter()
            _stage(
                f"__getitem__ {tag}: label resample {1e3*(t_lbl_e-t_lbl_s):.1f} ms "
                f"(nonzero={int(label_bin.sum())})"
            )

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
            if debug:
                t_aug_s = time.perf_counter()
            result = self.transforms(**result)
            if debug:
                t_aug_e = time.perf_counter()
                _stage(f"__getitem__ {tag}: transforms {1e3*(t_aug_e-t_aug_s):.1f} ms")
        if debug:
            _stage(f"__getitem__ {tag}: DONE total {1e3*(time.perf_counter()-t_open_s):.1f} ms")
        return result

    # helpers preserved for compatibility with BaseTrainer ------------------

    @property
    def valid_patches(self) -> List[PatchInfo]:
        """Return patch metadata as ``PatchInfo`` so BaseTrainer's
        split/leakage-prevention logic can read ``.volume_name`` and
        ``.position`` uniformly with ZarrDataset.
        """
        return [
            PatchInfo(
                volume_index=0,
                volume_name="fibers",
                position=pos,
                patch_size=self.patch_size,
            )
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


def _storage_options_for(
    url: str, explicit: Optional[dict], shared: dict
) -> dict:
    """Return protocol-appropriate storage options for ``url``.

    ``shared`` (from ``dataset_config.storage_options``) is convenient for the
    common S3 case (``{'anon': true}``), but options like ``anon`` blow up the
    HTTPS fsspec backend. Filter the shared options by URL scheme; always honor
    an explicit ``storage_options_image`` / ``storage_options_labels``.
    """
    if explicit:
        return dict(explicit)
    opts = dict(shared)
    if url.startswith(("http://", "https://")):
        opts.pop("anon", None)  # HTTP backend doesn't accept it
    return opts


def _require_str(mapping: dict, key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"dataset_config.{key} is required for CrossFrameZarrDataset "
            "(provide a local path, https://, or s3:// URL)"
        )
    return value


__all__ = ["CrossFrameZarrDataset"]
