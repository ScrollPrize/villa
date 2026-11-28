"""Dataset that samples patches from two zarr volumes in different frames.

The label volume is authored in a canonical "fixed" frame; the image volume
lives in its own "moving" frame. An affine transform in ``transform.json``
relates the two (``p_fixed = M @ p_moving`` in XYZ order). We enumerate
foreground patches natively in label-voxel coords, then resample the
matching image region through ``M^{-1}`` with trilinear interpolation.

All S3/HTTPS/local zarr I/O goes through ``vc_reader``.
The patch index is built once per unique (labels URL, patch size, thresholds,
transform checksum) tuple and cached on disk.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from vc_reader.reader import ZarrReader, open_node

from vesuvius.data import affine
from vesuvius.utils.utils import pad_or_crop_3d

from ..augmentation.pipelines import create_training_transforms, sample_seed
import vc_reader.transforms as A
from .zarr_dataset import PatchInfo


logger = logging.getLogger(__name__)


def _stage(msg: str) -> None:
    """Print a dataset-stage line to stdout with immediate flush.

    Used so operators can see exactly where __init__ is blocked (remote zarr
    probe vs. transform.json fetch vs. coarse scan vs. cache load).
    """
    print(f"[CrossFrameZarrDataset] {msg}", flush=True)


def _open_zarr_any(url: str, storage_options: Optional[dict]):
    """Group or array at ``url``, read in DataLoader workers. ``storage_options``
    may say ``anon``: true to send no credentials.
    """
    auth = None if (storage_options or {}).get("anon") else "auto"
    return open_node(url, auth=auth, threads=1)


def _resolve_zarr_array(obj) -> ZarrReader:
    """Return the array for a group-or-array object.

    ``open_node`` returns a group when the URL points at an OME-Zarr
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


# The coarse scan dedupes patch starts on the image's stride grid, one byte per
# grid cell. Grids with more cells than this (32 MiB) sort cell indices instead.
_DEDUPE_TABLE_MAX_CELLS = 1 << 25
# Voxels / patches handled per step of the coarse scan, so its temporaries stay
# at a few MiB however many foreground voxels the scan level has.
_DEDUPE_CHUNK_ROWS = 1 << 16
# Compact accumulated 1-D fallback keys once they reach ~32 MiB.
_DEDUPE_COMPACT_KEYS = 1 << 22
# Target voxels per Z-slab read (aligned up to chunks[0] so no chunk is read
# twice) and per argwhere sub-slab inside each read.
_SCAN_SLAB_TARGET_VOXELS = 1 << 24
_SCAN_SUBSLAB_TARGET_VOXELS = 1 << 22


def _unique_starts_by_row_sort(
    image_centers: np.ndarray,
    st: np.ndarray,
    ps: np.ndarray,
    image_shape: np.ndarray,
) -> np.ndarray:
    """Sorted unique patch starts for ``(N, 3)`` int64 image voxels.

    Snaps each voxel to the stride cell that contains it, drops the starts
    whose patch ``[start, start + ps)`` leaves the image, and dedupes the
    remaining rows with a lexicographic sort. Returns ``(K, 3)`` int64.
    """
    image_starts = (image_centers // st) * st
    keep = np.all(
        (image_starts >= 0) & (image_starts + ps <= image_shape), axis=1
    )
    image_starts = image_starts[keep]
    if image_starts.shape[0] == 0:
        return image_starts
    return np.unique(image_starts, axis=0)


class _StrideGridDeduper:
    """Streaming stride-grid deduplicator for ``(N, 3)`` int64 image voxels.

    A start is ``cell * st`` with ``cell = center // st``, and its patch fits
    the image exactly when ``0 <= cell <= (image_shape - ps) // st``. The
    distinct starts are therefore the occupied cells of a grid with that many
    cells per axis, and the grid's C-order index sorts cells the way a
    lexicographic sort orders their rows. Occupied cells are marked in a
    boolean table, or their indices are sorted when the grid has more than
    ``_DEDUPE_TABLE_MAX_CELLS`` cells.

    That argument needs a positive stride and int64 arithmetic that cannot
    wrap. Chunks outside it, such as voxels cast from non-finite coordinates,
    fall back to :func:`_unique_starts_by_row_sort` per chunk and are merged
    on :meth:`finalize`.
    """

    def __init__(
        self,
        st: np.ndarray,
        ps: np.ndarray,
        image_shape: np.ndarray,
    ) -> None:
        self._st = np.asarray(st, dtype=np.int64)
        self._ps = np.asarray(ps, dtype=np.int64)
        self._image_shape = np.asarray(image_shape, dtype=np.int64)
        stride = [int(v) for v in self._st]
        patch = [int(v) for v in self._ps]
        max_cell = [
            (int(size) - p) // s
            for size, p, s in zip(self._image_shape, patch, stride)
        ] if min(stride) > 0 else [-1, -1, -1]
        grid = tuple(m + 1 for m in max_cell)
        n_cells = math.prod(grid) if min(max_cell) >= 0 else 0
        # No start or ``start + ps`` computed from a voxel in [-exact, exact] wraps.
        exact = (
            np.iinfo(np.int64).max
            - max(abs(s) for s in stride)
            - max(abs(p) for p in patch)
        )
        self._fast_valid = (
            min(stride) > 0
            and min(max_cell) >= 0
            and n_cells <= np.iinfo(np.intp).max
            and exact >= 0
        )
        self._grid = grid
        self._exact = exact
        self._last_cell = np.asarray(max_cell, dtype=np.int64)
        self._use_table = self._fast_valid and (n_cells <= _DEDUPE_TABLE_MAX_CELLS)
        self._occupied: Optional[np.ndarray] = (
            np.zeros(n_cells, dtype=bool) if self._use_table else None
        )
        self._key_chunks: List[np.ndarray] = []
        self._key_count: int = 0
        self._compact_at: int = _DEDUPE_COMPACT_KEYS
        self._fallback_rows: List[np.ndarray] = []

    def add(self, image_centers: np.ndarray) -> None:
        n = image_centers.shape[0]
        if n == 0:
            return
        for lo in range(0, n, _DEDUPE_CHUNK_ROWS):
            block = image_centers[lo:lo + _DEDUPE_CHUNK_ROWS]
            if (
                not self._fast_valid
                or block.min() < -self._exact
                or block.max() > self._exact
            ):
                rows = _unique_starts_by_row_sort(
                    block, self._st, self._ps, self._image_shape
                )
                if rows.shape[0] > 0:
                    self._fallback_rows.append(rows)
                continue
            cells = block // self._st
            cells = cells[
                np.all((cells >= 0) & (cells <= self._last_cell), axis=1)
            ]
            if cells.shape[0] == 0:
                continue
            flat = np.ravel_multi_index(
                (cells[:, 0], cells[:, 1], cells[:, 2]), self._grid
            )
            if self._use_table:
                assert self._occupied is not None
                self._occupied[flat] = True
            else:
                self._key_chunks.append(flat)
                self._key_count += flat.size
                if self._key_count >= self._compact_at:
                    compacted = np.unique(np.concatenate(self._key_chunks))
                    self._key_chunks = [compacted]
                    self._key_count = compacted.size
                    # Wait for the keys to double so the re-sorts stay amortised.
                    self._compact_at = max(
                        _DEDUPE_COMPACT_KEYS, 2 * compacted.size
                    )

    def finalize(self) -> np.ndarray:
        parts: List[np.ndarray] = []
        if self._fast_valid:
            if self._use_table:
                assert self._occupied is not None
                flat = np.flatnonzero(self._occupied)
            elif self._key_chunks:
                flat = np.unique(np.concatenate(self._key_chunks))
            else:
                flat = np.empty(0, dtype=np.intp)
            if flat.size > 0:
                fast_unique = np.stack(
                    np.unravel_index(flat, self._grid), axis=1
                ).astype(np.int64, copy=False)
                fast_unique *= self._st
                parts.append(fast_unique)
        if self._fallback_rows:
            parts.extend(self._fallback_rows)
        if not parts:
            return np.empty((0, 3), dtype=np.int64)
        if len(parts) == 1 and not self._fallback_rows:
            return parts[0]
        return np.unique(np.vstack(parts), axis=0)


def _unique_starts(
    image_centers: np.ndarray,
    st: np.ndarray,
    ps: np.ndarray,
    image_shape: np.ndarray,
) -> np.ndarray:
    """The rows of :func:`_unique_starts_by_row_sort`, in the same order and
    dtype, without materialising or sorting ``(N, 3)`` rows.
    """
    deduper = _StrideGridDeduper(st, ps, image_shape)
    deduper.add(image_centers)
    return deduper.finalize()


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
        # Multi-class FG filter: a patch's coarse-scan cell must contain one of
        # these label values. Useful when `ignore_label` removes a value from
        # the training set but we still want to require real FG in patches.
        raw_vlv = ds_cfg.get("valid_label_values")
        self.valid_label_values: Optional[np.ndarray] = (
            None if raw_vlv is None
            else np.asarray([int(v) for v in raw_vlv], dtype=np.int64)
        )
        raw_binarize = ds_cfg.get("binarize_labels")
        if raw_binarize is None:
            # Default: binarize only when there's no explicit multi-class list.
            self.binarize_labels = self.valid_label_values is None
        else:
            self.binarize_labels = bool(raw_binarize)
        self.min_labeled_ratio = float(getattr(mgr, "min_labeled_ratio", 0.01))
        self.min_bbox_percent = float(getattr(mgr, "min_bbox_percent", 0.15))
        self.stride: Tuple[int, int, int] = tuple(
            int(v) for v in ds_cfg.get("patch_stride", self.patch_size)
        )

        # Zarr handles are opened lazily per-process (see
        # ``_ensure_process_local_handles``).
        self._image_array: Optional[ZarrReader] = None
        self._labels_array: Optional[ZarrReader] = None
        self._scan_array: Optional[ZarrReader] = None
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
        scheme = self.normalization_scheme
        if scheme == "ct":
            props = self.intensity_properties
            required = ("mean", "std", "percentile_00_5", "percentile_99_5")
            if not props or any(key not in props for key in required):
                raise ValueError("CT normalization requires mean, std and percentile bounds")
            op = A.Normalize("fixed", center=float(props["mean"]), scale=max(float(props["std"]), 1e-8),
                             bounds=(float(props["percentile_00_5"]), float(props["percentile_99_5"])))
        elif scheme == "zscore":
            op = A.Normalize("zscore", nonzero=False, eps=1e-8)
        elif scheme in ("minmax", "rescale_to_01", "robust"):
            op = A.Normalize("minmax" if scheme == "rescale_to_01" else scheme)
        elif scheme == "none":
            op = None
        else:
            raise ValueError(f"Unknown normalization scheme: {scheme}")
        self.normalizer = A.Pipeline([op], backend="cpu") if op is not None else None

        self.cache_dir = self._resolve_cache_dir(ds_cfg)
        # data_path is read by BaseTrainer._configure_dataloaders to decide
        # whether train/val datasets share a source. Setting it to the mgr's
        # data_path keeps the random-split path (no cross-dataset leakage scan)
        # when both train and val datasets are built from the same config.
        self.data_path = getattr(mgr, "data_path", None)
        self._patches: List[Tuple[int, int, int]] = []
        self._valid_patches_cache: Optional[List[PatchInfo]] = None
        _stage("building patch index ...")
        self._build_patch_index()
        _stage(f"patch index built: {len(self._patches)} patches")

        # Detect skeleton-aware targets so augmentation adds a <target>_skel
        # tensor that the composite MedialSurfaceRecall loss consumes.
        skeleton_targets, skeleton_ignore_values = self._collect_skeleton_targets()

        self.transforms = None
        if self.is_training:
            self.transforms = create_training_transforms(
                patch_size=self.patch_size,
                label_names=[self.target_name],
                backend="cuda" if getattr(mgr, "augment_on_device", False) else "cpu",
                no_spatial=bool(getattr(mgr, "no_spatial_augmentation", False)),
                no_scaling=bool(getattr(mgr, "no_scaling_augmentation", False)),
                skeleton_targets=skeleton_targets or None,
                skeleton_ignore_values=skeleton_ignore_values or None,
            )
        elif skeleton_targets:
            from ..augmentation.pipelines.training_transforms import create_validation_transforms
            self.transforms = create_validation_transforms(
                label_names=[self.target_name],
                skeleton_targets=skeleton_targets,
                skeleton_ignore_values=skeleton_ignore_values or None,
            )

        logger.info(
            "CrossFrameZarrDataset: image=%s (shape=%s) labels=%s (shape=%s) patches=%d",
            self.image_zarr_url, self._image_shape,
            self.labels_zarr_url, self._labels_shape,
            len(self._patches),
        )

    # ------------------------------------------------ skeleton targets --

    # Loss names that require a precomputed skeleton tensor alongside the
    # label. Kept in sync with ZarrDataset._get_skeleton_targets.
    _SKELETON_LOSSES = {
        "MedialSurfaceRecall",
        "SoftSkeletonRecallLoss",
        "DC_SkelREC_and_CE_loss",
    }

    def _collect_skeleton_targets(self) -> Tuple[List[str], Dict[str, int]]:
        skeleton_targets: List[str] = []
        skeleton_ignore_values: Dict[str, int] = {}
        for target_name, target_info in self.targets.items():
            losses = target_info.get("losses") if isinstance(target_info, dict) else None
            if not losses:
                continue
            if any(l.get("name") in self._SKELETON_LOSSES for l in losses):
                skeleton_targets.append(target_name)
                ignore_label = (
                    target_info.get("ignore_label")
                    or target_info.get("ignore_index")
                    or target_info.get("ignore_value")
                )
                if ignore_label is not None:
                    skeleton_ignore_values[target_name] = int(ignore_label)
        return skeleton_targets, skeleton_ignore_values

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
                "valid_label_values": (
                    None if self.valid_label_values is None
                    else [int(v) for v in self.valid_label_values]
                ),
                "binarize_labels": bool(self.binarize_labels),
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
        # PatchInfo cache is trivially re-derived; don't ship it to workers.
        state["_valid_patches_cache"] = None
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

        Strategy: stream the coarse label level in chunk-aligned Z-slabs,
        find label voxels with FG, map each to an image-space position via
        ``label_to_image_zyx``, snap the candidate image position to a
        patch-stride grid so the mapped image patch contains the FG label
        voxel, dedupe, and filter by image bounds.
        """
        assert self._scan_array is not None
        ps = np.asarray(self.patch_size, dtype=np.int64)
        st = np.asarray(self.stride, dtype=np.int64)
        factor = np.asarray(self._scan_factor, dtype=np.float64)
        half_factor = 0.5 * factor
        image_shape = np.asarray(self._image_shape, dtype=np.int64)
        deduper = _StrideGridDeduper(st, ps, image_shape)

        def _fg_mask(arr: np.ndarray) -> np.ndarray:
            if self.valid_patch_value is not None:
                return arr == self.valid_patch_value
            if self.valid_label_values is not None:
                return np.isin(arr, self.valid_label_values)
            return arr > 0

        def _add_fg_coords(fg_coords: np.ndarray, batched: bool = True) -> None:
            for lo in range(0, fg_coords.shape[0], _DEDUPE_CHUNK_ROWS):
                fg_label = fg_coords[lo:lo + _DEDUPE_CHUNK_ROWS].astype(np.float64)
                fg_label *= factor
                fg_label += half_factor
                if batched and fg_label.shape[0] == 1:
                    # matmul can round one point differently from a batch, and
                    # the cache must match a scan that mapped all FG at once.
                    fg_label = np.repeat(fg_label, 2, axis=0)
                # Map each label center to its image-space voxel and snap to
                # the stride cell that CONTAINS it (the old floor(c - ps/2)
                # snap could push the start past the mapped FG voxel).
                fg_image = affine.apply_affine_zyx(
                    self._matrix_zyx_label_to_image, fg_label
                )
                np.floor(fg_image, out=fg_image)
                deduper.add(fg_image.astype(np.int64))

        # A scan with a single FG voxel maps it unbatched, so hold the first
        # voxel until the count is known.
        lone_fg: List[np.ndarray] = []

        def _ingest_fg_coords(fg_coords: np.ndarray) -> None:
            if n_fg == 1:
                lone_fg.append(fg_coords)
                return
            if lone_fg:
                _add_fg_coords(lone_fg.pop())
            _add_fg_coords(fg_coords)

        scan_shape = tuple(int(v) for v in self._scan_array.shape)
        _stage(f"coarse scan: scanning {scan_shape}")
        n_fg = 0
        n_total = math.prod(scan_shape) if scan_shape else 0
        if len(scan_shape) == 3 and n_total > 0:
            nz, ny, nx = scan_shape
            plane_voxels = max(1, ny * nx)
            raw_chunks = getattr(self._scan_array, "chunks", None)
            chunk_z = (
                int(raw_chunks[0])
                if raw_chunks is not None and len(raw_chunks) == 3 and int(raw_chunks[0]) > 0
                else 64
            )
            raw_slab_z = max(1, _SCAN_SLAB_TARGET_VOXELS // plane_voxels)
            slab_z = ((raw_slab_z + chunk_z - 1) // chunk_z) * chunk_z
            sub_z = max(1, _SCAN_SUBSLAB_TARGET_VOXELS // plane_voxels)

            for z0 in range(0, nz, slab_z):
                z1 = min(z0 + slab_z, nz)
                slab = np.asarray(self._scan_array[z0:z1, 0:ny, 0:nx])
                for sz0 in range(0, slab.shape[0], sub_z):
                    sz1 = min(sz0 + sub_z, slab.shape[0])
                    sub_mask = _fg_mask(slab[sz0:sz1])
                    sub_fg = int(sub_mask.sum())
                    if sub_fg == 0:
                        continue
                    n_fg += sub_fg
                    fg_sub = np.argwhere(sub_mask)
                    fg_sub[:, 0] += z0 + sz0
                    _ingest_fg_coords(fg_sub)
        elif n_total > 0:
            coarse = np.asarray(self._scan_array[...])
            coarse_mask = _fg_mask(coarse)
            del coarse
            n_fg = int(coarse_mask.sum())
            if n_fg > 0:
                fg_coarse = np.argwhere(coarse_mask)
                del coarse_mask
                _ingest_fg_coords(fg_coarse)
        if lone_fg:
            _add_fg_coords(lone_fg.pop(), batched=False)

        pct = (100.0 * n_fg / n_total) if n_total else 0.0
        _stage(f"coarse scan: {n_fg} FG voxels / {n_total} total ({pct:.2f}%)")
        if n_fg == 0:
            return []

        unique = deduper.finalize()
        if unique.shape[0] == 0:
            return []
        _stage(
            f"coarse scan: {n_fg} coarse FG -> "
            f"{unique.shape[0]} unique image-space patch starts"
        )

        # Verify each candidate by forward-mapping the patch corners back to
        # label space and checking that the AABB is non-degenerate and inside
        # the label volume. Vectorised in bounded chunks so corner arrays stay
        # at a few MiB even when N is in the millions.
        n = unique.shape[0]
        ps_c = np.asarray(self.patch_size, dtype=np.float64)
        label_shape = np.asarray(self._labels_shape, dtype=np.float64)
        corners_offsets = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=np.float64) * ps_c[None, :]  # (8, 3)
        keep_verified = np.empty(n, dtype=bool)
        for lo in range(0, n, _DEDUPE_CHUNK_ROWS):
            hi = min(lo + _DEDUPE_CHUNK_ROWS, n)
            m = hi - lo
            corners = (
                corners_offsets[None, :, :]
                + unique[lo:hi].astype(np.float64)[:, None, :]
            ).reshape(m * 8, 3)
            mapped = affine.apply_affine_zyx(
                self._matrix_zyx_image_to_label, corners
            ).reshape(m, 8, 3)
            lo_c = np.clip(np.floor(mapped.min(axis=1)), 0.0, label_shape)
            hi_c = np.clip(np.ceil(mapped.max(axis=1)), 0.0, label_shape)
            keep_verified[lo:hi] = np.all(hi_c > lo_c, axis=1)
        unique = unique[keep_verified]
        _stage(
            f"coarse scan: {int(keep_verified.sum())} / {n} patches survive "
            f"label-AABB verification"
        )

        candidates: List[Tuple[int, int, int]] = []
        for lo in range(0, unique.shape[0], _DEDUPE_CHUNK_ROWS):
            candidates.extend(
                map(tuple, unique[lo:lo + _DEDUPE_CHUNK_ROWS].tolist())
            )
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
                    if self.valid_patch_value is not None:
                        mask = slab == self.valid_patch_value
                    elif self.valid_label_values is not None:
                        mask = np.isin(slab, self.valid_label_values)
                    else:
                        mask = slab > 0
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
        if self.valid_patch_value is not None:
            return bool((slab == self.valid_patch_value).any())
        if self.valid_label_values is not None:
            return bool(np.isin(slab, self.valid_label_values).any())
        return bool((slab > 0).any())

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

        if self.normalizer is not None and image_patch.size:
            normalized = self.normalizer(
                {"image": np.asarray(image_patch, dtype=np.float32).reshape(1, 1, -1)}, seed=0
            )["image"]
            image_patch = normalized.reshape(image_patch.shape)
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

        linear = self._matrix_zyx_image_to_label[:3, :3]
        label_patch = self._labels_array.read_affine(
            ps,
            linear @ np.asarray(pos, dtype=np.float64) + self._matrix_zyx_image_to_label[:3, 3],
            linear,
            order=0,  # nearest neighbor preserves integer labels
        )
        if self.binarize_labels:
            label_out = (label_patch > 0).astype(np.float32)
        else:
            label_out = label_patch.astype(np.float32, copy=False)
        if debug:
            t_lbl_e = time.perf_counter()
            _stage(
                f"__getitem__ {tag}: label resample {1e3*(t_lbl_e-t_lbl_s):.1f} ms "
                f"(nonzero={int((label_out != 0).sum())})"
            )

        padding_mask = np.ones(ps, dtype=np.float32)

        result: Dict[str, torch.Tensor] = {
            "image": torch.from_numpy(image_patch[np.newaxis, ...]),
            "padding_mask": torch.from_numpy(padding_mask[np.newaxis, ...]),
            self.target_name: torch.from_numpy(label_out[np.newaxis, ...]),
            "is_unlabeled": bool((label_out != 0).sum() == 0),
            "patch_info": {
                "volume_name": self.target_name,
                "position": pos,
            },
        }

        if self.transforms is not None and not getattr(self.mgr, "augment_on_device", False):
            if debug:
                t_aug_s = time.perf_counter()
            result = self.transforms(result, seed=sample_seed(index))
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

        Cached on first access: callers like ``save_train_val_filenames``
        do ``valid_patches[idx]`` inside a loop over 700k indices, so a
        fresh build per access turned 1 startup step into 524e9 object
        creations.
        """
        if self._valid_patches_cache is None:
            self._valid_patches_cache = [
                PatchInfo(
                    volume_index=0,
                    volume_name=self.target_name,
                    position=pos,
                    patch_size=self.patch_size,
                )
                for pos in self._patches
            ]
        return self._valid_patches_cache

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
