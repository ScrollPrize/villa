import atexit
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

from dinovol_2.dataset.cuda_pipeline import CropView, SpatialPlan
from dinovol_2.dataset.volume_reader import open_volume
from dinovol_2.dataset.object_labels import (
    FIBER_KIND,
    INK_KIND,
    SHEET_KIND,
    MeshSegment,
    crop_mesh_objects,
    decode_fiber_directions,
    distinct_fiber_pairs,
    fiber_objects,
    load_or_build_volume_meshes,
    slab_objects,
)
from dinovol_2.dataset.point_annotations import (
    load_point_collection,
    map_scale0_voxel_centers,
    xyz_to_zyx,
)


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    below = np.cumsum(hist)
    above = below[-1] - below
    sum_below = np.cumsum(hist * centers)
    mean_below = sum_below / np.maximum(below, 1)
    mean_above = (sum_below[-1] - sum_below) / np.maximum(above, 1)
    return float(centers[np.argmax(below * above * (mean_below - mean_above) ** 2)])


def object_mask_cells(means: np.ndarray, nonzero: np.ndarray, closing: int) -> np.ndarray:
    """Scanned-object mask on coarse cells from their mean intensity and any-nonzero flag.

    Masked volumes are zero outside a dilated object mask; inside it, tissue is separated
    from the surrounding air by an Otsu threshold, closed over `closing` cells (the gaps
    between wraps), and holes are filled per z slice and in 3D. Components smaller than 1%
    of the largest are dropped.
    """
    if not nonzero.any():
        return np.zeros_like(nonzero)
    tissue = means > _otsu_threshold(means[nonzero])
    padded = np.pad(tissue, closing)  # closing must not erode the object where it meets the volume edge
    closed = ndi.binary_closing(padded, structure=ndi.generate_binary_structure(3, 1), iterations=closing)
    closed = closed[(slice(closing, -closing),) * 3]
    filled = ndi.binary_fill_holes(np.stack([ndi.binary_fill_holes(plane) for plane in closed]))
    labels, count = ndi.label(filled)
    if count > 1:
        sizes = ndi.sum(filled, labels, range(1, count + 1))
        filled = np.isin(labels, 1 + np.flatnonzero(sizes >= 0.01 * sizes.max()))
    return filled & nonzero


def window_start_cells(inside: np.ndarray, factor: int, window, lo, hi, min_fraction: float = 1.0):
    """Cells c such that every window start in [c * factor, (c + 1) * factor) gives a window
    inside `inside` (cells of `factor` voxels at the origin) and within voxels [lo, hi).

    A window starting anywhere in a cell covers up to n = (factor + window - 2) // factor + 1
    cells along each axis; with min_fraction < 1, at least that fraction of those n**3 cells
    must be inside instead of all of them.
    """
    window = np.asarray(window)
    n = (factor + window - 2) // factor + 1
    shape = np.asarray(inside.shape)
    out = np.zeros(inside.shape, dtype=bool)
    span = shape - n + 1
    if np.any(span <= 0):
        return out
    # all(inside[c : c + n]) per axis via a summed-volume table
    table = np.pad(inside.astype(np.int32), ((1, 0), (1, 0), (1, 0))).cumsum(0).cumsum(1).cumsum(2)
    a = [slice(0, s) for s in span]
    b = [slice(k, k + s) for k, s in zip(n, span)]
    total = (
        table[b[0], b[1], b[2]] - table[a[0], b[1], b[2]] - table[b[0], a[1], b[2]] - table[b[0], b[1], a[2]]
        + table[a[0], a[1], b[2]] + table[a[0], b[1], a[2]] + table[b[0], a[1], a[2]] - table[a[0], a[1], a[2]]
    )
    full = total >= int(math.ceil(min_fraction * int(np.prod(n)) - 1e-9))
    cell = [np.arange(s) for s in span]
    ok = [(c * factor >= l) & (c * factor + factor - 1 + w <= h) for c, l, h, w in zip(cell, lo, hi, window)]
    out[tuple(slice(0, s) for s in span)] = full & ok[0][:, None, None] & ok[1][None, :, None] & ok[2][None, None, :]
    return out


@dataclass
class Volume:
    usable_bbox: tuple  # (z0, y0, x0, z1, y1, x1)
    valid_crop_starts: int
    scale: int
    path: str
    weight: float = 0.0
    point_coordinates: np.ndarray | None = None
    point_type_ids: np.ndarray | None = None
    # Dilated nonzero mask of a coarse pyramid level over the usable bbox; cell
    # occupancy_origin + (i, j, k) covers occupancy_factor**3 selected-scale voxels.
    occupancy: np.ndarray | None = None
    occupancy_origin: tuple[int, int, int] | None = None
    occupancy_factor: int = 1
    # Object mask (see object_mask in the config): cells of occupancy_factor**3 voxels where
    # a source window may start so that it lies entirely inside the scanned object.
    # Bit-packed along x over a [start_cells_shape] block at absolute cell start_cells_origin.
    pixel_size_um: float | None = None
    start_cells: np.ndarray | None = None
    start_cells_origin: tuple[int, int, int] | None = None
    start_cells_shape: tuple[int, int, int] | None = None
    # Pseudo-object label sources aligned with the selected scale (see object_points).
    surface_path: str | None = None
    surface_level: int = 0
    fiber_paths: tuple[str, str, str] | None = None  # presence, nx, ny
    fiber_level: int = 0
    fiber_factor: int = 1  # selected-scale voxels per fiber voxel
    object_mask_min_inside: float = 1.0  # fraction of a window that must lie inside the object mask
    object_mask_source: str = "object"  # "object" (tissue envelope) or "sam2" (the published mask as-is)
    meshes: list[MeshSegment] | None = None
    has_ink_meshes: bool = False
    mesh_scale: float = 0.0
    ink_scale: float = 0.0
    ink_anchor_weight: float = 1.0
    oversample: float = 1.0


@dataclass
class ObjectPoints:
    """Points of one sample in source-window coordinates with their pseudo-objects."""

    source: np.ndarray  # [P, 3] float32 voxel coordinates in the source window
    objects: np.ndarray  # [P] int64, 0 = unlabeled
    kinds: np.ndarray  # [P] int8, 0 = unlabeled, SHEET_KIND, FIBER_KIND
    negatives: np.ndarray  # [M + 1, M + 1] bool, provably different objects
    ink: np.ndarray  # [P] int8, +1 = confirmed ink, -1 = confirmed non-ink on surface, 0 = unsupervised


@dataclass(frozen=True)
class SourcePlan:
    volume: Volume
    array: Any
    starts: tuple[int, int, int]
    anchor_index: int | None
    mesh_anchor: np.ndarray | None = None


@dataclass(frozen=True)
class SampledCropRegion:
    starts: tuple[int, int, int]
    shape: tuple[int, int, int]


def _as_3tuple(value):
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError(f"Expected 3 values, got {result}")
    return result


def _as_float_pair(value, default):
    if value is None:
        return default
    return float(value[0]), float(value[1])


def _max_3tuple(*values):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return tuple(max(int(value[axis]) for value in filtered) for axis in range(3))


class SSLZarrDataset(Dataset):
    def __init__(self, config, do_augmentations=False, single_crop_only=False):
        self.config = config
        for key in ("global_view_size", "local_view_size", "gram_teacher_view_size"):
            if key in config:
                raise ValueError(f"{key} is no longer supported; configure the corresponding crop_size instead.")
        self.do_augmentations = do_augmentations
        self.single_crop_only = bool(self.config.get("single_crop_only", False) or single_crop_only)
        self.epoch_length = int(self.config["epoch_length"]) if "epoch_length" in self.config else None
        self.global_crop_size = _as_3tuple(
            self.config["global_crop_size"] if "global_crop_size" in self.config else self.config["crop_size"]
        )
        self.num_global_crops = self.config.get("num_global_crops", 2)
        self.local_crop_size = _as_3tuple(self.config["local_crop_size"] if "local_crop_size" in self.config else None)
        self.gram_teacher_crop_size = _as_3tuple(self.config.get("gram_teacher_crop_size"))
        if self.gram_teacher_crop_size is not None and self.single_crop_only:
            raise ValueError("gram_teacher_crop_size is not supported with single_crop_only datasets.")
        self.gram_teacher_no_augmentations = bool(self.config.get("gram_teacher_no_augmentations", True))
        self.paired_global_crop_size = _max_3tuple(self.global_crop_size, self.gram_teacher_crop_size)
        double_global = tuple(2 * int(d) for d in self.paired_global_crop_size)
        default_source_sampling_size = _max_3tuple(double_global, self.global_crop_size, self.local_crop_size)
        self.source_sampling_size = _as_3tuple(
            self.config.get("source_sampling_size", self.config.get("source_crop_size", default_source_sampling_size))
        )
        self.global_crop_scale = _as_float_pair(self.config.get("global_crop_scale"), (0.32, 1.0))
        self.local_crop_scale = _as_float_pair(self.config.get("local_crop_scale"), (0.05, 0.32))
        self.num_local_crops = self.config.get("num_local_crops", 8)
        self.volume_auth = self.config["volume_auth"] if "volume_auth" in self.config else None
        self.s3_storage_options = self.config.get("s3_storage_options")
        self.vol_trim_pct = self.config.get("vol_trim_pct", 0.60)
        point_config = dict(self.config.get("point_supervision") or {})
        self.point_supervision_enabled = bool(point_config.get("enabled", False)) and not self.single_crop_only
        self.point_sampling_probability = float(point_config.get("sampling_probability", 0.1))
        self.max_points_per_view = int(point_config.get("max_points_per_view", 64))
        if not 0.0 <= self.point_sampling_probability <= 1.0:
            raise ValueError(
                f"point sampling_probability must be in [0, 1], got {self.point_sampling_probability}."
            )
        if self.max_points_per_view <= 0:
            raise ValueError(f"max_points_per_view must be positive, got {self.max_points_per_view}.")
        configured_types = []
        if self.point_supervision_enabled:
            discovered_types = set()
            for dataset in self.config["datasets"]:
                entries = dataset.get("point_collections", [])
                if not isinstance(entries, list):
                    raise ValueError("point_collections must be a list of {path, type} objects.")
                for entry_index, collection in enumerate(entries):
                    if not isinstance(collection, dict) or "path" not in collection or "type" not in collection:
                        raise ValueError(
                            f"point_collections entry {entry_index} must contain path and type."
                        )
                    if not isinstance(collection["type"], str) or not collection["type"]:
                        raise ValueError("Point collection types must be non-empty strings.")
                    discovered_types.add(collection["type"])
            configured_types = sorted(discovered_types)
        if any(not type_name for type_name in configured_types):
            raise ValueError("Point collection types must be non-empty strings.")
        self.point_type_to_id = {type_name: index for index, type_name in enumerate(configured_types)}
        # occupancy_level: pyramid level whose nonzero voxels mark where source crops can
        # reach nonzero_threshold, so crops below it are rejected before any read.
        self.occupancy_level = self.config.get("occupancy_level")
        # object_mask: source windows start only where the whole window lies inside the
        # scanned object (the scroll body, not the object-loss pseudo-objects), found on the
        # occupancy_level of the (already SAM2-masked) volume: voxels above an Otsu threshold,
        # closed over gap_um so the space between wraps and interior chambers count as inside,
        # holes filled. It replaces vol_trim_pct; only the bottom and top z_trim_pct of z
        # slices are excluded. Per dataset, thin objects can set object_mask_min_inside (< 1)
        # and object_mask_source: "sam2" uses the published mask (nonzero cells) instead.
        object_mask = dict(self.config.get("object_mask") or {})
        self.object_mask_enabled = bool(object_mask.get("enabled", False))
        self.object_mask_gap_um = float(object_mask.get("gap_um", 2000.0))
        self.object_mask_z_trim_pct = float(object_mask.get("z_trim_pct", 0.05))
        if self.object_mask_enabled:
            if self.occupancy_level is None:
                raise ValueError("object_mask requires occupancy_level (the pyramid level the mask is computed on)")
            if not 0.0 <= self.object_mask_z_trim_pct < 0.5:
                raise ValueError(f"object_mask.z_trim_pct must be in [0, 0.5), got {self.object_mask_z_trim_pct}")
        # read_ahead: source crops each process reads in background threads while it
        # augments the current sample. Their locations are drawn in the calling thread.
        self.read_ahead = int(self.config.get("read_ahead", 0))
        if self.read_ahead < 0:
            raise ValueError(f"read_ahead must be non-negative, got {self.read_ahead}")
        self._volume_handles: dict[tuple[str, int], Any] = {}
        self._handle_pid: int | None = None
        self._atexit_pid: int | None = None
        self._read_executor: ThreadPoolExecutor | None = None
        self._pending_reads: deque = deque()
        # object_points: points with pseudo-object labels (sheets from surface meshes or
        # predictions, ink from supervised ink-labels, fibers from fiber predictions)
        # computed inside the overlap of both global crops and carried into each view.
        object_points = dict(self.config.get("object_points") or {})
        self.object_points_enabled = bool(object_points.get("enabled", False)) and not self.single_crop_only
        self.points_per_sample = int(object_points.get("points_per_sample", 384))
        self.max_labeled_points = int(object_points.get("max_labeled_points", self.points_per_sample))
        self.points_per_object = int(object_points.get("points_per_object", 48))
        self.points_per_sheet = int(object_points.get("points_per_sheet", self.points_per_object))
        self.points_per_fiber = int(
            object_points.get("points_per_fiber", object_points.get("points_per_object", 32))
        )
        self.max_objects = int(object_points.get("max_objects", 12))
        self.max_sheets = int(object_points.get("max_sheets", max(1, self.max_objects // 2)))
        self.max_fibers = int(object_points.get("max_fibers", max(1, self.max_objects - self.max_sheets)))
        self.fiber_presence_threshold = float(object_points.get("presence_threshold", 127))
        self.fiber_min_voxels = int(object_points.get("fiber_min_voxels", 30))
        self.fiber_min_span = float(object_points.get("fiber_min_span", 0.0))
        self.fiber_dedup_distance = float(object_points.get("fiber_dedup_distance", 1.5))
        self.mesh_min_points = int(object_points.get("mesh_min_points", 16))
        self.mesh_dedup_distance = float(object_points.get("mesh_dedup_distance", 1.5))
        self.mesh_min_separation = float(object_points.get("mesh_min_separation", 3.0))
        self.metadata_path = object_points.get("metadata_path", self.config.get("metadata_path"))
        self.mesh_cache_dir = object_points.get("mesh_cache_dir", self.config.get("mesh_cache_dir"))
        self.mesh_oversample = float(object_points.get("mesh_oversample", 1.5))
        self.ink_oversample = float(object_points.get("ink_oversample", 2.0))
        self.fiber_oversample = float(object_points.get("fiber_oversample", 1.25))
        self.mesh_anchor_prob = float(object_points.get("mesh_anchor_prob", 0.35))
        self.ink_anchor_prob = float(object_points.get("ink_anchor_prob", 0.50))
        self.mesh_saturation_points = int(object_points.get("mesh_saturation_points", 10_000_000))
        self.ink_saturation_points = int(object_points.get("ink_saturation_points", 100_000))
        if self.object_points_enabled and self.max_labeled_points > self.points_per_sample:
            raise ValueError("object_points.max_labeled_points must not exceed points_per_sample")
        # cross_view_token_patch_size: when set, each global view also carries, for every
        # token, the coordinates of its center in the other global view (cross-view iBOT
        # targets). Both views' spatial augmentations must map token centers onto token
        # centers (flips, right-angle rotations, transposes of a cubic crop).
        token_patch = self.config.get("cross_view_token_patch_size")
        self.cross_view_token_patch_size = None
        if token_patch is not None and not self.single_crop_only:
            token_patch = tuple(int(p) for p in token_patch)
            if len(token_patch) != 3 or any(c % p for c, p in zip(self.global_crop_size, token_patch)):
                raise ValueError(
                    f"cross_view_token_patch_size {token_patch} must divide global_crop_size {self.global_crop_size}"
                )
            self.cross_view_token_patch_size = token_patch

        if not self.single_crop_only:
            if self.num_global_crops != 2:
                raise ValueError(
                    f"SSLZarrDataset currently expects exactly 2 global crops, got {self.num_global_crops}")
            if self.local_crop_size is not None and self.num_local_crops < 0:
                raise ValueError(
                    f"SSLZarrDataset expects a non-negative number of local crops, got {self.num_local_crops}")
        required_source_sampling_size = _max_3tuple(
            self.global_crop_size,
            self.local_crop_size,
            self.gram_teacher_crop_size,
        )
        if required_source_sampling_size is not None and any(
            source < required for source, required in zip(self.source_sampling_size, required_source_sampling_size)
        ):
            raise ValueError(
                "source_sampling_size must be at least as large as the largest requested crop size, "
                f"got source_sampling_size={self.source_sampling_size} and required={required_source_sampling_size}"
            )
        # augmentation: "dino" keeps only flips, right-angle rotations, transposes and
        # brightness/contrast jitter on top of the random resized crops; "full"
        # (default) is the complete pipeline.
        augmentation = str(self.config.get("augmentation", "full"))
        if augmentation not in ("full", "dino"):
            raise ValueError(f"augmentation must be 'full' or 'dino', got {augmentation!r}")
        self.global_transforms = [SpatialPlan(self.global_crop_size) for _ in range(self.num_global_crops)]
        self.local_transforms = ([SpatialPlan(self.local_crop_size) for _ in range(self.num_local_crops)]
                                 if self.local_crop_size is not None else [])

        self.volumes = []

        for dataset in self.config["datasets"]:
            volume_path = dataset["volume_path"]
            volume_scale = dataset["volume_scale"]
            handle = open_volume(volume_path, volume_scale, self.volume_auth, self.s3_storage_options)
            try:
                d_zarr = handle
                z, y, x = d_zarr.shape
                if self.object_mask_enabled:
                    z0 = int(math.ceil(z * self.object_mask_z_trim_pct))
                    z1 = z - z0
                    y0, y1, x0, x1 = 0, y, 0, x
                else:
                    k_z = max(1, round(z * self.vol_trim_pct))
                    k_y = max(1, round(y * self.vol_trim_pct))
                    k_x = max(1, round(x * self.vol_trim_pct))
                    z0 = (z - k_z) // 2
                    y0 = (y - k_y) // 2
                    x0 = (x - k_x) // 2
                    z1 = (z0 + k_z)
                    y1 = (y0 + k_y)
                    x1 = (x0 + k_x)
                usable_bbox = (z0, y0, x0, z1, y1, x1)
                crop_z, crop_y, crop_x = self.source_sampling_size
                valid_z = max(0, (z1 - z0) - crop_z + 1)
                valid_y = max(0, (y1 - y0) - crop_y + 1)
                valid_x = max(0, (x1 - x0) - crop_x + 1)
                valid_crop_starts = valid_z * valid_y * valid_x

                point_coordinates, point_type_ids = self._load_volume_points(
                    dataset,
                    selected_shape=(z, y, x),
                    usable_bbox=usable_bbox,
                )
                fiber_paths = None
                if dataset.get("fiber_presence_path"):
                    fiber_paths = (
                        str(dataset["fiber_presence_path"]),
                        str(dataset["fiber_nx_path"]),
                        str(dataset["fiber_ny_path"]),
                    )
                meshes: list[MeshSegment] = []
                if self.object_points_enabled:
                    if dataset.get("meshes") is not None:
                        meshes = list(dataset["meshes"])
                    elif (
                        dataset.get("mesh_cache_path") is not None
                        or self.metadata_path is not None
                        or self.mesh_cache_dir is not None
                        or bool(object_points.get("use_metadata_meshes", False))
                    ):
                        meshes = load_or_build_volume_meshes(
                            str(volume_path),
                            int(volume_scale),
                            (z, y, x),
                            sample_id=dataset.get("sample_id"),
                            volume_id=dataset.get("volume_id"),
                            metadata_path=dataset.get("metadata_path", self.metadata_path),
                            cache_dir=dataset.get("mesh_cache_dir", self.mesh_cache_dir),
                            mesh_cache_path=dataset.get("mesh_cache_path"),
                            s3_storage_options=self.s3_storage_options,
                        )
                for seg in meshes:
                    seg.ensure_spatial_index()
                has_ink_meshes = any(m.has_ink for m in meshes)
                oversample = float(dataset.get("oversample", 1.0))
                self.volumes.append(Volume(
                    path=str(volume_path),
                    scale=volume_scale,
                    usable_bbox=usable_bbox,
                    valid_crop_starts=valid_crop_starts,
                    point_coordinates=point_coordinates,
                    point_type_ids=point_type_ids,
                    surface_path=dataset.get("surface_path"),
                    surface_level=int(dataset.get("surface_level", 0)),
                    fiber_paths=fiber_paths,
                    fiber_level=int(dataset.get("fiber_level", 0)),
                    fiber_factor=int(dataset.get("fiber_factor", 1)),
                    pixel_size_um=dataset.get("pixel_size_um"),
                    object_mask_min_inside=float(dataset.get("object_mask_min_inside", 1.0)),
                    object_mask_source=str(dataset.get("object_mask_source", "object")),
                    meshes=meshes if meshes else None,
                    has_ink_meshes=has_ink_meshes,
                    ink_anchor_weight=float(dataset.get("ink_anchor_weight", 1.0)),
                    oversample=oversample,
                ))
            finally:
                handle.close()

        if self.occupancy_level is not None:
            # Windows inside the object mask are fully occupied: the occupancy check is moot.
            loader = self._load_object_mask if self.object_mask_enabled else self._load_occupancy
            with ThreadPoolExecutor(max_workers=min(8, len(self.volumes))) as executor:
                list(executor.map(loader, self.volumes))

        self.total_valid_crop_starts = sum(vol.valid_crop_starts for vol in self.volumes)
        if self.total_valid_crop_starts <= 0:
            raise ValueError("No valid crop starts found across configured volumes for the selected source crop size.")

        self.global_ink_anchor_prob = 0.0
        self.global_mesh_anchor_prob = 0.0
        self.ink_anchor_vol_weights: np.ndarray | None = None
        self.mesh_anchor_vol_weights: np.ndarray | None = None
        if self.object_points_enabled:
            max_mesh_pts = max(
                (sum(m.n_valid_points for m in vol.meshes) for vol in self.volumes if vol.meshes),
                default=0,
            )
            max_ink_pts = max(
                (sum(m.n_ink_points for m in vol.meshes) for vol in self.volumes if vol.meshes),
                default=0,
            )
            ref_mesh_pts = float(min(self.mesh_saturation_points, max_mesh_pts)) if max_mesh_pts > 0 else 1.0
            ref_ink_pts = float(min(self.ink_saturation_points, max_ink_pts)) if max_ink_pts > 0 else 1.0
            for vol in self.volumes:
                if vol.meshes:
                    vol_mesh_pts = sum(m.n_valid_points for m in vol.meshes)
                    vol_ink_pts = sum(m.n_ink_points for m in vol.meshes)
                    vol.mesh_scale = min(1.0, vol_mesh_pts / ref_mesh_pts) if ref_mesh_pts > 0 else 0.0
                    vol.ink_scale = min(1.0, vol_ink_pts / ref_ink_pts) if ref_ink_pts > 0 else 0.0
                    mesh_mult = 1.0 + (self.mesh_oversample - 1.0) * vol.mesh_scale
                    ink_mult = 1.0 + (self.ink_oversample - 1.0) * vol.ink_scale
                    vol.oversample *= max(mesh_mult, ink_mult)
                if vol.fiber_paths is not None:
                    vol.oversample *= self.fiber_oversample

            n_active_vols = max(1, sum(1 for vol in self.volumes if vol.valid_crop_starts > 0))
            w_ink = np.asarray(
                [
                    vol.ink_scale * vol.ink_anchor_weight
                    if (vol.has_ink_meshes and vol.valid_crop_starts > 0)
                    else 0.0
                    for vol in self.volumes
                ],
                dtype=np.float64,
            )
            w_mesh = np.asarray(
                [
                    vol.mesh_scale * (1.0 - 0.7 * vol.ink_scale * min(1.0, vol.ink_anchor_weight))
                    if (vol.meshes and vol.valid_crop_starts > 0)
                    else 0.0
                    for vol in self.volumes
                ],
                dtype=np.float64,
            )
            if w_ink.sum() > 0:
                self.global_ink_anchor_prob = 0.7 * self.ink_anchor_prob * float(w_ink.sum() / n_active_vols)
                self.ink_anchor_vol_weights = w_ink / w_ink.sum()
            if w_mesh.sum() > 0:
                self.global_mesh_anchor_prob = self.mesh_anchor_prob * float(w_mesh.sum() / n_active_vols)
                self.mesh_anchor_vol_weights = w_mesh / w_mesh.sum()

        total_weighted = sum(vol.valid_crop_starts * vol.oversample for vol in self.volumes)
        for volume in self.volumes:
            volume.weight = (volume.valid_crop_starts * volume.oversample) / total_weighted

    def _load_volume_points(self, dataset_config, *, selected_shape, usable_bbox):
        entries = dataset_config.get("point_collections", [])
        if not self.point_supervision_enabled or not entries:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int64)
        if not isinstance(entries, list):
            raise ValueError("point_collections must be a list of {path, type} objects.")

        volume_path = dataset_config["volume_path"]
        volume_scale = int(dataset_config["volume_scale"])
        scale0_handle = None
        try:
            if volume_scale == 0:
                scale0_shape = selected_shape
            else:
                scale0_handle = open_volume(
                    volume_path,
                    0,
                    self.volume_auth,
                    self.s3_storage_options,
                )
                scale0_shape = tuple(int(value) for value in scale0_handle.shape)

            all_coordinates = []
            all_type_ids = []
            usable_start = np.asarray(usable_bbox[:3], dtype=np.float64)
            usable_stop = np.asarray(usable_bbox[3:], dtype=np.float64)
            for entry_index, entry in enumerate(entries):
                if not isinstance(entry, dict) or "path" not in entry or "type" not in entry:
                    raise ValueError(
                        f"point_collections entry {entry_index} for {volume_path} must contain path and type."
                    )
                type_name = entry["type"]
                if not isinstance(type_name, str) or not type_name:
                    raise ValueError("Point collection types must be non-empty strings.")
                points = map_scale0_voxel_centers(
                    xyz_to_zyx(load_point_collection(entry["path"])),
                    scale0_shape,
                    selected_shape,
                )
                usable = np.all((points >= usable_start) & (points < usable_stop), axis=1)
                points = points[usable]
                if points.shape[0] == 0:
                    raise ValueError(
                        f"Point collection {entry['path']} has no annotations usable in the trimmed bounds "
                        f"of volume {volume_path} at scale {volume_scale}."
                    )
                all_coordinates.append(points)
                all_type_ids.append(
                    np.full((points.shape[0],), self.point_type_to_id[type_name], dtype=np.int64)
                )
            return np.concatenate(all_coordinates), np.concatenate(all_type_ids)
        finally:
            if scale0_handle is not None:
                scale0_handle.close()

    def _load_occupancy(self, volume: Volume) -> None:
        level = int(self.occupancy_level)
        read_factor = 2 ** (level - int(volume.scale))
        if read_factor < 1:
            raise ValueError(f"occupancy_level {level} must not be finer than volume_scale {volume.scale}")
        # Cells cover 2**occupancy_level selected-scale voxels on every volume; a volume
        # selected below full resolution reads the same level and max-pools it.
        pool = max(1, 2 ** level // read_factor)
        factor = read_factor * pool
        bbox_lo = np.asarray(volume.usable_bbox[:3]) // factor
        bbox_hi = -(-np.asarray(volume.usable_bbox[3:]) // factor)
        handle = open_volume(volume.path, level, self.volume_auth, self.s3_storage_options)
        try:
            array = handle
            # Read one extra cell on each side for the dilation below.
            read_lo = np.maximum((bbox_lo - 1) * pool, 0)
            read_hi = np.minimum((bbox_hi + 1) * pool, array.shape)
            fine = np.zeros(tuple((bbox_hi - bbox_lo + 2) * pool), dtype=bool)
            selection = tuple(slice(int(a), int(b)) for a, b in zip(read_lo, read_hi))
            values = np.asarray(array[selection])
            offset = read_lo - (bbox_lo - 1) * pool
            fine[tuple(slice(int(o), int(o) + n) for o, n in zip(offset, values.shape))] = values > 0
            shape = bbox_hi - bbox_lo + 2
            coarse = fine.reshape(shape[0], pool, shape[1], pool, shape[2], pool).any(axis=(1, 3, 5))
        finally:
            handle.close()
        # Downsampling can round sparse cells to zero; one cell of dilation keeps every
        # nonzero selected-scale voxel inside an occupied cell.
        coarse = binary_dilation(coarse, structure=np.ones((3, 3, 3), dtype=bool))
        volume.occupancy = np.ascontiguousarray(coarse[1:-1, 1:-1, 1:-1])
        volume.occupancy_origin = tuple(int(v) for v in bbox_lo)
        volume.occupancy_factor = factor

    def _load_object_mask(self, volume: Volume) -> None:
        if volume.pixel_size_um is None:
            raise ValueError(f"object_mask needs pixel_size_um for {volume.path}")
        level = int(self.occupancy_level)
        read_factor = 2 ** (level - int(volume.scale))
        if read_factor < 1:
            raise ValueError(f"occupancy_level {level} must not be finer than volume_scale {volume.scale}")
        # Same cells as the occupancy grid: 2**occupancy_level selected-scale voxels.
        pool = max(1, 2 ** level // read_factor)
        factor = read_factor * pool
        handle = open_volume(volume.path, level, self.volume_auth, self.s3_storage_options)
        try:
            array = handle
            cells = -(-np.asarray(array.shape) // pool)
            means = np.zeros(tuple(cells), dtype=np.float32)
            nonzero = np.zeros(tuple(cells), dtype=bool)
            slab = 64 * pool  # z planes per read: the finest levels are too large to hold at once
            for z in range(0, array.shape[0], slab):
                values = np.asarray(array[z:z + slab])
                padded = np.zeros(tuple(-(-n // pool) * pool for n in values.shape), dtype=values.dtype)
                padded[tuple(slice(0, n) for n in values.shape)] = values
                blocks = padded.reshape(
                    padded.shape[0] // pool, pool, padded.shape[1] // pool, pool, padded.shape[2] // pool, pool
                )
                rows = slice(z // pool, z // pool + blocks.shape[0])
                means[rows] = blocks.mean(axis=(1, 3, 5), dtype=np.float32)
                nonzero[rows] = (blocks > 0).any(axis=(1, 3, 5))
        finally:
            handle.close()
        if volume.object_mask_source == "sam2":
            # The published (SAM2) mask as-is, air margin included. Cells count as nonzero if any
            # voxel is, so edge cells straddle the mask: erode by one cell to keep windows off it.
            inside = ndi.binary_erosion(nonzero, structure=np.ones((3, 3, 3), dtype=bool))
        elif volume.object_mask_source == "object":
            closing = max(1, round(self.object_mask_gap_um / (float(volume.pixel_size_um) * factor)))
            inside = object_mask_cells(means, nonzero, closing)
        else:
            raise ValueError(f"object_mask_source must be 'object' or 'sam2', got {volume.object_mask_source!r}")
        bbox = np.asarray(volume.usable_bbox)
        starts = window_start_cells(
            inside, factor, self.source_sampling_size, bbox[:3], bbox[3:], volume.object_mask_min_inside
        )
        volume.occupancy = None
        volume.occupancy_factor = factor
        volume.valid_crop_starts = int(starts.sum()) * factor ** 3
        if starts.any():
            nz = np.nonzero(starts)
            lo = np.array([a.min() for a in nz]); hi = np.array([a.max() for a in nz]) + 1
            block = starts[tuple(slice(a, b) for a, b in zip(lo, hi))]
            volume.start_cells = np.packbits(block, axis=2)
            volume.start_cells_origin = tuple(int(v) for v in lo)
            volume.start_cells_shape = tuple(int(v) for v in block.shape)
        else:
            print(f"object_mask: no source window fits inside the object of {volume.path}; volume skipped", flush=True)

    def _start_allowed(self, volume: Volume, starts) -> bool:
        """Whether a source window at these starts lies inside the object mask."""
        cell = np.asarray(starts) // volume.occupancy_factor - np.asarray(volume.start_cells_origin)
        if np.any(cell < 0) or np.any(cell >= np.asarray(volume.start_cells_shape)):
            return False
        z, y, x = (int(v) for v in cell)
        return bool((volume.start_cells[z, y, x >> 3] >> (7 - (x & 7))) & 1)

    def _sample_object_start(self, volume: Volume):
        """Uniform over the window starts that keep the whole window inside the object mask."""
        shape = volume.start_cells_shape
        factor = volume.occupancy_factor
        while True:
            z, y, x = (np.random.randint(n) for n in shape)
            if (volume.start_cells[z, y, x >> 3] >> (7 - (x & 7))) & 1:
                break
        return tuple(
            int((origin + cell) * factor + np.random.randint(factor))
            for origin, cell in zip(volume.start_cells_origin, (z, y, x))
        )

    def _occupied_fraction(self, volume: Volume, starts) -> float:
        """Fraction of the source window inside occupied cells, an upper bound on its nonzero fraction."""
        factor = volume.occupancy_factor
        slices, weights = [], []
        for start, size, origin in zip(starts, self.source_sampling_size, volume.occupancy_origin):
            stop = int(start) + int(size)
            first, last = int(start) // factor, (stop - 1) // factor
            edges = np.arange(first, last + 2) * factor
            weights.append((np.minimum(edges[1:], stop) - np.maximum(edges[:-1], int(start))).astype(np.float64))
            slices.append(slice(first - origin, last + 1 - origin))
        block = volume.occupancy[tuple(slices)].astype(np.float64)
        return float(np.einsum("ijk,i,j,k->", block, *weights)) / math.prod(self.source_sampling_size)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_volume_handles"] = {}
        state["_handle_pid"] = None
        state["_atexit_pid"] = None
        state["_read_executor"] = None
        state["_pending_reads"] = deque()
        return state

    def _register_close_hook(self) -> None:
        pid = os.getpid()
        if self._atexit_pid == pid:
            return
        atexit.register(self.close)
        self._atexit_pid = pid

    def _ensure_process_local_handles(self) -> None:
        pid = os.getpid()
        if self._handle_pid == pid:
            return
        self.close()
        self._read_executor = None
        self._pending_reads = deque()
        self._volume_handles = {}
        self._handle_pid = pid
        self._register_close_hook()

    def _get_volume_array(self, volume: Volume):
        return self._get_array(volume.path, volume.scale)

    def _get_array(self, path, level):
        self._ensure_process_local_handles()
        key = (str(path), int(level))
        handle = self._volume_handles.get(key)
        if handle is None:
            handle = open_volume(path, level, self.volume_auth, self.s3_storage_options)
            self._volume_handles[key] = handle
        return handle

    def _read_region(self, array, lo, hi):
        selection = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        return np.asarray(array[selection])

    def _read_coarse_region(self, array, lo, hi, factor):
        """Read [lo, hi) of the selected-scale grid from an array `factor` times coarser."""
        lo, hi = np.asarray(lo), np.asarray(hi)
        coarse_lo = lo // factor
        coarse_hi = -(-hi // factor)
        values = self._read_region(array, coarse_lo, coarse_hi)
        for axis in range(3):
            values = values.repeat(factor, axis=axis)
        offset = lo - coarse_lo * factor
        return values[tuple(slice(int(o), int(o) + int(n)) for o, n in zip(offset, hi - lo))]

    @staticmethod
    def _sample_fiber_voxels(voxels: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        """Sample `count` voxels spread along the full length of a 3D fiber component."""
        n = len(voxels)
        if count >= n:
            return voxels
        centered = voxels.astype(np.float32) - voxels.mean(axis=0, keepdims=True)
        _, eigvecs = np.linalg.eigh(centered.T @ centered)
        proj = centered @ eigvecs[:, -1]
        order = np.argsort(proj, kind="stable")
        edges = np.linspace(0, n, count + 1, dtype=np.int64)
        picks = [int(order[rng.integers(edges[i], max(edges[i] + 1, edges[i + 1]))]) for i in range(count)]
        return voxels[np.asarray(picks, dtype=np.int64)]

    def _sample_object_points(self, volume: Volume, window_starts, crop_regions) -> ObjectPoints:
        """Points inside both global crops; labeled ones come from meshes/ink and/or slab/volume predictions."""
        overlap_lo = np.max([region.starts for region in crop_regions], axis=0)
        overlap_hi = np.min([np.add(region.starts, region.shape) for region in crop_regions], axis=0)
        if np.any(overlap_hi <= overlap_lo):
            raise RuntimeError("Global crops of one source window must overlap.")
        window_starts_arr = np.asarray(window_starts)
        absolute_lo = window_starts_arr + overlap_lo
        absolute_hi = window_starts_arr + overlap_hi

        rng = np.random.default_rng(np.random.randint(2**31))
        sources, objects, kinds, inks = [], [], [], []
        negatives = np.zeros((self.max_objects + 1, self.max_objects + 1), dtype=bool)
        budget = min(self.max_labeled_points, self.points_per_sample)
        next_obj_id = 1

        # 1. Surface sheets + ink from pre-cached 3D meshes (when available on this volume).
        if volume.meshes:
            max_mesh_sheets = min(self.max_sheets, self.max_objects)
            sampled_sheets, mesh_neg_pairs = crop_mesh_objects(
                volume.meshes,
                absolute_lo,
                absolute_hi,
                min_points=self.mesh_min_points,
                dedup_distance=self.mesh_dedup_distance,
                min_separation=self.mesh_min_separation,
                max_sheets=max_mesh_sheets,
                points_per_sheet=self.points_per_sheet,
                rng=rng,
            )
            mesh_id_map: dict[int, int] = {}
            for sheet_idx, (coords_in_overlap, ink_state) in enumerate(sampled_sheets, start=1):
                if budget <= 0 or next_obj_id > self.max_objects:
                    break
                take = min(len(coords_in_overlap), budget)
                if take <= 0:
                    continue
                coords_take = coords_in_overlap[:take] + overlap_lo.astype(np.float32)
                ink_take = ink_state[:take]
                obj_id = next_obj_id
                next_obj_id += 1
                mesh_id_map[sheet_idx] = obj_id
                sources.append(coords_take.astype(np.float32))
                objects.append(np.full(take, obj_id, dtype=np.int64))
                kinds.append(np.full(take, SHEET_KIND, dtype=np.int8))
                inks.append(ink_take.astype(np.int8))
                budget -= take
            for a, b in mesh_neg_pairs:
                if a in mesh_id_map and b in mesh_id_map:
                    ia, ib = mesh_id_map[a], mesh_id_map[b]
                    negatives[ia, ib] = negatives[ib, ia] = True

        # 2. Fallback binary surface predictions (when meshes are absent) and 3D fiber predictions.
        surface = None
        if not volume.meshes and volume.surface_path:
            surface = self._read_region(
                self._get_array(volume.surface_path, volume.surface_level), absolute_lo, absolute_hi
            )
        presence = direction = None
        if volume.fiber_paths is not None and budget > 0 and next_obj_id <= self.max_objects:
            presence_path, nx_path, ny_path = volume.fiber_paths
            presence = self._read_coarse_region(
                self._get_array(presence_path, volume.fiber_level), absolute_lo, absolute_hi, volume.fiber_factor
            )
            if int((presence > self.fiber_presence_threshold).sum()) < self.fiber_min_voxels:
                presence = None
            else:
                nx_arr = self._get_array(nx_path, volume.fiber_level)
                ny_arr = self._get_array(ny_path, volume.fiber_level)
                if self._read_executor is None:
                    self._read_executor = ThreadPoolExecutor(max_workers=max(2, self.read_ahead), thread_name_prefix="source-read")
                fut_nx = self._read_executor.submit(
                    self._read_coarse_region, nx_arr, absolute_lo, absolute_hi, volume.fiber_factor
                )
                ny = self._read_coarse_region(ny_arr, absolute_lo, absolute_hi, volume.fiber_factor)
                nx = fut_nx.result()
                direction = decode_fiber_directions(nx, ny)

        if surface is not None or presence is not None:
            labels = slab_objects(
                surface,
                presence,
                direction,
                presence_threshold=self.fiber_presence_threshold,
                fiber_min_voxels=self.fiber_min_voxels,
                fiber_min_span=self.fiber_min_span,
                fiber_dedup_distance=self.fiber_dedup_distance,
                rng=rng,
            )
            remaining_objects = max(0, self.max_objects - (next_obj_id - 1))
            if surface is not None:
                n_sheets = min(labels.n_sheets, max(remaining_objects // 2, remaining_objects - labels.n_fibers))
            else:
                n_sheets = 0
            n_fibers = min(labels.n_fibers, remaining_objects - n_sheets)
            def _coords_by_label(vol_labels: np.ndarray, max_lbl: int) -> dict[int, np.ndarray]:
                if max_lbl <= 0:
                    return {}
                flat = np.flatnonzero((vol_labels > 0) & (vol_labels <= max_lbl))
                if flat.size == 0:
                    return {}
                ids = vol_labels.ravel()[flat]
                order_idx = np.argsort(ids, kind="stable")
                flat, ids = flat[order_idx], ids[order_idx]
                bounds = np.searchsorted(ids, np.arange(1, max_lbl + 2))
                return {
                    lbl: np.stack(np.unravel_index(flat[bounds[lbl - 1]:bounds[lbl]], vol_labels.shape), axis=-1)
                    for lbl in range(1, max_lbl + 1)
                    if bounds[lbl] > bounds[lbl - 1]
                }

            sheet_coords = _coords_by_label(labels.sheet_labels, n_sheets)
            fiber_coords = _coords_by_label(labels.fiber_labels, n_fibers)
            new_id: dict[int, int] = {}
            candidates = [(SHEET_KIND, sheet_coords, label, label) for label in range(1, n_sheets + 1)]
            candidates += [
                (FIBER_KIND, fiber_coords, label, label + labels.n_sheets) for label in range(1, n_fibers + 1)
            ]
            for order in rng.permutation(len(candidates)):
                kind, coord_map, label, combined = candidates[order]
                if budget <= 0 or next_obj_id > self.max_objects:
                    break
                voxels = coord_map.get(label)
                if voxels is None or len(voxels) == 0:
                    continue
                per_obj = self.points_per_sheet if kind == SHEET_KIND else self.points_per_fiber
                take = min(per_obj, len(voxels), budget)
                if kind == FIBER_KIND:
                    chosen = self._sample_fiber_voxels(voxels, take, rng)
                else:
                    chosen = voxels[rng.choice(len(voxels), take, replace=False)]
                obj_id = next_obj_id
                next_obj_id += 1
                new_id[combined] = obj_id
                sources.append(chosen.astype(np.float32) + overlap_lo)
                objects.append(np.full(take, obj_id, dtype=np.int64))
                kinds.append(np.full(take, kind, dtype=np.int8))
                inks.append(np.zeros(take, dtype=np.int8))
                budget -= take
            for a, b in labels.negative_pairs:
                if a in new_id and b in new_id:
                    ia, ib = new_id[a], new_id[b]
                    negatives[ia, ib] = negatives[ib, ia] = True

        labeled = sum(len(points) for points in sources)
        unlabeled = max(0, self.points_per_sample - labeled)
        if unlabeled > 0:
            sources.append(
                (np.random.randint(overlap_lo, overlap_hi, size=(unlabeled, 3))).astype(np.float32)
            )
            objects.append(np.zeros(unlabeled, dtype=np.int64))
            kinds.append(np.zeros(unlabeled, dtype=np.int8))
            inks.append(np.zeros(unlabeled, dtype=np.int8))
        return ObjectPoints(
            source=np.concatenate(sources),
            objects=np.concatenate(objects),
            kinds=np.concatenate(kinds),
            negatives=negatives,
            ink=np.concatenate(inks),
        )

    def close(self) -> None:
        executor = getattr(self, "_read_executor", None)
        if executor is not None and self._handle_pid == os.getpid():
            executor.shutdown(wait=True, cancel_futures=True)
            self._read_executor = None
            self._pending_reads.clear()
        for handle in self._volume_handles.values():
            handle.close()
        self._volume_handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _sample_crop_shape(self, scale_range):
        ref_depth, ref_height, ref_width = self.source_sampling_size
        min_scale, max_scale = scale_range
        scale = np.random.uniform(min_scale, max_scale)
        scale_per_dim = scale ** (1.0 / 3.0)
        return (
            max(1, int(round(ref_depth * scale_per_dim))),
            max(1, int(round(ref_height * scale_per_dim))),
            max(1, int(round(ref_width * scale_per_dim))),
        )

    def _finalize_crop(self, crop, target_size):
        return CropView(torch.from_numpy(np.array(crop, copy=True, order="C")), tuple(target_size))

    def _sample_source_start(self, usable_bbox):
        z0, y0, x0, z1, y1, x1 = usable_bbox
        crop_d, crop_h, crop_w = self.source_sampling_size
        z_start = np.random.randint(z0, z1 - crop_d + 1)
        y_start = np.random.randint(y0, y1 - crop_h + 1)
        x_start = np.random.randint(x0, x1 - crop_w + 1)
        return z_start, y_start, x_start

    def _read_source_window(self, d_zarr, starts):
        z_start, y_start, x_start = starts
        crop_d, crop_h, crop_w = self.source_sampling_size
        selection = (
            slice(z_start, z_start + crop_d),
            slice(y_start, y_start + crop_h),
            slice(x_start, x_start + crop_w),
        )
        return np.asarray(d_zarr[selection])

    def _read_source_crop_3d(self, d_zarr, usable_bbox):
        return self._read_source_window(d_zarr, self._sample_source_start(usable_bbox))

    def _read_point_centered_source_crop_3d(self, d_zarr, usable_bbox, anchor):
        starts = self._sample_point_centered_source_start(usable_bbox, anchor)
        return self._read_source_window(d_zarr, starts), starts

    def _sample_point_centered_source_start(self, usable_bbox, anchor):
        starts = []
        for bbox_start, bbox_stop, crop_size, coordinate in zip(
            usable_bbox[:3],
            usable_bbox[3:],
            self.source_sampling_size,
            anchor,
        ):
            minimum = max(
                int(bbox_start),
                int(np.ceil(float(coordinate) - (int(crop_size) - 1))),
            )
            maximum = min(
                int(bbox_stop) - int(crop_size),
                int(np.floor(float(coordinate))),
            )
            if maximum < minimum:
                raise ValueError(
                    f"Annotated point {tuple(float(value) for value in anchor)} cannot fit in a source window "
                    f"of size {self.source_sampling_size} inside bounds {usable_bbox}."
                )
            starts.append(int(np.random.randint(minimum, maximum + 1)))
        return tuple(starts)

    def _sample_random_resized_crop_region(
        self,
        source_shape,
        scale_range,
        target_size,
        *,
        anchor=None,
        anchor_target_size=None,
    ):
        crop_shape = self._sample_crop_shape(scale_range)
        starts = []
        for axis, (source_dim, crop_dim) in enumerate(zip(source_shape, crop_shape)):
            low = 0
            high_inclusive = int(source_dim) - int(crop_dim)
            if anchor is not None:
                if anchor_target_size is None:
                    raise ValueError("anchor_target_size is required with an anchor.")
                # Keep the point inside the resized crop's voxel-center bounds.
                target_dim = float(anchor_target_size[axis])
                relative_min = 0.5 * int(crop_dim) / target_dim - 0.5
                relative_max = (target_dim - 0.5) * int(crop_dim) / target_dim - 0.5
                low = max(low, int(np.ceil(float(anchor[axis]) - relative_max - 1e-7)))
                high_inclusive = min(
                    high_inclusive, int(np.floor(float(anchor[axis]) - relative_min + 1e-7))
                )
            if high_inclusive < low:
                raise ValueError(
                    f"Cannot sample crop {crop_shape} inside source {source_shape} with anchor={anchor}"
                )
            starts.append(int(np.random.randint(low, high_inclusive + 1)))
        return SampledCropRegion(starts=tuple(starts), shape=tuple(int(size) for size in crop_shape))

    def _materialize_crop_from_region(self, source_crop, crop_region, target_size):
        z_start, y_start, x_start = crop_region.starts
        crop_d, crop_h, crop_w = crop_region.shape
        crop = source_crop[
            z_start:z_start + crop_d,
            y_start:y_start + crop_h,
            x_start:x_start + crop_w,
        ]
        return self._finalize_crop(crop, target_size)

    def _random_resized_crop_3d_from_array(self, source_crop, scale_range, target_size):
        crop_region = self._sample_random_resized_crop_region(
            source_crop.shape,
            scale_range,
            target_size,
        )
        return self._materialize_crop_from_region(source_crop, crop_region, target_size)

    def _read_random_resized_crop_3d(self, d_zarr, usable_bbox, scale_range, target_size):
        source_crop = self._read_source_crop_3d(d_zarr, usable_bbox)
        return self._random_resized_crop_3d_from_array(
            source_crop,
            scale_range,
            target_size,
        )

    def _apply_paired_global_transform(self, transform, global_view, gram_teacher_view):
        transformed = transform(image=global_view, paired_spatial_image=gram_teacher_view)
        if gram_teacher_view is not None:
            gram_teacher_view.augment = not self.gram_teacher_no_augmentations
        return transformed["image"], transformed["paired_spatial_image"]

    def _apply_paired_global_transform_with_points(
        self, transform, global_view, gram_teacher_view, point_coordinates,
    ):
        transformed = transform(image=global_view, paired_spatial_image=gram_teacher_view,
                                keypoints=point_coordinates)
        if gram_teacher_view is not None:
            gram_teacher_view.augment = not self.gram_teacher_no_augmentations
        return transformed["image"], transformed["paired_spatial_image"], transformed["keypoints"]

    @staticmethod
    def _map_points_to_view(points, crop_region, target_size):
        if points.shape[0] == 0:
            return torch.empty((0, 3), dtype=torch.float32)
        starts = np.asarray(crop_region.starts, dtype=np.float64)
        shape = np.asarray(crop_region.shape, dtype=np.float64)
        target = np.asarray(target_size, dtype=np.float64)
        mapped = (points - starts + 0.5) * (target / shape) - 0.5
        return torch.from_numpy(mapped.astype(np.float32, copy=False))

    def _visible_points_for_crop(self, volume, source_starts, crop_region, anchor_index):
        source_starts_array = np.asarray(source_starts, dtype=np.float64)
        source_points = volume.point_coordinates - source_starts_array
        crop_starts = np.asarray(crop_region.starts, dtype=np.float64)
        crop_stops = crop_starts + np.asarray(crop_region.shape, dtype=np.float64) - 1.0
        visible = np.all((source_points >= crop_starts) & (source_points <= crop_stops), axis=1)
        visible_indices = np.flatnonzero(visible)
        if anchor_index not in visible_indices:
            raise RuntimeError("The selected point anchor was not retained in its global crop.")
        ordered_indices = np.concatenate((
            np.asarray([anchor_index], dtype=np.int64),
            visible_indices[visible_indices != anchor_index],
        ))
        return (
            self._map_points_to_view(source_points[ordered_indices], crop_region, self.global_crop_size),
            torch.from_numpy(volume.point_type_ids[ordered_indices].copy()).long(),
        )

    def _filter_and_cap_view_points(self, coordinates, type_ids):
        maximum = coordinates.new_tensor(tuple(float(value - 1) for value in self.global_crop_size))
        visible = torch.all((coordinates >= 0.0) & (coordinates <= maximum), dim=1)
        if coordinates.shape[0] and not bool(visible[0]):
            raise RuntimeError("The selected point anchor fell outside the student output crop.")
        coordinates = coordinates[visible]
        type_ids = type_ids[visible]
        if coordinates.shape[0] <= self.max_points_per_view:
            return coordinates, type_ids

        selected = [0]
        remaining_budget = self.max_points_per_view - 1
        candidates_by_type = {}
        for type_id in torch.unique(type_ids, sorted=True).tolist():
            candidates = torch.nonzero(type_ids == type_id, as_tuple=False).flatten().tolist()
            candidates = [index for index in candidates if index != 0]
            random.shuffle(candidates)
            candidates_by_type[int(type_id)] = candidates
        type_order = list(candidates_by_type)
        while remaining_budget > 0 and type_order:
            next_order = []
            for type_id in type_order:
                candidates = candidates_by_type[type_id]
                if candidates and remaining_budget > 0:
                    selected.append(candidates.pop())
                    remaining_budget -= 1
                if candidates:
                    next_order.append(type_id)
            type_order = next_order
        selected_tensor = torch.tensor(selected, dtype=torch.long)
        return coordinates[selected_tensor], type_ids[selected_tensor]

    def __len__(self):
        if self.epoch_length is not None:
            return self.epoch_length
        return self.total_valid_crop_starts

    @staticmethod
    def _sample_volume_anchor(volume):
        represented_types = np.unique(volume.point_type_ids)
        selected_type = int(np.random.choice(represented_types))
        candidates = np.flatnonzero(volume.point_type_ids == selected_type)
        return int(np.random.choice(candidates))

    def _sample_mesh_centered_start(
        self,
        vol: Volume,
        *,
        prefer_ink: bool | None = None,
    ) -> tuple[tuple[int, int, int], np.ndarray] | None:
        """Sample a source window start centered around a valid mesh or ink point."""
        if not vol.meshes:
            return None
        if prefer_ink is None:
            prefer_ink = vol.has_ink_meshes and (np.random.random() < 0.7)
        candidates = [m for m in vol.meshes if m.has_ink] if prefer_ink else vol.meshes
        if not candidates:
            candidates = vol.meshes
            prefer_ink = False
        for seg in candidates:
            seg.ensure_spatial_index()
        weights = np.asarray(
            [max(1, seg.n_ink_points if prefer_ink else seg.n_valid_points) for seg in candidates],
            dtype=np.float64,
        )
        weights /= weights.sum()
        bbox_lo = np.asarray(vol.usable_bbox[:3], dtype=np.float32)
        bbox_hi = np.asarray(vol.usable_bbox[3:], dtype=np.float32)
        for _ in range(8):
            seg = candidates[int(np.random.choice(len(candidates), p=weights))]
            pts = seg.anchor_ink_zyx if (prefer_ink and seg.has_ink and seg.anchor_ink_zyx.shape[0] > 0) else seg.anchor_valid_zyx
            if pts is None or pts.shape[0] == 0:
                continue
            pt = pts[int(np.random.randint(pts.shape[0]))]
            if not np.all((pt >= bbox_lo) & (pt < bbox_hi)):
                continue
            try:
                starts = self._sample_point_centered_source_start(vol.usable_bbox, pt)
            except ValueError:
                continue
            if self.object_mask_enabled and not self._start_allowed(vol, starts):
                continue
            return starts, pt.astype(np.float64, copy=True)
        return None

    def _draw_source_plan(self, nonzero_threshold):
        vol_weights = [vol.weight for vol in self.volumes]
        while True:
            anchor_index = None
            mesh_anchor = None
            starts = None
            u_anchor = np.random.random()
            if (
                self.object_points_enabled
                and self.ink_anchor_vol_weights is not None
                and u_anchor < self.global_ink_anchor_prob
            ):
                vol = self.volumes[int(np.random.choice(len(self.volumes), p=self.ink_anchor_vol_weights))]
                centered = self._sample_mesh_centered_start(vol, prefer_ink=True)
                if centered is not None:
                    starts, mesh_anchor = centered
            elif (
                self.object_points_enabled
                and self.mesh_anchor_vol_weights is not None
                and u_anchor < self.global_ink_anchor_prob + self.global_mesh_anchor_prob
            ):
                vol = self.volumes[int(np.random.choice(len(self.volumes), p=self.mesh_anchor_vol_weights))]
                centered = self._sample_mesh_centered_start(vol, prefer_ink=False)
                if centered is not None:
                    starts, mesh_anchor = centered
            else:
                vol = self.volumes[int(np.random.choice(len(self.volumes), p=vol_weights))]
                if (
                    self.point_supervision_enabled
                    and vol.point_coordinates is not None
                    and vol.point_coordinates.shape[0] > 0
                    and np.random.random() < self.point_sampling_probability
                ):
                    anchor_index = self._sample_volume_anchor(vol)
                    starts = self._sample_point_centered_source_start(
                        vol.usable_bbox,
                        vol.point_coordinates[anchor_index],
                    )
            if starts is None:
                if self.object_mask_enabled:
                    starts = self._sample_object_start(vol)
                else:
                    starts = self._sample_source_start(vol.usable_bbox)
            if self.object_mask_enabled and not self._start_allowed(vol, starts):
                continue  # point-centered windows that would leave the object
            # The occupied fraction bounds the nonzero fraction from above, so this
            # skips only windows the check after reading would reject.
            if vol.occupancy is not None and self._occupied_fraction(vol, starts) < nonzero_threshold:
                continue
            return SourcePlan(vol, self._get_volume_array(vol), tuple(starts), anchor_index, mesh_anchor)

    def _next_source_crop(self, nonzero_threshold):
        while True:
            if self._pending_reads:
                plan, future = self._pending_reads.popleft()
                source_crop = future.result()
            else:
                plan = self._draw_source_plan(nonzero_threshold)
                source_crop = self._read_source_window(plan.array, plan.starts)
            if source_crop.size == 0:
                continue
            # Windows drawn from the object mask are kept as they are; otherwise the
            # nonzero fraction decides.
            if self.object_mask_enabled or np.count_nonzero(source_crop) / source_crop.size >= nonzero_threshold:
                break
        if self.read_ahead:
            if self._read_executor is None:
                self._read_executor = ThreadPoolExecutor(max_workers=self.read_ahead, thread_name_prefix="source-read")
            while len(self._pending_reads) < self.read_ahead:
                next_plan = self._draw_source_plan(nonzero_threshold)
                self._pending_reads.append(
                    (next_plan, self._read_executor.submit(self._read_source_window, next_plan.array, next_plan.starts))
                )
        return plan, source_crop

    def __getitem__(self, idx):
        nonzero_threshold = float(self.config.get("nonzero_threshold", 0.30))
        plan, source_crop = self._next_source_crop(nonzero_threshold)
        vol = plan.volume
        anchor_index = plan.anchor_index
        mesh_anchor = plan.mesh_anchor
        source_starts = plan.starts if (anchor_index is not None or mesh_anchor is not None) else None

        if self.single_crop_only:
            crop = self._random_resized_crop_3d_from_array(
                source_crop,
                self.global_crop_scale,
                self.global_crop_size,
            )
            if self.do_augmentations:
                crop = self.global_transforms[0](image=crop)["image"]
            return crop

        global_views = []
        gram_teacher_views = []
        global_point_coordinates = []
        global_point_type_ids = []
        object_points = None
        object_point_views = []
        crop_regions = [None] * len(self.global_transforms)
        token_keypoints = []
        if self.object_points_enabled or self.cross_view_token_patch_size is not None:
            # Object points and token correspondences need both regions first.
            crop_regions = [
                self._sample_global_crop_region(source_crop, vol, anchor_index, source_starts, mesh_anchor=mesh_anchor)
                for _ in self.global_transforms
            ]
        if self.object_points_enabled:
            object_points = self._sample_object_points(vol, plan.starts, crop_regions)
        for view_index, (transform, crop_region) in enumerate(zip(self.global_transforms, crop_regions)):
            # Sample one shared region so the student and Gram teacher observe the
            # exact same 3D field of view, optionally at different resolutions.
            if crop_region is None:
                crop_region = self._sample_global_crop_region(
                    source_crop, vol, anchor_index, source_starts, mesh_anchor=mesh_anchor
                )
            extra_keypoints = None
            if self.cross_view_token_patch_size is not None:
                # This view's token centers, then the other view's token centers mapped
                # into this view (both before augmentation).
                centers = self._token_centers()
                other_region = crop_regions[1 - view_index]
                other_centers = self._map_points_to_view(
                    self._view_to_source(centers, other_region), crop_region, self.global_crop_size
                )
                extra_keypoints = torch.cat([centers, other_centers])
            global_view = self._materialize_crop_from_region(source_crop, crop_region, self.global_crop_size)
            gram_teacher_view = None
            if self.gram_teacher_crop_size is not None:
                gram_teacher_view = self._materialize_crop_from_region(
                    source_crop,
                    crop_region,
                    self.gram_teacher_crop_size,
                )
            if object_points is not None:
                global_view, gram_teacher_view, view_points = self._transform_object_points(
                    transform, global_view, gram_teacher_view, object_points, crop_region, vol, anchor_index,
                    source_starts, extra_keypoints=extra_keypoints,
                )
                point_coordinates, point_type_ids, object_point_coordinates, extra_keypoints = view_points
                object_point_views.append(object_point_coordinates)
            elif anchor_index is not None:
                point_coordinates, point_type_ids = self._visible_points_for_crop(
                    vol,
                    source_starts,
                    crop_region,
                    anchor_index,
                )
                if self.do_augmentations:
                    n_points = point_coordinates.shape[0]
                    keypoints = point_coordinates if extra_keypoints is None else torch.cat(
                        [point_coordinates, extra_keypoints]
                    )
                    global_view, gram_teacher_view, keypoints = (
                        self._apply_paired_global_transform_with_points(
                            transform,
                            global_view,
                            gram_teacher_view,
                            keypoints,
                        )
                    )
                    keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
                    point_coordinates = keypoints[:n_points]
                    if extra_keypoints is not None:
                        extra_keypoints = keypoints[n_points:]
                point_coordinates, point_type_ids = self._filter_and_cap_view_points(
                    point_coordinates,
                    point_type_ids,
                )
            else:
                point_coordinates = torch.empty((0, 3), dtype=torch.float32)
                point_type_ids = torch.empty((0,), dtype=torch.long)
                if self.do_augmentations and extra_keypoints is not None:
                    global_view, gram_teacher_view, extra_keypoints = self._apply_paired_global_transform_with_points(
                        transform, global_view, gram_teacher_view, extra_keypoints
                    )
                    extra_keypoints = torch.as_tensor(extra_keypoints, dtype=torch.float32)
                elif self.do_augmentations:
                    global_view, gram_teacher_view = self._apply_paired_global_transform(
                        transform,
                        global_view,
                        gram_teacher_view,
                    )
            if extra_keypoints is not None:
                token_keypoints.append(extra_keypoints)
            global_views.append(global_view)
            global_point_coordinates.append(point_coordinates)
            global_point_type_ids.append(point_type_ids)
            if gram_teacher_view is not None:
                gram_teacher_views.append(gram_teacher_view)

        local_views = []
        if self.local_crop_size is not None:
            local_views = [
                self._random_resized_crop_3d_from_array(
                    source_crop,
                    self.local_crop_scale,
                    self.local_crop_size,
                )
                for _ in range(self.num_local_crops)
            ]
            if self.do_augmentations:
                local_views = [
                    transform(image=view)["image"]
                    for transform, view in zip(self.local_transforms, local_views)
                ]

        sample = {
            "global_views": global_views,
            "local_views": local_views,
            "global_point_coordinates": global_point_coordinates,
            "global_point_type_ids": global_point_type_ids,
        }
        if gram_teacher_views:
            sample["gram_teacher_views"] = gram_teacher_views
        if token_keypoints:
            sample["global_token_targets"] = self._cross_view_token_targets(token_keypoints)
        if object_points is not None:
            # Valid = inside the view's field of view (voxel edges at -0.5 and size - 0.5).
            size = torch.tensor(self.global_crop_size, dtype=torch.float32) - 0.5
            sample["object_point_coordinates"] = object_point_views
            sample["object_point_valid"] = [
                torch.all((coordinates >= -0.5) & (coordinates <= size), dim=1) for coordinates in object_point_views
            ]
            sample["object_point_source"] = torch.from_numpy(object_points.source)
            sample["object_point_objects"] = torch.from_numpy(object_points.objects)
            sample["object_point_kinds"] = torch.from_numpy(object_points.kinds)
            sample["object_negatives"] = torch.from_numpy(object_points.negatives)
            sample["object_point_ink"] = torch.from_numpy(object_points.ink)
        return sample

    def _sample_global_crop_region(self, source_crop, vol, anchor_index, source_starts, *, mesh_anchor=None):
        anchor = None
        if anchor_index is not None:
            anchor = vol.point_coordinates[anchor_index] - np.asarray(source_starts, dtype=np.float64)
        elif mesh_anchor is not None and source_starts is not None:
            raw_anchor = np.asarray(mesh_anchor, dtype=np.float64) - np.asarray(source_starts, dtype=np.float64)
            hi_bound = np.maximum(1.0, np.asarray(source_crop.shape, dtype=np.float64) - 2.0)
            anchor = np.clip(raw_anchor, np.minimum(1.0, hi_bound), hi_bound)
        return self._sample_random_resized_crop_region(
            source_crop.shape,
            self.global_crop_scale,
            self.paired_global_crop_size,
            anchor=anchor,
            anchor_target_size=self.global_crop_size if anchor is not None else None,
        )

    def _transform_object_points(
        self, transform, global_view, gram_teacher_view, object_points, crop_region, vol, anchor_index, source_starts,
        extra_keypoints=None,
    ):
        """Map object points (and any supervision points and extra keypoints) into one augmented global view."""
        object_coordinates = self._map_points_to_view(object_points.source, crop_region, self.global_crop_size)
        if anchor_index is not None:
            point_coordinates, point_type_ids = self._visible_points_for_crop(
                vol, source_starts, crop_region, anchor_index
            )
        else:
            point_coordinates = torch.empty((0, 3), dtype=torch.float32)
            point_type_ids = torch.empty((0,), dtype=torch.long)
        if self.do_augmentations:
            parts = [point_coordinates, object_coordinates] + ([extra_keypoints] if extra_keypoints is not None else [])
            global_view, gram_teacher_view, keypoints = self._apply_paired_global_transform_with_points(
                transform, global_view, gram_teacher_view, torch.cat(parts)
            )
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            point_coordinates, object_coordinates, *rest = keypoints.split([p.shape[0] for p in parts])
            extra_keypoints = rest[0] if rest else None
        if anchor_index is not None:
            point_coordinates, point_type_ids = self._filter_and_cap_view_points(point_coordinates, point_type_ids)
        return global_view, gram_teacher_view, (point_coordinates, point_type_ids, object_coordinates, extra_keypoints)

    def _token_centers(self):
        """Token centers of a global view in voxel coordinates, [tokens, 3] in token order."""
        axes = [
            torch.arange(size // patch, dtype=torch.float32) * patch + (patch - 1) / 2
            for size, patch in zip(self.global_crop_size, self.cross_view_token_patch_size)
        ]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)

    def _view_to_source(self, coordinates, crop_region):
        """Inverse of _map_points_to_view (before augmentation)."""
        starts = np.asarray(crop_region.starts, dtype=np.float64)
        shape = np.asarray(crop_region.shape, dtype=np.float64)
        target = np.asarray(self.global_crop_size, dtype=np.float64)
        return (coordinates.double().numpy() + 0.5) * (shape / target) - 0.5 + starts

    def _cross_view_token_targets(self, token_keypoints):
        """For each global view, the center of each of its tokens in the other view, [views, tokens, 3].

        token_keypoints[v] holds view v's own token centers and then the other view's
        token centers mapped into view v, both after view v's augmentation and in the
        order of the token grid before augmentation. Coordinates outside the other view
        (beyond -0.5 / size - 0.5) mark tokens without a counterpart.
        """
        centers = self._token_centers()
        n_tokens = centers.shape[0]
        grid = tuple(size // patch for size, patch in zip(self.global_crop_size, self.cross_view_token_patch_size))
        patch = torch.tensor(self.cross_view_token_patch_size, dtype=torch.float32)
        targets = []
        for view_index in range(2):
            own = token_keypoints[view_index][:n_tokens]
            in_other = token_keypoints[1 - view_index][n_tokens:]
            index = (own - (patch - 1) / 2) / patch
            rounded = index.round()
            if (index - rounded).abs().max() > 1e-3:
                raise ValueError("cross-view token targets need augmentations that map token centers onto token centers")
            rounded = rounded.long()
            flat = (rounded[:, 0] * grid[1] + rounded[:, 1]) * grid[2] + rounded[:, 2]
            if not torch.equal(torch.sort(flat).values, torch.arange(n_tokens)):
                raise ValueError("cross-view token targets need augmentations that permute the token grid")
            view_targets = torch.empty_like(in_other)
            view_targets[flat] = in_other
            targets.append(view_targets)
        return torch.stack(targets)
