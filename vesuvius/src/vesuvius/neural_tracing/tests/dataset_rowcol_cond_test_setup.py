import random
import sys
from functools import lru_cache
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import zarr


PROJECT_SRC = Path(__file__).resolve().parents[3]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import vesuvius.tifxyz as tifxyz
from vesuvius.neural_tracing.datasets.common import ChunkPatch
from vesuvius.neural_tracing.datasets.dataset_defaults import (
    setdefault_rowcol_cond_dataset_config,
    validate_rowcol_cond_dataset_config,
)
from vesuvius.neural_tracing.datasets.dataset_rowcol_cond import EdtSegDataset
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.tifxyz import Tifxyz


PATCH_SIZE = (64, 64, 64)
TEST_TIFXYZ_ROOT = Path(__file__).resolve().parent / "test_tifxyzs"


def build_default_training_augmentation():
    return create_training_transforms(
        patch_size=PATCH_SIZE,
        no_spatial=False,
        no_scaling=False,
        only_spatial_and_intensity=False,
    )


def set_all_rng_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_real_test_segments(
    num_segments: int,
    *,
    collection: str = "PHerc0343p",
    retarget_factor: int = 1,
):
    if num_segments < 1:
        raise ValueError(f"num_segments must be >= 1, got {num_segments}")
    if retarget_factor < 1:
        raise ValueError(f"retarget_factor must be >= 1, got {retarget_factor}")

    if collection:
        root = TEST_TIFXYZ_ROOT / str(collection)
    else:
        root = TEST_TIFXYZ_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Missing tifxyz test directory: {root}")

    surfaces = list(tifxyz.load_folder(root, recursive=True))
    surfaces.sort(key=lambda s: str(Path(getattr(s, "path", ""))))

    selected = []
    for surface in surfaces:
        surface.use_stored_resolution()
        if not np.any(surface._valid_mask):
            continue
        seg = surface.retarget(retarget_factor) if int(retarget_factor) != 1 else surface
        selected.append(seg)
        if len(selected) >= num_segments:
            break

    if len(selected) < num_segments:
        raise RuntimeError(
            f"Requested {num_segments} real surfaces from {root}, found {len(selected)}"
        )
    return selected


def make_temp_test_volume_zarr(
    tmp_path: Path,
    *,
    shape=(16, 16, 16),
    scale: int = 0,
    dtype=np.float32,
):
    volume_path = Path(tmp_path) / "test_volume.zarr"
    group = zarr.open_group(str(volume_path), mode="w")
    group.create_dataset(
        str(int(scale)),
        shape=tuple(int(x) for x in shape),
        dtype=dtype,
        fill_value=0,
    )
    return volume_path


def build_rowcol_patchfinding_config(
    *,
    volume_path,
    segments_path,
    volume_scale: int = 0,
    sample_mode: str = "wrap",
    require_all_valid_in_bbox: bool = False,
    overrides: dict | None = None,
):
    config = {
        "verbose": False,
        "crop_size": list(PATCH_SIZE),
        "sample_mode": str(sample_mode),
        "cond_percent": [0.35, 0.35],
        "use_triplet_wrap_displacement": False,
        "use_sdt": False,
        "use_heatmap_targets": False,
        "use_segmentation": False,
        "use_other_wrap_cond": False,
        "validate_result_tensors": True,
        "force_recompute_patches": True,
        "overlap_fraction": 0.0,
        "min_span_ratio": 0.1,
        "edge_touch_frac": 0.01,
        "edge_touch_min_count": 1,
        "edge_touch_pad": 0,
        "min_points_per_wrap": 8,
        "scale_normalize_patch_counts": False,
        "bbox_pad_2d": 0,
        "require_all_valid_in_bbox": bool(require_all_valid_in_bbox),
        "skip_chunk_if_any_invalid": False,
        "inner_bbox_fraction": 1.0,
        "chunk_pad": 0.0,
        "datasets": [
            {
                "volume_path": str(volume_path),
                "volume_scale": int(volume_scale),
                "segments_path": str(segments_path),
            }
        ],
    }
    if overrides:
        config.update(dict(overrides))

    setdefault_rowcol_cond_dataset_config(config)
    validate_rowcol_cond_dataset_config(config)
    return config


def _find_all_valid_square_bbox(valid_mask: np.ndarray, min_side: int = 8, max_side: int = 24):
    valid = np.asarray(valid_mask, dtype=bool)
    h, w = valid.shape
    ii = np.pad(valid.astype(np.int32), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    max_test_side = int(min(max_side, h, w))
    for side in range(max_test_side, min_side - 1, -1):
        area = side * side
        sums = ii[side:, side:] - ii[:-side, side:] - ii[side:, :-side] + ii[:-side, :-side]
        ys, xs = np.where(sums == area)
        if ys.size == 0:
            continue
        mid = int(ys.size // 2)
        r0 = int(ys[mid])
        c0 = int(xs[mid])
        r1 = r0 + side - 1
        c1 = c0 + side - 1
        return (r0, r1, c0, c1)
    return None


@lru_cache(maxsize=1)
def _load_candidate_surfaces():
    if not TEST_TIFXYZ_ROOT.exists():
        raise FileNotFoundError(f"Missing test tifxyz directory: {TEST_TIFXYZ_ROOT}")

    candidates = []
    for surface in tifxyz.load_folder(TEST_TIFXYZ_ROOT, recursive=True):
        surface.use_stored_resolution()
        bbox = _find_all_valid_square_bbox(surface._valid_mask, min_side=8, max_side=24)
        if bbox is None:
            continue
        candidates.append((surface, bbox))
        if len(candidates) >= 8:
            break

    if len(candidates) < 3:
        raise RuntimeError(
            f"Need at least 3 tifxyz surfaces with valid windows in {TEST_TIFXYZ_ROOT}, "
            f"found {len(candidates)}"
        )
    return tuple(candidates)


def _normalize_surfaces_to_patch(candidates, *, margin: float = 4.0):
    points = []
    for surface, bbox in candidates:
        r0, r1, c0, c1 = bbox
        valid = surface._valid_mask[r0:r1 + 1, c0:c1 + 1]
        x = surface._x[r0:r1 + 1, c0:c1 + 1]
        y = surface._y[r0:r1 + 1, c0:c1 + 1]
        z = surface._z[r0:r1 + 1, c0:c1 + 1]
        points.append(np.stack([z[valid], y[valid], x[valid]], axis=1))

    all_points = np.concatenate(points, axis=0)
    global_min = all_points.min(axis=0)
    global_max = all_points.max(axis=0)
    out_min = np.array([margin, margin, margin], dtype=np.float32)
    out_max = np.array([PATCH_SIZE[0] - 1 - margin, PATCH_SIZE[1] - 1 - margin, PATCH_SIZE[2] - 1 - margin], dtype=np.float32)
    scale = (out_max - out_min) / np.maximum(global_max - global_min, 1e-6)
    offset = out_min - (global_min * scale)

    normalized = []
    for surface, bbox in candidates:
        valid = surface._valid_mask
        x_new = np.where(valid, surface._x * scale[2] + offset[2], -1.0).astype(np.float32)
        y_new = np.where(valid, surface._y * scale[1] + offset[1], -1.0).astype(np.float32)
        z_new = np.where(valid, surface._z * scale[0] + offset[0], -1.0).astype(np.float32)
        normalized.append(
            (
                Tifxyz(
                    _x=x_new,
                    _y=y_new,
                    _z=z_new,
                    uuid=surface.uuid,
                    _scale=surface._scale,
                    bbox=(0.0, 0.0, 0.0, float(PATCH_SIZE[2] - 1), float(PATCH_SIZE[1] - 1), float(PATCH_SIZE[0] - 1)),
                    _mask=valid.copy(),
                    path=surface.path,
                    interp_method=surface.interp_method,
                    resolution="stored",
                ),
                bbox,
            )
        )
    return normalized


def _build_real_chunk_patch(num_wraps: int):
    candidates = list(_load_candidate_surfaces()[:num_wraps])
    normalized = _normalize_surfaces_to_patch(candidates)
    segments = [seg for seg, _ in normalized]
    wraps = []
    for wrap_idx, (segment, bbox) in enumerate(normalized):
        wraps.append(
            {
                "segment": segment,
                "bbox_2d": bbox,
                "wrap_id": wrap_idx,
                "segment_idx": wrap_idx,
            }
        )
    return ChunkPatch(
        chunk_id=(0, 0, 0),
        volume=np.zeros(PATCH_SIZE, dtype=np.float32),
        scale=0,
        world_bbox=(0.0, float(PATCH_SIZE[0]), 0.0, float(PATCH_SIZE[1]), 0.0, float(PATCH_SIZE[2])),
        wraps=wraps,
        segments=segments,
    )


def _make_default_aligned_config(*, use_triplet_wrap_displacement: bool):
    config = {
        "crop_size": list(PATCH_SIZE),
        "sample_mode": "wrap",
        "cond_percent": [0.35, 0.35],
        "use_triplet_wrap_displacement": bool(use_triplet_wrap_displacement),
        "use_sdt": False,
        "use_heatmap_targets": False,
        "use_segmentation": False,
        "use_other_wrap_cond": False,
        "validate_result_tensors": True,
    }
    setdefault_rowcol_cond_dataset_config(config)
    validate_rowcol_cond_dataset_config(config)
    if str(config["sample_mode"]).lower() != "wrap":
        raise RuntimeError(f"Expected default-aligned sample_mode='wrap', got {config['sample_mode']!r}")
    return config


def _make_patch_metadata(*, patch: ChunkPatch, use_triplet_wrap_displacement: bool):
    metadata = {
        "patches": [patch],
        "sample_index": [(0, 0)],
        "cond_percent": [0.35, 0.35],
        "sample_mode": "wrap",
        "use_triplet_wrap_displacement": bool(use_triplet_wrap_displacement),
        "triplet_neighbor_lookup": {},
        "triplet_lookup_stats": {},
        "triplet_overlap_kept_indices": tuple(),
    }
    if use_triplet_wrap_displacement:
        metadata["triplet_neighbor_lookup"] = {(0, 0): {"behind_wrap_idx": 1, "front_wrap_idx": 2}}
    return metadata


def _build_dataset(*, patch: ChunkPatch, augmentation, use_triplet_wrap_displacement: bool):
    config = _make_default_aligned_config(
        use_triplet_wrap_displacement=use_triplet_wrap_displacement,
    )
    metadata = _make_patch_metadata(
        patch=patch,
        use_triplet_wrap_displacement=use_triplet_wrap_displacement,
    )
    ds = EdtSegDataset(
        config=config,
        apply_augmentation=True,
        patch_metadata=metadata,
    )
    ds._augmentations = augmentation
    if str(ds.sample_mode).lower() != "wrap":
        raise RuntimeError(f"Dataset sample_mode must be 'wrap' for this test, got {ds.sample_mode!r}")
    return ds


def _attach_split_capture_hooks(ds: EdtSegDataset):
    orig_create_split_masks = ds.create_split_masks
    orig_create_split_targets = ds.create_split_targets

    def _create_split_masks(self, idx, patch_idx, wrap_idx):
        out = orig_create_split_masks(idx, patch_idx, wrap_idx)
        if out is not None:
            self._last_split_masks_pre_aug = {
                "vol": out["vol"].detach().clone(),
                "cond_gt": out["cond_gt"].detach().clone(),
                "masked_seg": out["masked_seg"].detach().clone(),
                "other_wraps": out["other_wraps"].detach().clone(),
            }
        return out

    def _create_split_targets(self, *, cond_seg_gt, masked_seg):
        self._last_split_target_inputs = {
            "cond_seg_gt": cond_seg_gt.detach().clone(),
            "masked_seg": masked_seg.detach().clone(),
        }
        return orig_create_split_targets(cond_seg_gt=cond_seg_gt, masked_seg=masked_seg)

    ds.create_split_masks = MethodType(_create_split_masks, ds)
    ds.create_split_targets = MethodType(_create_split_targets, ds)


def _attach_triplet_capture_hooks(ds: EdtSegDataset):
    orig_create_neighbor_masks = ds.create_neighbor_masks
    orig_create_neighbor_targets = ds.create_neighbor_targets

    def _create_neighbor_masks(self, idx, patch_idx, wrap_idx):
        out = orig_create_neighbor_masks(idx, patch_idx, wrap_idx)
        if out is not None:
            self._last_triplet_masks_pre_aug = {
                "vol": out["vol"].detach().clone(),
                "cond_gt": out["cond_gt"].detach().clone(),
                "behind_seg": out["behind_seg"].detach().clone(),
                "front_seg": out["front_seg"].detach().clone(),
            }
        return out

    def _create_neighbor_targets(
        self,
        *,
        cond_seg_gt,
        behind_seg,
        front_seg,
        idx,
        patch_idx,
        wrap_idx,
    ):
        self._last_triplet_target_inputs = {
            "cond_seg_gt": cond_seg_gt.detach().clone(),
            "behind_seg": behind_seg.detach().clone(),
            "front_seg": front_seg.detach().clone(),
        }
        return orig_create_neighbor_targets(
            cond_seg_gt=cond_seg_gt,
            behind_seg=behind_seg,
            front_seg=front_seg,
            idx=idx,
            patch_idx=patch_idx,
            wrap_idx=wrap_idx,
        )

    ds.create_neighbor_masks = MethodType(_create_neighbor_masks, ds)
    ds.create_neighbor_targets = MethodType(_create_neighbor_targets, ds)


def build_minimal_split_dataset(*, augmentation=None):
    if augmentation is None:
        augmentation = build_default_training_augmentation()
    ds = _build_dataset(
        patch=_build_real_chunk_patch(num_wraps=1),
        augmentation=augmentation,
        use_triplet_wrap_displacement=False,
    )
    _attach_split_capture_hooks(ds)
    return ds


def build_minimal_triplet_dataset(*, augmentation=None):
    if augmentation is None:
        augmentation = build_default_training_augmentation()
    ds = _build_dataset(
        patch=_build_real_chunk_patch(num_wraps=3),
        augmentation=augmentation,
        use_triplet_wrap_displacement=True,
    )
    _attach_triplet_capture_hooks(ds)
    return ds
