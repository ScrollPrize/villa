import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from tqdm import tqdm
import vc

try:
    import trimesh
except Exception:  # pragma: no cover - optional dependency at runtime
    trimesh = None

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency at runtime
    njit = None

from vesuvius.neural_tracing.datasets.common import (
    _read_volume_crop,
    voxelize_surface_grid_masked,
)
from vesuvius.neural_tracing.inference.displacement_tta import (
    TTA_MERGE_METHODS,
    TTA_TRANSFORM_MODES,
)
from vesuvius.neural_tracing.inference.displacement_helpers import load_model, predict_displacement
from vesuvius.neural_tracing.inference.generate_segment_cover_bboxes import (
    _generate_segment_cover_records,
    _serialize_bbox_record,
)
from vesuvius.neural_tracing.heatmap_single_point.tifxyz import save_tifxyz
from vesuvius.tifxyz import read_tifxyz

_TTA_TRANSFORM_ALIASES = {
    "mirror+rot90": "rotate3",
}
_TTA_MERGE_ALIASES = {
    "vector_mean": "mean",
    "vector_median": "median",
}
_COPY_ARG_ALIASES = {
    "dense_checkpoint_path": "checkpoint_path",
    "volume_zarr": "volume_path",
    "tifxyz_out_dir": "out_dir",
}
_COPY_ARG_TO_CLI = {
    "tifxyz_path": "--tifxyz-path",
    "volume_path": "--volume-path",
    "checkpoint_path": "--checkpoint-path",
    "device": "--device",
    "volume_scale": "--volume-scale",
    "volume_cache_dir": "--volume-cache-dir",
    "volume_cache_retry_seconds": "--volume-cache-retry-seconds",
    "volume_chunk_cache_gb": "--volume-chunk-cache-gb",
    "compile_mode": "--compile-mode",
    "crop_size": "--crop-size",
    "batch_size": "--batch-size",
    "crop_input_workers": "--crop-input-workers",
    "crop_prefetch_batches": "--crop-prefetch-batches",
    "bbox_overlap": "--bbox-overlap",
    "bbox_prune_max_remove_per_band": "--bbox-prune-max-remove-per-band",
    "bbox_band_workers": "--bbox-band-workers",
    "tta_merge_method": "--tta-merge-method",
    "tta_transform": "--tta-transform",
    "tta_outlier_drop_thresh": "--tta-outlier-drop-thresh",
    "tta_outlier_drop_min_keep": "--tta-outlier-drop-min-keep",
    "tta_batch_size": "--tta-batch-size",
    "disp_sample_radius": "--disp-sample-radius",
    "disp_sample_spacing": "--disp-sample-spacing",
    "disp_sample_min_count": "--disp-sample-min-count",
    "disp_sample_reduce": "--disp-sample-reduce",
    "merge_temp_dir": "--merge-temp-dir",
    "out_dir": "--out-dir",
    "output_prefix": "--output-prefix",
    "iterations": "--iterations",
    "iter_direction": "--iter-direction",
    "tifxyz_step_size": "--tifxyz-step-size",
    "tifxyz_voxel_size_um": "--tifxyz-voxel-size-um",
}
_DEFAULT_VOLUME_CHUNK_CACHE_GB = 20.0


class _VcVolumeLevel:
    """Small array adapter over vc.Volume for the crop reader used here."""

    def __init__(self, volume, level):
        self._volume = volume
        self._level = int(level)
        self.shape = tuple(int(v) for v in volume.shape_at(self._level))
        self.dtype = np.dtype(volume.dtype)
        self.chunks = tuple(int(v) for v in volume.chunk_shape(self._level))

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 3:
            raise TypeError(f"vc volume reads require 3 ZYX slices, got {key!r}")
        starts = []
        stops = []
        for axis, item in enumerate(key):
            if not isinstance(item, slice):
                raise TypeError(f"vc volume reads require slices, got axis {axis}: {item!r}")
            step = 1 if item.step is None else int(item.step)
            if step != 1:
                raise ValueError(f"vc volume reads only support unit-step slices, got axis {axis} step={step}")
            start = 0 if item.start is None else int(item.start)
            stop = self.shape[axis] if item.stop is None else int(item.stop)
            starts.append(start)
            stops.append(stop)
        read_shape = [stop - start for start, stop in zip(starts, stops)]
        if any(v < 0 for v in read_shape):
            raise ValueError(f"Invalid vc volume slice {key!r} for shape {self.shape}")
        if any(v == 0 for v in read_shape):
            return np.empty(tuple(read_shape), dtype=self.dtype)
        return np.asarray(
            self._volume.read_zyx(starts, read_shape, level=self._level, missing_policy="all_fill")
        )


def _open_vc_volume_level(volume_path, volume_scale, cache_dir, chunk_cache_gb):
    path = str(volume_path)
    cache_bytes = int(round(float(chunk_cache_gb) * (1024 ** 3)))

    def _set_cache_budget(volume):
        volume.set_cache_budget(cache_bytes)
        return volume

    target_level = int(volume_scale)
    if path.startswith(("http://", "https://", "s3://")):
        volume = _set_cache_budget(vc.Volume.open_url(path, cache_root=str(cache_dir)))
    else:
        try:
            volume = _set_cache_budget(vc.Volume.open(path))
        except RuntimeError as exc:
            scale_path = Path(path) / str(target_level)
            if not scale_path.exists():
                raise
            try:
                volume = _set_cache_budget(vc.Volume.open(str(scale_path)))
            except RuntimeError:
                raise exc
            return _VcVolumeLevel(volume, 0), target_level

    if volume.has_scale_level(target_level):
        level = target_level
    else:
        present_levels = [int(v) for v in volume.present_scale_levels()]
        if not present_levels:
            raise RuntimeError(f"vc volume {path!r} does not report any scale levels")
        level = min(present_levels, key=lambda v: abs(v - target_level))

    return _VcVolumeLevel(volume, level), level


def _empty_uv(dtype=np.int64):
    return np.zeros((0, 2), dtype=dtype)


def _empty_world(dtype=np.float32):
    return np.zeros((0, 3), dtype=dtype)


def _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32):
    return _empty_uv(dtype=uv_dtype), _empty_world(dtype=world_dtype)


def _normalize_scale_rc(scale_rc, source_label):
    if scale_rc is None:
        return None
    if not isinstance(scale_rc, (list, tuple)) or len(scale_rc) != 2:
        raise RuntimeError(f"Expected {source_label} scale as [scale_y, scale_x], got {scale_rc!r}")
    scale_y = float(scale_rc[0])
    scale_x = float(scale_rc[1])
    if (not np.isfinite(scale_y)) or (not np.isfinite(scale_x)) or scale_y <= 0.0 or scale_x <= 0.0:
        raise RuntimeError(f"Invalid {source_label} scale: {scale_rc!r}")
    return (scale_y, scale_x)


def _read_tifxyz_scale_from_meta(tifxyz_path):
    if tifxyz_path is None:
        return None
    meta_path = Path(tifxyz_path) / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(
            f"Unable to resolve stored tifxyz density scale: missing meta.json at {meta_path}"
        )
    with open(meta_path, "rt") as meta_fp:
        meta = json.load(meta_fp)
    return _normalize_scale_rc(meta.get("scale"), source_label=str(meta_path))


def _scale_to_subsample_stride(scale):
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid tifxyz scale: {scale}")
    return max(1, int(round(1.0 / scale)))


def resolve_tifxyz_params(args, model_config, volume_scale, input_scale=None):
    tifxyz_step_size = args.tifxyz_step_size
    tifxyz_voxel_size_um = args.tifxyz_voxel_size_um
    stored_scale_rc = _read_tifxyz_scale_from_meta(getattr(args, "tifxyz_path", None))
    if stored_scale_rc is None and input_scale is not None:
        stored_scale_rc = _normalize_scale_rc(input_scale, source_label="input_scale")

    stored_step_size = None
    if stored_scale_rc is not None:
        step_y = _scale_to_subsample_stride(stored_scale_rc[0])
        step_x = _scale_to_subsample_stride(stored_scale_rc[1])
        if step_y != step_x:
            raise RuntimeError(
                "infer_rowcol_triplet_wraps requires isotropic stored tifxyz scale for output, "
                f"but got stored scale={stored_scale_rc!r} -> steps ({step_y}, {step_x})."
            )
        stored_step_size = int(step_y)

    if tifxyz_step_size is None:
        if stored_step_size is not None:
            tifxyz_step_size = stored_step_size
        elif model_config is not None:
            tifxyz_step_size = model_config.get(
                "step_size",
                model_config.get("heatmap_step_size", 10),
            )
        else:
            tifxyz_step_size = 10

    if tifxyz_voxel_size_um is None:
        meta_path = os.path.join(args.volume_path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "rt") as meta_fp:
                tifxyz_voxel_size_um = json.load(meta_fp).get("voxelsize", None)
    if tifxyz_voxel_size_um is None:
        tifxyz_voxel_size_um = 8.24

    tifxyz_step_size = int(round(float(tifxyz_step_size)))
    if stored_step_size is not None and tifxyz_step_size != stored_step_size:
        raise RuntimeError(
            "--tifxyz-step-size cannot override the stored input tifxyz scale for infer_rowcol_triplet_wraps. "
            f"Expected step_size={stored_step_size} for scale={stored_scale_rc}, got {tifxyz_step_size}."
        )

    return tifxyz_step_size, tifxyz_voxel_size_um, stored_scale_rc


def _resolve_segment_volume(segment, volume_scale=None):
    volume = segment.volume
    if hasattr(volume, "read_zyx") and hasattr(volume, "shape_at"):
        target_level = 0 if volume_scale is None else int(volume_scale)
        return _VcVolumeLevel(volume, target_level)
    if hasattr(volume, "keys") and not hasattr(volume, "shape"):
        target_level = None
        if volume_scale is not None:
            target_level = int(volume_scale)
        else:
            extra = getattr(segment, "extra", None)
            if isinstance(extra, dict):
                for key in ("volume_scale", "vol_scale", "zarr_level", "volume_level", "level"):
                    if key in extra:
                        try:
                            target_level = int(extra[key])
                            break
                        except (TypeError, ValueError):
                            continue
        if target_level is None:
            target_level = 0
        level_key = str(target_level)
        if level_key in volume:
            return volume[level_key]
        numeric_levels = sorted([k for k in volume.keys() if k.isdigit()], key=int)
        if numeric_levels:
            level_ints = [int(k) for k in numeric_levels]
            nearest = min(level_ints, key=lambda v: abs(v - target_level))
            return volume[str(nearest)]
    return volume


def _bbox_to_min_corner_and_bounds_array(bbox):
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    min_corner = np.floor([z_min, y_min, x_min]).astype(np.int64)
    bounds_array = np.asarray(
        [
            [z_min, y_min, x_min],
            [z_max, y_max, x_max],
        ],
        dtype=np.int32,
    )
    return min_corner, bounds_array


def _as_displacement_tensor(disp):
    if torch.is_tensor(disp):
        disp_t = disp.detach()
        if disp_t.ndim == 4:
            disp_t = disp_t.unsqueeze(0)
        if disp_t.ndim != 5 or disp_t.shape[0] != 1 or disp_t.shape[1] < 3:
            raise RuntimeError(f"Expected displacement tensor with shape [1, 3+, D, H, W], got {tuple(disp_t.shape)}")
        return disp_t[:, :3].to(dtype=torch.float32, device="cpu").contiguous()

    disp_np = np.asarray(disp, dtype=np.float32)
    if disp_np.ndim != 4 or disp_np.shape[0] < 3:
        raise RuntimeError(f"Expected displacement array with shape [3+, D, H, W], got {tuple(disp_np.shape)}")
    return torch.from_numpy(disp_np[:3]).unsqueeze(0).contiguous()


def _coords_local_to_grid(coords_t, d, h, w):
    d_denom = max(int(d) - 1, 1)
    h_denom = max(int(h) - 1, 1)
    w_denom = max(int(w) - 1, 1)
    coords_norm = coords_t.clone()
    coords_norm[:, 0] = 2.0 * coords_norm[:, 0] / float(d_denom) - 1.0
    coords_norm[:, 1] = 2.0 * coords_norm[:, 1] / float(h_denom) - 1.0
    coords_norm[:, 2] = 2.0 * coords_norm[:, 2] / float(w_denom) - 1.0
    return coords_norm[:, [2, 1, 0]].view(1, -1, 1, 1, 3)


def _displacement_sample_offsets_rc(radius, spacing, device="cpu"):
    radius = float(radius)
    spacing = float(spacing)
    if radius <= 0.0:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)
    if spacing <= 0.0:
        raise ValueError(f"disp_sample_spacing must be > 0, got {spacing}")

    axis = np.arange(-radius, radius + (0.5 * spacing), spacing, dtype=np.float32)
    rr, cc = np.meshgrid(axis, axis, indexing="ij")
    offsets = np.stack([rr, cc], axis=-1).reshape(-1, 2)
    center = np.zeros((1, 2), dtype=np.float32)
    non_center = np.linalg.norm(offsets, axis=1) > 1e-6
    offsets = np.concatenate([center, offsets[non_center]], axis=0)
    return torch.from_numpy(offsets.astype(np.float32, copy=False)).to(device=device)


def _reduce_sampled_displacements(sampled_t, valid_t, reduce, min_count):
    reduce = str(reduce).lower()
    min_count = max(1, int(min_count))
    n_points = int(sampled_t.shape[0])
    out_t = torch.zeros((n_points, 3), dtype=sampled_t.dtype, device=sampled_t.device)
    counts_t = valid_t.sum(dim=1)
    enough_t = counts_t >= int(min_count)
    if not bool(enough_t.any()):
        return out_t, enough_t

    if reduce == "mean":
        weighted_t = sampled_t * valid_t.unsqueeze(-1).to(dtype=sampled_t.dtype)
        denom_t = counts_t.clamp(min=1).to(dtype=sampled_t.dtype).unsqueeze(-1)
        out_t[enough_t] = weighted_t.sum(dim=1)[enough_t] / denom_t[enough_t]
        return out_t, enough_t

    if reduce == "median":
        enough_idx = torch.nonzero(enough_t, as_tuple=False).flatten()
        for idx_t in enough_idx:
            idx = int(idx_t.item())
            vals_t = sampled_t[idx, valid_t[idx]]
            out_t[idx] = torch.median(vals_t, dim=0).values
        return out_t, enough_t

    raise ValueError(f"Unknown displacement sample reducer: {reduce!r}")


def _coords_local_tensor(coords_local, device="cpu"):
    if coords_local is None:
        return None
    if torch.is_tensor(coords_local):
        return coords_local.to(dtype=torch.float32, device=device)
    coords_np = np.asarray(coords_local, dtype=np.float32)
    if coords_np.ndim != 2 or coords_np.shape[1] != 3:
        return None
    return torch.from_numpy(coords_np).to(device=device)


def _sample_displacement_at_local_coords_tensor(disp_t, coords_t):
    if coords_t is None or coords_t.ndim != 2 or coords_t.shape[1] != 3 or coords_t.shape[0] == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.bool),
        )

    _, _, d, h, w = disp_t.shape
    valid_mask_t = (
        (coords_t[:, 0] >= 0.0) & (coords_t[:, 0] <= float(d - 1)) &
        (coords_t[:, 1] >= 0.0) & (coords_t[:, 1] <= float(h - 1)) &
        (coords_t[:, 2] >= 0.0) & (coords_t[:, 2] <= float(w - 1))
    )

    sampled_disp_t = torch.zeros((coords_t.shape[0], 3), dtype=disp_t.dtype, device=disp_t.device)
    if bool(valid_mask_t.any()):
        valid_coords_t = coords_t[valid_mask_t]
        grid = _coords_local_to_grid(valid_coords_t, d=d, h=h, w=w)
        sampled_valid_t = F.grid_sample(
            disp_t,
            grid,
            mode="bilinear",
            align_corners=True,
        ).view(3, -1).permute(1, 0)
        sampled_disp_t[valid_mask_t] = sampled_valid_t
    return sampled_disp_t, valid_mask_t


def _sample_local_surface_coords_from_uv_tensor(uv_rc, local_zyx, query_uv_t, device="cpu"):
    uv_arr = np.asarray(uv_rc, dtype=np.int64)
    local_arr = np.asarray(local_zyx, dtype=np.float32)
    if (
        uv_arr.ndim != 2 or uv_arr.shape[1] != 2 or
        local_arr.ndim != 2 or local_arr.shape[1] != 3 or
        uv_arr.shape[0] == 0 or uv_arr.shape[0] != local_arr.shape[0]
    ):
        return (
            torch.zeros((0, 3), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.bool, device=device),
        )

    finite = np.isfinite(local_arr).all(axis=1)
    if not bool(finite.any()):
        return (
            torch.zeros((query_uv_t.shape[0], 3), dtype=torch.float32, device=device),
            torch.zeros((query_uv_t.shape[0],), dtype=torch.bool, device=device),
        )
    uv_arr = uv_arr[finite]
    local_arr = local_arr[finite]

    r_min = int(uv_arr[:, 0].min())
    c_min = int(uv_arr[:, 1].min())
    r_max = int(uv_arr[:, 0].max())
    c_max = int(uv_arr[:, 1].max())
    h = int(r_max - r_min + 1)
    w = int(c_max - c_min + 1)
    if h <= 0 or w <= 0:
        return (
            torch.zeros((query_uv_t.shape[0], 3), dtype=torch.float32, device=device),
            torch.zeros((query_uv_t.shape[0],), dtype=torch.bool, device=device),
        )

    grid_np = np.zeros((h, w, 3), dtype=np.float32)
    mask_np = np.zeros((h, w), dtype=np.float32)
    rr = (uv_arr[:, 0] - r_min).astype(np.int64, copy=False)
    cc = (uv_arr[:, 1] - c_min).astype(np.int64, copy=False)
    grid_np[rr, cc] = local_arr
    mask_np[rr, cc] = 1.0

    value_t = torch.from_numpy(grid_np.transpose(2, 0, 1)).unsqueeze(0).contiguous().to(device=device)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).contiguous().to(device=device)
    q_t = query_uv_t.to(dtype=torch.float32, device=device)
    q_r = q_t[:, 0] - float(r_min)
    q_c = q_t[:, 1] - float(c_min)
    y_norm = (2.0 * q_r / float(max(h - 1, 1))) - 1.0
    x_norm = (2.0 * q_c / float(max(w - 1, 1))) - 1.0
    query_grid_t = torch.stack([x_norm, y_norm], dim=1).view(1, -1, 1, 2)
    with torch.no_grad():
        sampled_num_t = F.grid_sample(
            value_t,
            query_grid_t,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_den_t = F.grid_sample(
            mask_t,
            query_grid_t,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
    sampled_num_t = sampled_num_t.view(3, -1).permute(1, 0).contiguous()
    sampled_den_t = sampled_den_t.view(-1)
    can_sample_t = sampled_den_t > 1e-6
    sampled_coords_t = torch.zeros_like(sampled_num_t)
    if bool(can_sample_t.any()):
        sampled_coords_t[can_sample_t] = sampled_num_t[can_sample_t] / sampled_den_t[can_sample_t, None]
    return sampled_coords_t, can_sample_t


def _sample_trilinear_displacement_stack_tensor(
    disp_t,
    coords_local,
    uv_rc=None,
    sample_radius=0.0,
    sample_spacing=1.0,
    sample_min_count=1,
    sample_reduce="mean",
):
    device = disp_t.device
    coords_t = _coords_local_tensor(coords_local, device=device)
    if coords_t is None or coords_t.ndim != 2 or coords_t.shape[1] != 3 or coords_t.shape[0] == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.bool),
        )

    if float(sample_radius) <= 0.0:
        return _sample_displacement_at_local_coords_tensor(disp_t, coords_t)

    if uv_rc is None:
        offsets_t = _displacement_sample_offsets_rc(
            radius=float(sample_radius),
            spacing=float(sample_spacing),
            device=device,
        ).to(dtype=torch.float32, device=device)
        query_coords_t = coords_t[:, None, :].repeat(1, offsets_t.shape[0], 1)
        query_coords_t[:, :, 1] += offsets_t[None, :, 0]
        query_coords_t[:, :, 2] += offsets_t[None, :, 1]
        sampled_flat_t, valid_flat_t = _sample_displacement_at_local_coords_tensor(
            disp_t,
            query_coords_t.reshape(-1, 3),
        )
        sampled_stack_t = sampled_flat_t.view(coords_t.shape[0], offsets_t.shape[0], 3)
        valid_stack_t = valid_flat_t.view(coords_t.shape[0], offsets_t.shape[0])
        return _reduce_sampled_displacements(
            sampled_stack_t,
            valid_stack_t,
            reduce=sample_reduce,
            min_count=int(sample_min_count),
        )

    uv_t = torch.from_numpy(np.asarray(uv_rc, dtype=np.float32)).to(device=device)
    if uv_t.ndim != 2 or uv_t.shape[1] != 2 or uv_t.shape[0] != coords_t.shape[0]:
        return _sample_displacement_at_local_coords_tensor(disp_t, coords_t)

    offsets_t = _displacement_sample_offsets_rc(
        radius=float(sample_radius),
        spacing=float(sample_spacing),
        device=device,
    ).to(dtype=torch.float32, device=device)
    query_uv_t = (uv_t[:, None, :] + offsets_t[None, :, :]).reshape(-1, 2)
    sample_coords_t, surface_valid_t = _sample_local_surface_coords_from_uv_tensor(
        uv_rc=uv_rc,
        local_zyx=coords_local,
        query_uv_t=query_uv_t,
        device=device,
    )
    sampled_flat_t, disp_valid_t = _sample_displacement_at_local_coords_tensor(disp_t, sample_coords_t)
    valid_flat_t = surface_valid_t & disp_valid_t

    sampled_stack_t = sampled_flat_t.view(coords_t.shape[0], offsets_t.shape[0], 3)
    valid_stack_t = valid_flat_t.view(coords_t.shape[0], offsets_t.shape[0])
    sampled_disp_t, valid_mask_t = _reduce_sampled_displacements(
        sampled_stack_t,
        valid_stack_t,
        reduce=sample_reduce,
        min_count=int(sample_min_count),
    )
    return sampled_disp_t, valid_mask_t


def _sample_trilinear_displacement_stack(
    disp,
    coords_local,
    uv_rc=None,
    sample_radius=0.0,
    sample_spacing=1.0,
    sample_min_count=1,
    sample_reduce="mean",
):
    if coords_local is None or len(coords_local) == 0:
        return (
            _empty_world(dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    disp_t = _as_displacement_tensor(disp)
    sampled_disp_t, valid_mask_t = _sample_trilinear_displacement_stack_tensor(
        disp_t,
        coords_local,
        uv_rc=uv_rc,
        sample_radius=sample_radius,
        sample_spacing=sample_spacing,
        sample_min_count=sample_min_count,
        sample_reduce=sample_reduce,
    )
    return (
        sampled_disp_t.numpy().astype(np.float32, copy=False),
        valid_mask_t.numpy().astype(bool, copy=False),
    )


def _surface_to_stored_uv_samples_lattice(
    grid,
    valid,
    uv_offset,
    sub_r,
    sub_c,
    phase_rc=(0, 0),
):
    grid = np.asarray(grid, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    sub_r = max(1, int(sub_r))
    sub_c = max(1, int(sub_c))
    phase_r = int(phase_rc[0])
    phase_c = int(phase_rc[1])

    empty_meta = {
        "mode": "lattice_bilinear_torch",
        "stride_rc": [sub_r, sub_c],
        "phase_rc": [phase_r, phase_c],
        "n_full_valid": 0,
        "n_stored_valid": 0,
    }
    if grid.ndim != 3 or grid.shape[-1] != 3:
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32) + (empty_meta,)

    h, w = grid.shape[:2]
    support = valid & np.isfinite(grid).all(axis=2)
    n_full_valid = int(support.sum())
    if n_full_valid < 1:
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32) + (empty_meta,)

    r_abs0 = int(uv_offset[0])
    c_abs0 = int(uv_offset[1])
    r_abs1 = r_abs0 + h - 1
    c_abs1 = c_abs0 + w - 1
    s_r_min = int(np.ceil((r_abs0 - phase_r) / float(sub_r)))
    s_r_max = int(np.floor((r_abs1 - phase_r) / float(sub_r)))
    s_c_min = int(np.ceil((c_abs0 - phase_c) / float(sub_c)))
    s_c_max = int(np.floor((c_abs1 - phase_c) / float(sub_c)))
    if s_r_max < s_r_min or s_c_max < s_c_min:
        meta = dict(empty_meta)
        meta["n_full_valid"] = n_full_valid
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32) + (meta,)

    stored_rows = np.arange(s_r_min, s_r_max + 1, dtype=np.int64)
    stored_cols = np.arange(s_c_min, s_c_max + 1, dtype=np.int64)
    sr, sc = np.meshgrid(stored_rows, stored_cols, indexing="ij")
    uv_q = np.stack([sr.reshape(-1), sc.reshape(-1)], axis=-1).astype(np.int64, copy=False)

    q_r_abs = uv_q[:, 0].astype(np.int64, copy=False) * int(sub_r) + int(phase_r)
    q_c_abs = uv_q[:, 1].astype(np.int64, copy=False) * int(sub_c) + int(phase_c)
    q_r = (q_r_abs - int(uv_offset[0])).astype(np.float32, copy=False)
    q_c = (q_c_abs - int(uv_offset[1])).astype(np.float32, copy=False)

    in_grid = (
        (q_r >= 0.0) &
        (q_r <= float(max(h - 1, 0))) &
        (q_c >= 0.0) &
        (q_c <= float(max(w - 1, 0)))
    )
    if not bool(in_grid.any()):
        meta = dict(empty_meta)
        meta["n_full_valid"] = n_full_valid
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32) + (meta,)

    uv_q = uv_q[in_grid].astype(np.int64, copy=False)
    q_r = q_r[in_grid].astype(np.float32, copy=False)
    q_c = q_c[in_grid].astype(np.float32, copy=False)

    denom_r = float(max(h - 1, 1))
    denom_c = float(max(w - 1, 1))
    value_np = np.where(support[..., None], grid, 0.0).astype(np.float32, copy=False)
    value_t = torch.from_numpy(value_np.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    mask_t = torch.from_numpy(support.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0).contiguous()

    def _sample_world_and_mask(qr, qc):
        y_norm = (2.0 * qr / denom_r) - 1.0
        x_norm = (2.0 * qc / denom_c) - 1.0
        query_grid = np.stack([x_norm, y_norm], axis=-1).astype(np.float32, copy=False)
        query_t = torch.from_numpy(query_grid.reshape(1, -1, 1, 2)).contiguous()
        with torch.no_grad():
            sampled_num_t = F.grid_sample(
                value_t,
                query_t,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled_den_t = F.grid_sample(
                mask_t,
                query_t,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
        sampled_num = (
            sampled_num_t[0, :, :, 0]
            .permute(1, 0)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        sampled_den = sampled_den_t[0, 0, :, 0].cpu().numpy().astype(np.float32, copy=False)
        return sampled_num, sampled_den

    sampled_num, sampled_den = _sample_world_and_mask(q_r, q_c)
    can_sample = sampled_den > 1e-6
    n_anchor_sampled = int(can_sample.sum())
    n_fallback_tried = 0
    n_fallback_recovered = 0

    if not bool(can_sample.all()):
        fallback_idx = np.nonzero(~can_sample)[0]
        n_fallback_tried = int(fallback_idx.shape[0])
        if n_fallback_tried > 0:
            q_r_fallback = (q_r[fallback_idx] + (0.5 * float(sub_r - 1))).astype(np.float32, copy=False)
            q_c_fallback = (q_c[fallback_idx] + (0.5 * float(sub_c - 1))).astype(np.float32, copy=False)
            sampled_num_fb, sampled_den_fb = _sample_world_and_mask(q_r_fallback, q_c_fallback)
            can_fb = sampled_den_fb > 1e-6
            if bool(can_fb.any()):
                recovered_idx = fallback_idx[can_fb]
                sampled_num[recovered_idx] = sampled_num_fb[can_fb]
                sampled_den[recovered_idx] = sampled_den_fb[can_fb]
                can_sample[recovered_idx] = True
                n_fallback_recovered = int(can_fb.sum())

    if not bool(can_sample.any()):
        meta = dict(empty_meta)
        meta["n_full_valid"] = n_full_valid
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32) + (meta,)

    uv_keep = uv_q[can_sample].astype(np.int64, copy=False)
    pts_bilinear = (
        sampled_num[can_sample] / sampled_den[can_sample, None]
    ).astype(np.float32, copy=False)

    projection_meta = {
        "mode": "lattice_bilinear_torch",
        "stride_rc": [sub_r, sub_c],
        "phase_rc": [phase_r, phase_c],
        "n_full_valid": n_full_valid,
        "n_anchor_sampled": n_anchor_sampled,
        "n_fallback_tried": n_fallback_tried,
        "n_fallback_recovered": n_fallback_recovered,
        "n_stored_valid": int(uv_keep.shape[0]),
    }
    return uv_keep, pts_bilinear, projection_meta


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run triplet wrap displacement inference and save _front/_back tifxyz outputs."
    )
    cpu_count = int(os.cpu_count() or 1)
    max_band_workers = max(1, cpu_count // 2)
    parser.add_argument("--tifxyz-path", type=str, required=True)
    parser.add_argument("--volume-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--volume-scale", type=int, default=0)
    parser.add_argument("--volume-cache-dir", type=str, default="/tmp/vesuvius-volume-cache")
    parser.add_argument("--volume-cache-retry-seconds", type=float, default=60.0)
    parser.add_argument("--volume-chunk-cache-gb", type=float, default=_DEFAULT_VOLUME_CHUNK_CACHE_GB)
    parser.add_argument("--compile", dest="compile_model", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile_model", action="store_false")
    parser.add_argument("--compile-mode", type=str, default="default")

    parser.add_argument(
        "--crop-size",
        type=int,
        nargs=3,
        default=None,
        metavar=("D", "H", "W"),
        help="Crop size for bbox inference. Defaults to checkpoint crop_size (or 128^3).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--crop-input-workers",
        type=int,
        default=1,
        help="Number of worker threads used to prepare per-bbox crop inputs.",
    )
    parser.add_argument(
        "--crop-prefetch-batches",
        type=int,
        default=2,
        help="Number of prepared crop batches to keep queued ahead of GPU inference.",
    )

    parser.add_argument("--bbox-overlap", type=float, default=0.0)
    parser.add_argument("--bbox-prune", dest="bbox_prune", action="store_true")
    parser.add_argument("--no-bbox-prune", dest="bbox_prune", action="store_false")
    parser.set_defaults(bbox_prune=True)
    parser.add_argument("--bbox-prune-max-remove-per-band", type=int, default=None)
    parser.add_argument("--bbox-band-workers", type=int, default=max_band_workers)

    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument(
        "--tta-merge-method",
        type=str,
        default="vector_geomedian",
        choices=sorted(set(TTA_MERGE_METHODS).union(_TTA_MERGE_ALIASES.keys())),
    )
    parser.add_argument(
        "--tta-transform",
        type=str,
        default="mirror",
        choices=sorted(set(TTA_TRANSFORM_MODES).union(_TTA_TRANSFORM_ALIASES.keys())),
    )
    parser.add_argument("--tta-outlier-drop-thresh", type=float, default=1.25)
    parser.add_argument("--tta-outlier-drop-min-keep", type=int, default=4)
    parser.add_argument("--tta-batch-size", type=int, default=2)
    parser.add_argument(
        "--disp-sample-radius",
        type=float,
        default=1.0,
        help=(
            "Dense parameterized row/col radius around each surface point used when "
            "sampling predicted displacements. Use 0 for the legacy single trilinear sample."
        ),
    )
    parser.add_argument(
        "--disp-sample-spacing",
        type=float,
        default=1.0,
        help="Parameterized row/col spacing for the dense displacement sampling kernel.",
    )
    parser.add_argument(
        "--disp-sample-min-count",
        type=int,
        default=1,
        help="Minimum in-bounds samples required for a displacement estimate.",
    )
    parser.add_argument(
        "--disp-sample-reduce",
        type=str,
        default="mean",
        choices=("mean", "median"),
        help="Reducer used across the dense displacement sampling kernel.",
    )
    parser.add_argument(
        "--merge-temp-dir",
        type=str,
        default=None,
        help="Parent directory for temporary dense overlap-merge memmaps. Defaults to the system temp directory.",
    )

    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--save-original-copy", dest="save_original_copy", action="store_true")
    parser.add_argument("--no-save-original-copy", dest="save_original_copy", action="store_false")
    parser.set_defaults(save_original_copy=False)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Total number of iterative passes to run. Requires --iter-direction when provided.",
    )
    parser.add_argument(
        "--iter-direction",
        type=str,
        default=None,
        choices=("front", "back"),
        help="Output direction used as the next iteration input when --iterations is provided.",
    )
    parser.add_argument("--keep-previous-wrap", dest="keep_previous_wrap", action="store_true")
    parser.add_argument("--no-keep-previous-wrap", dest="keep_previous_wrap", action="store_false")
    parser.set_defaults(keep_previous_wrap=True)

    parser.add_argument("--tifxyz-step-size", type=int, default=None)
    parser.add_argument("--tifxyz-voxel-size-um", type=float, default=None)

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.crop_input_workers < 1:
        parser.error("--crop-input-workers must be >= 1")
    if args.crop_prefetch_batches < 1:
        parser.error("--crop-prefetch-batches must be >= 1")
    if args.bbox_band_workers < 1:
        parser.error("--bbox-band-workers must be >= 1")
    if args.bbox_band_workers > max_band_workers:
        parser.error(
            f"--bbox-band-workers must be <= {max_band_workers} "
            f"(half available CPUs; detected cpu_count={cpu_count})."
        )
    if args.bbox_overlap < 0.0 or args.bbox_overlap >= 1.0:
        parser.error("--bbox-overlap must satisfy 0.0 <= overlap < 1.0")
    if args.crop_size is not None and any(v < 1 for v in args.crop_size):
        parser.error("--crop-size values must be >= 1")
    if args.volume_cache_retry_seconds < 0.0:
        parser.error("--volume-cache-retry-seconds must be >= 0")
    if args.volume_chunk_cache_gb < 0.0:
        parser.error("--volume-chunk-cache-gb must be >= 0")
    if args.disp_sample_radius < 0.0:
        parser.error("--disp-sample-radius must be >= 0")
    if args.disp_sample_spacing <= 0.0:
        parser.error("--disp-sample-spacing must be > 0")
    if args.disp_sample_min_count < 1:
        parser.error("--disp-sample-min-count must be >= 1")
    if args.merge_temp_dir is not None:
        merge_temp_dir = Path(args.merge_temp_dir)
        if merge_temp_dir.exists() and not merge_temp_dir.is_dir():
            parser.error("--merge-temp-dir must be a directory when it already exists")
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be >= 1 when provided.")
    if args.iterations is not None and args.iter_direction is None:
        parser.error("--iter-direction is required when --iterations is provided.")
    if args.iterations is None and args.iter_direction is not None:
        parser.error("--iter-direction requires --iterations.")
    return args


def normalize_copy_args(copy_args):
    if not isinstance(copy_args, dict):
        raise RuntimeError(f"copy_args must be a dict, got {type(copy_args).__name__}")
    normalized = {}
    for key, value in copy_args.items():
        key_norm = str(key).replace("-", "_")
        normalized[_COPY_ARG_ALIASES.get(key_norm, key_norm)] = value
    return normalized


def _append_cli_arg(argv, flag, value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        argv.append(flag)
        argv.extend(str(v) for v in value)
        return
    argv.extend([flag, str(value)])


def _copy_args_to_argv(copy_args):
    copy_args = normalize_copy_args(copy_args)
    argv = []

    for key, flag in _COPY_ARG_TO_CLI.items():
        if key not in copy_args:
            continue
        value = copy_args.get(key)
        if value is None:
            continue
        _append_cli_arg(argv, flag, value)

    if "tta" in copy_args and bool(copy_args.get("tta")) is False:
        argv.append("--no-tta")
    if "compile_model" in copy_args and bool(copy_args.get("compile_model")) is False:
        argv.append("--no-compile")
    if "bbox_prune" in copy_args and bool(copy_args.get("bbox_prune")) is False:
        argv.append("--no-bbox-prune")
    if "save_original_copy" in copy_args and bool(copy_args.get("save_original_copy")):
        argv.append("--save-original-copy")
    if "keep_previous_wrap" in copy_args and bool(copy_args.get("keep_previous_wrap")) is False:
        argv.append("--no-keep-previous-wrap")
    if "verbose" in copy_args and bool(copy_args.get("verbose")):
        argv.append("--verbose")

    return argv


def _log(verbose, msg):
    if verbose:
        print(msg)


def _canonicalize_tta_settings(args):
    raw_transform = str(args.tta_transform).strip().lower()
    canonical_transform = _TTA_TRANSFORM_ALIASES.get(raw_transform, raw_transform)
    if canonical_transform != raw_transform:
        _log(
            args.verbose,
            f"mapped --tta-transform {raw_transform!r} -> {canonical_transform!r} for backend compatibility.",
        )
    args.tta_transform = canonical_transform

    raw_merge = str(args.tta_merge_method).strip().lower()
    canonical_merge = _TTA_MERGE_ALIASES.get(raw_merge, raw_merge)
    if canonical_merge != raw_merge:
        _log(
            args.verbose,
            f"mapped --tta-merge-method {raw_merge!r} -> {canonical_merge!r} for backend compatibility.",
        )
    args.tta_merge_method = canonical_merge
    return args


def _load_input_grid(tifxyz_path, retarget_factor=1):
    surface = read_tifxyz(tifxyz_path)
    if float(retarget_factor) != 1.0:
        surface = surface.retarget(float(retarget_factor))
    surface.use_stored_resolution()
    x_s, y_s, z_s, valid_s = surface[:]
    grid = np.stack([z_s, y_s, x_s], axis=-1).astype(np.float32, copy=False)
    valid = np.asarray(valid_s, dtype=bool) & np.isfinite(grid).all(axis=2)
    grid = grid.copy()
    grid[~valid] = -1.0
    return surface, grid, valid


def _generate_cover_bboxes_from_points(
    points_zyx,
    tifxyz_uuid,
    crop_size,
    overlap=0.0,
    prune_bboxes=False,
    prune_max_remove_per_band=None,
    band_workers=1,
    show_progress=False,
):
    result = _generate_segment_cover_records(
        np.asarray(points_zyx, dtype=np.float64),
        crop_size,
        overlap=overlap,
        prune_bboxes=prune_bboxes,
        prune_max_remove_per_band=prune_max_remove_per_band,
        band_workers=band_workers,
        show_progress=show_progress,
        progress_desc="triplet_bbox_bands",
    )
    return [
        {
            "tifxyz_uuid": str(tifxyz_uuid),
            **_serialize_bbox_record(item),
        }
        for item in result["final_records"]
    ]


def _resolve_crop_size(args, model_config):
    if args.crop_size is not None:
        return tuple(int(v) for v in args.crop_size)

    cfg_crop = model_config.get("crop_size", 128)
    if isinstance(cfg_crop, int):
        c = int(cfg_crop)
        return (c, c, c)
    if isinstance(cfg_crop, (list, tuple)) and len(cfg_crop) == 3:
        return tuple(int(v) for v in cfg_crop)
    raise RuntimeError(f"Unable to resolve crop_size from checkpoint config: {cfg_crop!r}")


def _split_triplet_displacement_channels(disp_batch):
    if disp_batch.ndim != 5:
        raise RuntimeError(f"Expected displacement batch [B, C, D, H, W], got {tuple(disp_batch.shape)}")
    if disp_batch.shape[1] < 6:
        raise RuntimeError(
            "Triplet inference requires at least 6 displacement channels "
            f"(slot A + slot B), got C={int(disp_batch.shape[1])}."
        )
    branch_a = disp_batch[:, 0:3]
    branch_b = disp_batch[:, 3:6]
    return branch_a, branch_b


def _crop_center_distance_weights(crop_size, eps=1e-3):
    crop_size = tuple(int(v) for v in crop_size)
    if len(crop_size) != 3:
        raise RuntimeError(f"crop_size must be length 3, got {crop_size}")

    axes = []
    for n in crop_size:
        n = int(n)
        center = 0.5 * float(n - 1)
        radius = max(center, float(n) - 1.0 - center, 1.0)
        axis = 1.0 - (np.abs(np.arange(n, dtype=np.float32) - np.float32(center)) / np.float32(radius))
        axes.append(np.maximum(axis, np.float32(eps)))
    weights = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return weights.astype(np.float32, copy=False)


def _crop_center_distance_weights_at_points(crop_size, local_zyx, eps=1e-3):
    coords = np.asarray(local_zyx, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    weights = np.ones((coords.shape[0],), dtype=np.float32)
    finite = np.isfinite(coords).all(axis=1)
    for axis, n in enumerate(tuple(int(v) for v in crop_size)):
        center = np.float32(0.5 * float(n - 1))
        radius = np.float32(max(float(center), float(n) - 1.0 - float(center), 1.0))
        axis_w = 1.0 - (np.abs(coords[:, axis] - center) / radius)
        axis_w = np.maximum(axis_w.astype(np.float32, copy=False), np.float32(eps))
        in_bounds = (coords[:, axis] >= 0.0) & (coords[:, axis] <= float(n - 1))
        weights *= axis_w
        finite &= in_bounds
    weights[~finite] = 0.0
    return weights.astype(np.float32, copy=False)


def _records_window(records, crop_size):
    crop_arr = np.asarray(crop_size, dtype=np.int64)
    if len(records) == 0:
        raise RuntimeError("Cannot build merged displacement window without bbox records.")
    starts = []
    ends = []
    for rec in records:
        min_corner, _ = _bbox_to_min_corner_and_bounds_array(tuple(rec["bbox"]))
        start = min_corner.astype(np.int64, copy=False)
        starts.append(start)
        ends.append(start + crop_arr)
    window_min = np.min(np.stack(starts, axis=0), axis=0).astype(np.int64, copy=False)
    window_max = np.max(np.stack(ends, axis=0), axis=0).astype(np.int64, copy=False)
    window_shape = (window_max - window_min).astype(np.int64, copy=False)
    return window_min, tuple(int(v) for v in window_shape.tolist())


def _slice_len(s):
    return int(s.stop) - int(s.start)


def _chunk_slices_for_region(start, end, chunks):
    start = np.asarray(start, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    chunks = np.asarray(chunks, dtype=np.int64)
    chunk_start = start // chunks
    chunk_end = (end - 1) // chunks
    for zz in range(int(chunk_start[0]), int(chunk_end[0]) + 1):
        z0 = max(int(start[0]), zz * int(chunks[0]))
        z1 = min(int(end[0]), (zz + 1) * int(chunks[0]))
        for yy in range(int(chunk_start[1]), int(chunk_end[1]) + 1):
            y0 = max(int(start[1]), yy * int(chunks[1]))
            y1 = min(int(end[1]), (yy + 1) * int(chunks[1]))
            for xx in range(int(chunk_start[2]), int(chunk_end[2]) + 1):
                x0 = max(int(start[2]), xx * int(chunks[2]))
                x1 = min(int(end[2]), (xx + 1) * int(chunks[2]))
                yield (zz, yy, xx), (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def _format_bytes(num_bytes):
    num_bytes = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(num_bytes) < 1024.0 or unit == units[-1]:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TiB"


def _estimate_dense_merge_memmap_bytes(records, window_min, window_shape, crop_size, channels=6, chunk_size=128):
    crop_arr = np.asarray(crop_size, dtype=np.int64)
    window_min = np.asarray(window_min, dtype=np.int64)
    window_shape = tuple(int(v) for v in window_shape)
    chunks = tuple(min(max(1, int(chunk_size)), int(v)) for v in crop_arr)
    chunk_keys = set()
    for rec in records:
        min_corner, _ = _bbox_to_min_corner_and_bounds_array(tuple(rec["bbox"]))
        start = min_corner.astype(np.int64, copy=False) - window_min
        end = start + crop_arr
        for chunk_key, _ in _chunk_slices_for_region(start, end, chunks):
            chunk_keys.add(tuple(int(v) for v in chunk_key))

    bytes_per_float = np.dtype(np.float32).itemsize
    total = 0
    for chunk_key in chunk_keys:
        zz, yy, xx = (int(v) for v in chunk_key)
        z0 = zz * chunks[0]
        y0 = yy * chunks[1]
        x0 = xx * chunks[2]
        shape = (
            min(z0 + chunks[0], window_shape[0]) - z0,
            min(y0 + chunks[1], window_shape[1]) - y0,
            min(x0 + chunks[2], window_shape[2]) - x0,
        )
        voxels = int(np.prod(shape, dtype=np.int64))
        total += voxels * bytes_per_float * (int(channels) + 1)
    return int(total), len(chunk_keys)


def _check_dense_merge_temp_space(args, records, window_min, window_shape, crop_size):
    parent_dir = Path(args.merge_temp_dir) if args.merge_temp_dir is not None else Path(tempfile.gettempdir())
    parent_dir.mkdir(parents=True, exist_ok=True)
    estimated_bytes, n_chunks = _estimate_dense_merge_memmap_bytes(
        records=records,
        window_min=window_min,
        window_shape=window_shape,
        crop_size=crop_size,
        channels=6,
    )
    free_bytes = int(shutil.disk_usage(parent_dir).free)
    required_bytes = int(max(estimated_bytes * 1.10, estimated_bytes + 512 * 1024 * 1024))
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Insufficient free space for dense overlap-merge temporary memmaps. "
            f"temp_dir={str(parent_dir)!r} free={_format_bytes(free_bytes)} "
            f"estimated_needed={_format_bytes(estimated_bytes)} chunks={n_chunks}. "
            "Pass --merge-temp-dir pointing at a filesystem with enough space, for example "
            "/mnt/raid_nvme/vesuvius_tmp/infer_merge."
        )
    _log(
        args.verbose,
        "dense merge temp space OK: "
        f"temp_dir={parent_dir} free={_format_bytes(free_bytes)} "
        f"estimated_needed={_format_bytes(estimated_bytes)} chunks={n_chunks}",
    )


class _WeightedDenseDisplacementMerger:
    def __init__(self, window_min, window_shape, crop_size, channels=6, temp_dir=None, chunk_size=128):
        self.window_min = np.asarray(window_min, dtype=np.int64)
        self.window_shape = tuple(int(v) for v in window_shape)
        self.crop_size = tuple(int(v) for v in crop_size)
        self.channels = int(channels)
        chunk_size = max(1, int(chunk_size))
        self.chunks_3d = tuple(min(chunk_size, int(v)) for v in self.crop_size)
        parent_dir = None if temp_dir is None else str(temp_dir)
        if parent_dir is not None:
            Path(parent_dir).mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(prefix="infer_rowcol_triplet_merge_", dir=parent_dir)
        self.path = Path(self._tmpdir.name)
        self.weighted_sum_chunks = {}
        self.weight_sum_chunks = {}
        self.crop_weights = _crop_center_distance_weights(self.crop_size)
        self.crop_count = 0
        self.current_bytes = 0

    def _chunk_path(self, kind, chunk_key):
        zz, yy, xx = (int(v) for v in chunk_key)
        return self.path / f"{kind}_{zz}_{yy}_{xx}.npy"

    def _chunk_region(self, chunk_key):
        zz, yy, xx = (int(v) for v in chunk_key)
        z0 = zz * self.chunks_3d[0]
        y0 = yy * self.chunks_3d[1]
        x0 = xx * self.chunks_3d[2]
        return (
            slice(z0, min(z0 + self.chunks_3d[0], self.window_shape[0])),
            slice(y0, min(y0 + self.chunks_3d[1], self.window_shape[1])),
            slice(x0, min(x0 + self.chunks_3d[2], self.window_shape[2])),
        )

    def _close_memmap(self, arr):
        if arr is None:
            return
        if hasattr(arr, "flush"):
            arr.flush()
        mmap_obj = getattr(arr, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

    def _create_chunk(self, kind, chunk_key, shape):
        path = self._chunk_path(kind, chunk_key)
        arr = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        arr[...] = 0.0
        self._close_memmap(arr)
        self.current_bytes += int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
        return path

    def _open_chunk(self, path, mode):
        return np.lib.format.open_memmap(path, mode=mode, dtype=np.float32)

    def _ensure_chunk(self, chunk_key):
        chunk_key = tuple(int(v) for v in chunk_key)
        sum_path = self.weighted_sum_chunks.get(chunk_key)
        if sum_path is not None:
            return sum_path, self.weight_sum_chunks[chunk_key]

        region = self._chunk_region(chunk_key)
        chunk_shape = tuple(_slice_len(s) for s in region)
        sum_path = self._create_chunk("weighted_sum", chunk_key, (self.channels, *chunk_shape))
        weight_path = self._create_chunk("weight_sum", chunk_key, chunk_shape)
        self.weighted_sum_chunks[chunk_key] = sum_path
        self.weight_sum_chunks[chunk_key] = weight_path
        return sum_path, weight_path

    def close(self):
        self.weighted_sum_chunks.clear()
        self.weight_sum_chunks.clear()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _relative_crop_bounds(self, min_corner):
        start = np.asarray(min_corner, dtype=np.int64) - self.window_min
        end = start + np.asarray(self.crop_size, dtype=np.int64)
        if np.any(start < 0) or np.any(end > np.asarray(self.window_shape, dtype=np.int64)):
            raise RuntimeError(
                "BBox crop lies outside merged displacement window: "
                f"start={start.tolist()} end={end.tolist()} window_shape={self.window_shape}"
            )
        return start, end

    def accumulate_batch(self, disp_batch, items):
        disp_arr = np.asarray(disp_batch, dtype=np.float32)
        if disp_arr.ndim != 5:
            raise RuntimeError(f"Expected dense displacement batch [B,C,D,H,W], got {tuple(disp_arr.shape)}")
        if disp_arr.shape[1] != self.channels:
            raise RuntimeError(f"Expected {self.channels} displacement channels, got {int(disp_arr.shape[1])}")
        if tuple(int(v) for v in disp_arr.shape[2:]) != self.crop_size:
            raise RuntimeError(
                f"Dense displacement spatial shape {tuple(disp_arr.shape[2:])} does not match crop_size {self.crop_size}"
            )
        if int(disp_arr.shape[0]) != len(items):
            raise RuntimeError(f"Displacement batch size {int(disp_arr.shape[0])} does not match item count {len(items)}")

        weights = self.crop_weights
        for i, item in enumerate(items):
            start, end = self._relative_crop_bounds(item["min_corner"])
            for chunk_key, region in _chunk_slices_for_region(start, end, self.chunks_3d):
                crop_region = tuple(
                    slice(int(region_axis.start) - int(start_axis), int(region_axis.stop) - int(start_axis))
                    for region_axis, start_axis in zip(region, start)
                )
                chunk_region = self._chunk_region(chunk_key)
                local_region = tuple(
                    slice(int(region_axis.start) - int(chunk_axis.start), int(region_axis.stop) - int(chunk_axis.start))
                    for region_axis, chunk_axis in zip(region, chunk_region)
                )
                sum_path, weight_path = self._ensure_chunk(chunk_key)
                weight_region = weights[crop_region]
                sum_chunk = self._open_chunk(sum_path, mode="r+")
                weight_chunk = self._open_chunk(weight_path, mode="r+")
                try:
                    sum_chunk[(slice(None), *local_region)] += (
                        disp_arr[i][(slice(None), *crop_region)] * weight_region[None, ...]
                    )
                    weight_chunk[local_region] += weight_region
                finally:
                    self._close_memmap(sum_chunk)
                    self._close_memmap(weight_chunk)
            self.crop_count += 1

    def read_crop(self, min_corner):
        start, end = self._relative_crop_bounds(min_corner)
        out = np.zeros((self.channels, *self.crop_size), dtype=np.float32)
        for chunk_key, region in _chunk_slices_for_region(start, end, self.chunks_3d):
            chunk_key = tuple(int(v) for v in chunk_key)
            sum_path = self.weighted_sum_chunks.get(chunk_key)
            if sum_path is None:
                continue
            weight_path = self.weight_sum_chunks[chunk_key]
            crop_region = tuple(
                slice(int(region_axis.start) - int(start_axis), int(region_axis.stop) - int(start_axis))
                for region_axis, start_axis in zip(region, start)
            )
            chunk_region = self._chunk_region(chunk_key)
            local_region = tuple(
                slice(int(region_axis.start) - int(chunk_axis.start), int(region_axis.stop) - int(chunk_axis.start))
                for region_axis, chunk_axis in zip(region, chunk_region)
            )
            sum_chunk = self._open_chunk(sum_path, mode="r")
            weight_chunk = self._open_chunk(weight_path, mode="r")
            try:
                sums = np.asarray(sum_chunk[(slice(None), *local_region)], dtype=np.float32)
                weights = np.asarray(weight_chunk[local_region], dtype=np.float32)
                valid = weights > 0.0
                if bool(valid.any()):
                    np.divide(
                        sums,
                        weights[None, ...],
                        out=out[(slice(None), *crop_region)],
                        where=valid[None, ...],
                    )
            finally:
                self._close_memmap(sum_chunk)
                self._close_memmap(weight_chunk)
        return out


def _estimate_global_unit_normal(input_normals, input_normals_valid):
    normals = np.asarray(input_normals, dtype=np.float32)
    valid = np.asarray(input_normals_valid, dtype=bool)
    if normals.ndim != 3 or normals.shape[2] != 3:
        raise RuntimeError(f"Expected input_normals shape [H,W,3], got {tuple(normals.shape)}")
    if valid.shape != normals.shape[:2]:
        raise RuntimeError(
            f"input_normals_valid shape {tuple(valid.shape)} does not match normals shape {tuple(normals.shape[:2])}"
        )
    if not bool(valid.any()):
        raise RuntimeError(
            "No valid surface normals available; cannot build global triplet direction priors."
        )

    vecs = normals[valid]
    finite = np.isfinite(vecs).all(axis=1)
    vecs = vecs[finite]
    if vecs.shape[0] == 0:
        raise RuntimeError(
            "No finite surface normals available; cannot build global triplet direction priors."
        )

    mags = np.linalg.norm(vecs, axis=1)
    vecs = vecs[mags > 1e-6]
    if vecs.shape[0] == 0:
        raise RuntimeError(
            "Surface normals are degenerate; cannot build global triplet direction priors."
        )

    unit_vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-6)
    mean_vec = np.mean(unit_vecs, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    mean_norm = float(np.linalg.norm(mean_vec))
    if (not np.isfinite(mean_norm)) or mean_norm <= 1e-6:
        raise RuntimeError(
            "Global surface normal estimate is degenerate; cannot build triplet direction priors."
        )
    return (mean_vec / mean_norm).astype(np.float32, copy=False)


def _build_triplet_direction_priors_for_crop(
    crop_size,
    cond_vox,
    local_zyx,
    local_normals,
    local_normals_valid,
    fallback_unit_normal,
    mask_mode="cond",
):
    crop_size = tuple(int(v) for v in crop_size)
    if len(crop_size) != 3:
        raise RuntimeError(f"crop_size must be length 3, got {crop_size}")

    cond = np.asarray(cond_vox, dtype=np.float32)
    if cond.shape != crop_size:
        raise RuntimeError(f"cond_vox shape must match crop_size {crop_size}, got {tuple(cond.shape)}")

    priors_zyx = np.zeros(crop_size + (3,), dtype=np.float32)
    counts = np.zeros(crop_size, dtype=np.uint32)

    local_arr = np.asarray(local_zyx, dtype=np.float32)
    normals_arr = np.asarray(local_normals, dtype=np.float32)
    normals_valid = np.asarray(local_normals_valid, dtype=bool)
    if local_arr.ndim == 2 and local_arr.shape[1] == 3 and normals_arr.shape == local_arr.shape:
        finite = (
            normals_valid
            & np.isfinite(local_arr).all(axis=1)
            & np.isfinite(normals_arr).all(axis=1)
        )
        if bool(finite.any()):
            ijk = np.rint(local_arr[finite]).astype(np.int64, copy=False)
            in_bounds = (
                (ijk[:, 0] >= 0)
                & (ijk[:, 0] < crop_size[0])
                & (ijk[:, 1] >= 0)
                & (ijk[:, 1] < crop_size[1])
                & (ijk[:, 2] >= 0)
                & (ijk[:, 2] < crop_size[2])
            )
            if bool(in_bounds.any()):
                ijk = ijk[in_bounds]
                n = normals_arr[finite][in_bounds]
                np.add.at(priors_zyx[..., 0], (ijk[:, 0], ijk[:, 1], ijk[:, 2]), n[:, 0])
                np.add.at(priors_zyx[..., 1], (ijk[:, 0], ijk[:, 1], ijk[:, 2]), n[:, 1])
                np.add.at(priors_zyx[..., 2], (ijk[:, 0], ijk[:, 1], ijk[:, 2]), n[:, 2])
                np.add.at(counts, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1)

    have_prior = counts > 0
    if bool(have_prior.any()):
        priors_zyx[have_prior] /= counts[have_prior, None].astype(np.float32, copy=False)
        norms = np.linalg.norm(priors_zyx, axis=3)
        finite = np.isfinite(priors_zyx).all(axis=3) & np.isfinite(norms) & (norms > 1e-6)
        have_prior &= finite
        priors_zyx[have_prior] /= norms[have_prior, None].astype(np.float32, copy=False)

    fallback = np.asarray(fallback_unit_normal, dtype=np.float32).reshape(3)
    fill_mask = (cond > 0.5) & (~have_prior)
    if bool(fill_mask.any()):
        priors_zyx[fill_mask] = fallback
        have_prior[fill_mask] = True

    if str(mask_mode).lower() == "full":
        if bool(have_prior.any()):
            n = np.mean(priors_zyx[have_prior], axis=0, dtype=np.float64).astype(np.float32, copy=False)
            norm = float(np.linalg.norm(n))
            if np.isfinite(norm) and norm > 1e-6:
                n /= norm
            else:
                n = fallback
        else:
            n = fallback
        priors_zyx[:, :, :] = n
    elif str(mask_mode).lower() == "cond":
        priors_zyx[cond <= 0.5] = 0.0
    else:
        raise RuntimeError(f"Unknown triplet direction prior mask mode: {mask_mode!r}")

    priors = np.zeros((6, *crop_size), dtype=np.float32)
    for axis in range(3):
        priors[axis, ...] = priors_zyx[..., axis]
        priors[axis + 3, ...] = -priors_zyx[..., axis]
    return priors


def _assign_triplet_slots_to_chart_sides(
    world,
    uv_rc,
    slot_a_disp,
    slot_a_valid,
    slot_b_disp,
    slot_b_valid,
    normals,
    normals_valid,
):
    uv = np.asarray(uv_rc, dtype=np.int64)
    side_normals = normals[uv[:, 0], uv[:, 1]]
    side_normals_valid = normals_valid[uv[:, 0], uv[:, 1]] & np.isfinite(side_normals).all(axis=1)
    score_a = np.sum(np.asarray(slot_a_disp, dtype=np.float32) * side_normals, axis=1)
    score_b = np.sum(np.asarray(slot_b_disp, dtype=np.float32) * side_normals, axis=1)
    score_ok = side_normals_valid & np.isfinite(score_a) & np.isfinite(score_b)

    a_is_front = score_a >= score_b
    world_a = world + slot_a_disp
    world_b = world + slot_b_disp
    world_front = np.where(a_is_front[:, None], world_a, world_b)
    world_back = np.where(a_is_front[:, None], world_b, world_a)
    front_valid = np.where(a_is_front, slot_a_valid, slot_b_valid)
    back_valid = np.where(a_is_front, slot_b_valid, slot_a_valid)

    fallback = ~score_ok
    if bool(fallback.any()):
        world_front[fallback] = world_a[fallback]
        world_back[fallback] = world_b[fallback]
        front_valid[fallback] = slot_a_valid[fallback]
        back_valid[fallback] = slot_b_valid[fallback]

    return world_front, front_valid, world_back, back_valid, int(np.count_nonzero(score_ok))


def _compute_surface_tangent_axis(surface_grid, surface_valid, axis):
    grid = np.asarray(surface_grid, dtype=np.float32)
    valid = np.asarray(surface_valid, dtype=bool)
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise RuntimeError(f"Expected surface_grid shape [H,W,3], got {tuple(grid.shape)}")
    if valid.shape != grid.shape[:2]:
        raise RuntimeError(f"surface_valid shape {tuple(valid.shape)} does not match grid shape {tuple(grid.shape[:2])}")
    if axis not in (0, 1):
        raise RuntimeError(f"axis must be 0 or 1, got {axis}")

    tangent = np.zeros_like(grid, dtype=np.float32)
    tangent_valid = np.zeros(valid.shape, dtype=bool)
    h, w = valid.shape

    if axis == 0:
        if h >= 3:
            central_ok = valid[1:-1, :] & valid[:-2, :] & valid[2:, :]
            central_delta = 0.5 * (grid[2:, :, :] - grid[:-2, :, :])
            tangent[1:-1, :, :][central_ok] = central_delta[central_ok]
            tangent_valid[1:-1, :][central_ok] = True

        if h >= 2:
            diff = grid[1:, :, :] - grid[:-1, :, :]
            diff_ok = valid[1:, :] & valid[:-1, :]

            use_forward = (~tangent_valid[:-1, :]) & diff_ok
            tangent[:-1, :, :][use_forward] = diff[use_forward]
            tangent_valid[:-1, :][use_forward] = True

            use_backward = (~tangent_valid[1:, :]) & diff_ok
            tangent[1:, :, :][use_backward] = diff[use_backward]
            tangent_valid[1:, :][use_backward] = True
    else:
        if w >= 3:
            central_ok = valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
            central_delta = 0.5 * (grid[:, 2:, :] - grid[:, :-2, :])
            tangent[:, 1:-1, :][central_ok] = central_delta[central_ok]
            tangent_valid[:, 1:-1][central_ok] = True

        if w >= 2:
            diff = grid[:, 1:, :] - grid[:, :-1, :]
            diff_ok = valid[:, 1:] & valid[:, :-1]

            use_forward = (~tangent_valid[:, :-1]) & diff_ok
            tangent[:, :-1, :][use_forward] = diff[use_forward]
            tangent_valid[:, :-1][use_forward] = True

            use_backward = (~tangent_valid[:, 1:]) & diff_ok
            tangent[:, 1:, :][use_backward] = diff[use_backward]
            tangent_valid[:, 1:][use_backward] = True

    return tangent, tangent_valid


def _compute_surface_normals_from_input_grid(input_grid, input_valid):
    grid = np.asarray(input_grid, dtype=np.float32)
    valid = np.asarray(input_valid, dtype=bool)
    row_tangent, row_tangent_valid = _compute_surface_tangent_axis(grid, valid, axis=0)
    col_tangent, col_tangent_valid = _compute_surface_tangent_axis(grid, valid, axis=1)
    # This is the signed chart normal. Flipping exactly one UV axis flips it;
    # flipping both axes preserves it. Direction priors intentionally follow
    # this handedness rather than trying to infer an absolute inside/outside.
    normals = np.cross(col_tangent, row_tangent)
    norms = np.linalg.norm(normals, axis=2)
    finite = np.isfinite(normals).all(axis=2) & np.isfinite(norms)
    normals_valid = valid & row_tangent_valid & col_tangent_valid & finite & (norms > 1e-6)
    out = np.zeros_like(normals, dtype=np.float32)
    if bool(normals_valid.any()):
        out[normals_valid] = normals[normals_valid] / norms[normals_valid, None]
    return out, normals_valid


def _accumulate_displaced(sum_grid, count_grid, uv_rc, world_zyx, weights=None):
    if uv_rc.size == 0 or world_zyx.size == 0:
        return
    rr = uv_rc[:, 0].astype(np.int32, copy=False)
    cc = uv_rc[:, 1].astype(np.int32, copy=False)
    if weights is None:
        point_weights = np.ones((uv_rc.shape[0],), dtype=np.float32)
    else:
        point_weights = np.asarray(weights, dtype=np.float32)
        if point_weights.ndim != 1 or point_weights.shape[0] != uv_rc.shape[0]:
            raise RuntimeError(
                f"weights shape {tuple(point_weights.shape)} does not match uv count {uv_rc.shape[0]}"
            )
    weighted_world = np.asarray(world_zyx, dtype=np.float32) * point_weights[:, None]
    np.add.at(sum_grid[..., 0], (rr, cc), weighted_world[:, 0].astype(np.float32, copy=False))
    np.add.at(sum_grid[..., 1], (rr, cc), weighted_world[:, 1].astype(np.float32, copy=False))
    np.add.at(sum_grid[..., 2], (rr, cc), weighted_world[:, 2].astype(np.float32, copy=False))
    np.add.at(count_grid, (rr, cc), point_weights)


def _finalize_sparse_prediction(sum_grid, count_grid, shape_hw):
    h, w = int(shape_hw[0]), int(shape_hw[1])
    pred_grid = np.full((h, w, 3), -1.0, dtype=np.float32)
    pred_valid = np.zeros((h, w), dtype=bool)
    rr, cc = np.where(count_grid > 0)
    if rr.size == 0:
        return pred_grid, pred_valid

    denom = count_grid[rr, cc].astype(np.float32, copy=False)[:, None]
    vals = (sum_grid[rr, cc] / denom).astype(np.float32, copy=False)
    finite = np.isfinite(vals).all(axis=1)
    if finite.any():
        rr = rr[finite]
        cc = cc[finite]
        vals = vals[finite]
        pred_grid[rr, cc] = vals
        pred_valid[rr, cc] = True
    return pred_grid, pred_valid


def _project_sparse_to_input_lattice(pred_grid, pred_valid):
    # Project through a mask-aware bilinear lattice sampler so output writing always
    # happens on an explicit lattice projection step, even though this surface is
    # already indexed in stored UV space.
    uv_keep, pts_keep, projection_meta = _surface_to_stored_uv_samples_lattice(
        pred_grid,
        pred_valid,
        uv_offset=(0, 0),
        sub_r=1,
        sub_c=1,
        phase_rc=(0, 0),
    )
    out_grid = np.full_like(pred_grid, -1.0, dtype=np.float32)
    out_valid = np.zeros(pred_valid.shape, dtype=bool)
    if uv_keep.shape[0] == 0:
        return out_grid, out_valid, projection_meta

    finite = np.isfinite(pts_keep).all(axis=1)
    uv_keep = uv_keep[finite]
    pts_keep = pts_keep[finite]
    if uv_keep.shape[0] > 0:
        rr = uv_keep[:, 0].astype(np.int32, copy=False)
        cc = uv_keep[:, 1].astype(np.int32, copy=False)
        out_grid[rr, cc] = pts_keep.astype(np.float32, copy=False)
        out_valid[rr, cc] = True
    return out_grid, out_valid, projection_meta


def _merge_with_original(original_grid, original_valid, pred_grid, pred_valid):
    original_arr = np.asarray(original_grid, dtype=np.float32)
    original_mask = np.asarray(original_valid, dtype=bool)
    pred_arr = np.asarray(pred_grid, dtype=np.float32)
    pred_mask = np.asarray(pred_valid, dtype=bool)

    merged = np.full_like(original_arr, -1.0, dtype=np.float32)
    merged_valid = np.zeros_like(original_mask, dtype=bool)

    support = pred_mask & np.isfinite(pred_arr).all(axis=2)
    if bool(support.any()):
        merged[support] = pred_arr[support]
        merged_valid[support] = True

    needs_fill = original_mask & ~support
    if bool(needs_fill.any()) and bool(support.any()):
        # Map each unresolved original-valid cell to its nearest predicted support cell in UV.
        _, nearest_idx = ndimage.distance_transform_edt(~support, return_indices=True)
        nearest_r = nearest_idx[0][needs_fill]
        nearest_c = nearest_idx[1][needs_fill]
        merged[needs_fill] = pred_arr[nearest_r, nearest_c]
        merged_valid[needs_fill] = True

    merged[~merged_valid] = -1.0
    return merged, merged_valid


def _iter_bbox_batches(records, batch_size):
    for start in range(0, len(records), batch_size):
        yield start, records[start:start + batch_size]


class _WorldPointSpatialIndex:
    def __init__(self, world_points, cell_size):
        points = np.asarray(world_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise RuntimeError(f"Expected world_points shape [N,3], got {tuple(points.shape)}")
        self.cell_size = np.maximum(np.asarray(cell_size, dtype=np.float32), np.float32(1.0))
        cell = np.floor(points / self.cell_size[None, :]).astype(np.int64, copy=False)
        self.origin = cell.min(axis=0)
        rel = cell - self.origin[None, :]
        self.grid_shape = rel.max(axis=0) + 1
        linear = ((rel[:, 0] * self.grid_shape[1]) + rel[:, 1]) * self.grid_shape[2] + rel[:, 2]
        order = np.argsort(linear, kind="stable")
        linear_sorted = linear[order]
        unique_keys, starts, counts = np.unique(linear_sorted, return_index=True, return_counts=True)
        self.sorted_indices = order.astype(np.int64, copy=False)
        self.unique_keys = unique_keys.astype(np.int64, copy=False)
        self.starts = starts.astype(np.int64, copy=False)
        self.counts = counts.astype(np.int64, copy=False)

    def query_indices(self, min_corner, max_corner):
        min_corner = np.asarray(min_corner, dtype=np.float32)
        max_corner = np.asarray(max_corner, dtype=np.float32)
        min_cell = np.floor(min_corner / self.cell_size).astype(np.int64) - self.origin
        max_cell = np.floor((max_corner - np.float32(1e-4)) / self.cell_size).astype(np.int64) - self.origin
        min_cell = np.maximum(min_cell, 0)
        max_cell = np.minimum(max_cell, self.grid_shape - 1)
        if np.any(max_cell < min_cell):
            return np.zeros((0,), dtype=np.int64)

        parts = []
        for zz in range(int(min_cell[0]), int(max_cell[0]) + 1):
            for yy in range(int(min_cell[1]), int(max_cell[1]) + 1):
                base = ((zz * int(self.grid_shape[1])) + yy) * int(self.grid_shape[2])
                for xx in range(int(min_cell[2]), int(max_cell[2]) + 1):
                    key = int(base + xx)
                    pos = np.searchsorted(self.unique_keys, key)
                    if pos >= self.unique_keys.shape[0] or int(self.unique_keys[pos]) != key:
                        continue
                    start = int(self.starts[pos])
                    stop = start + int(self.counts[pos])
                    parts.append(self.sorted_indices[start:stop])
        if len(parts) == 0:
            return np.zeros((0,), dtype=np.int64)
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=0)


def _voxelize_local_surface_from_uv_points(local_points, uv_points, crop_size):
    crop_size_arr = np.asarray(crop_size, dtype=np.int64)
    vox = np.zeros(tuple(crop_size_arr.tolist()), dtype=np.float32)
    if local_points is None or uv_points is None:
        return vox

    local_arr = np.asarray(local_points, dtype=np.float64)
    uv_arr = np.asarray(uv_points, dtype=np.int64)
    if local_arr.ndim != 2 or local_arr.shape[1] != 3 or uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
        return vox
    if local_arr.shape[0] == 0 or uv_arr.shape[0] == 0 or local_arr.shape[0] != uv_arr.shape[0]:
        return vox

    finite = np.isfinite(local_arr).all(axis=1)
    if not bool(finite.any()):
        return vox
    local_arr = local_arr[finite]
    uv_arr = uv_arr[finite]

    r_min = int(uv_arr[:, 0].min())
    c_min = int(uv_arr[:, 1].min())
    r_max = int(uv_arr[:, 0].max())
    c_max = int(uv_arr[:, 1].max())
    h = int(r_max - r_min + 1)
    w = int(c_max - c_min + 1)
    if h <= 0 or w <= 0:
        return vox

    grid_local = np.zeros((h, w, 3), dtype=np.float64)
    grid_valid = np.zeros((h, w), dtype=bool)
    rr = (uv_arr[:, 0] - r_min).astype(np.int64, copy=False)
    cc = (uv_arr[:, 1] - c_min).astype(np.int64, copy=False)
    grid_local[rr, cc] = local_arr
    grid_valid[rr, cc] = True
    return voxelize_surface_grid_masked(grid_local, tuple(int(v) for v in crop_size_arr.tolist()), grid_valid).astype(
        np.float32,
        copy=False,
    )


def _prepare_bbox_item(record, crop_size, world_points, uv_points, volume_arr, point_index=None):
    bbox = tuple(record["bbox"])
    min_corner, _ = _bbox_to_min_corner_and_bounds_array(bbox)
    crop_arr = np.asarray(crop_size, dtype=np.int32)
    max_corner = min_corner + crop_arr

    if point_index is None:
        candidate_world = world_points
        candidate_uv = uv_points
    else:
        candidate_idx = point_index.query_indices(min_corner, max_corner)
        if candidate_idx.shape[0] == 0:
            return None
        candidate_world = world_points[candidate_idx]
        candidate_uv = uv_points[candidate_idx]

    in_bounds = (
        (candidate_world[:, 0] >= float(min_corner[0]))
        & (candidate_world[:, 0] < float(max_corner[0]))
        & (candidate_world[:, 1] >= float(min_corner[1]))
        & (candidate_world[:, 1] < float(max_corner[1]))
        & (candidate_world[:, 2] >= float(min_corner[2]))
        & (candidate_world[:, 2] < float(max_corner[2]))
    )
    if not bool(in_bounds.any()):
        return None

    uv_sel = candidate_uv[in_bounds].astype(np.int32, copy=False)
    world_sel = candidate_world[in_bounds].astype(np.float32, copy=False)
    local_sel = (world_sel - min_corner[None, :].astype(np.float32, copy=False)).astype(np.float32, copy=False)

    cond_vox = _voxelize_local_surface_from_uv_points(local_sel, uv_sel, crop_size).astype(np.float32, copy=False)
    vol_crop = _read_volume_crop(volume_arr, crop_size, min_corner, max_corner)
    vol_crop = vol_crop.astype(np.float32, copy=False)

    return {
        "bbox_id": int(record.get("bbox_id", -1)),
        "min_corner": min_corner.astype(np.int32, copy=False),
        "uv": uv_sel,
        "world": world_sel,
        "local": local_sel,
        "cond_vox": cond_vox,
        "volume": vol_crop,
    }


def _prepare_bbox_sample_item(record, crop_size, world_points, uv_points):
    bbox = tuple(record["bbox"])
    min_corner, _ = _bbox_to_min_corner_and_bounds_array(bbox)
    crop_arr = np.asarray(crop_size, dtype=np.int32)
    max_corner = min_corner + crop_arr

    in_bounds = (
        (world_points[:, 0] >= float(min_corner[0]))
        & (world_points[:, 0] < float(max_corner[0]))
        & (world_points[:, 1] >= float(min_corner[1]))
        & (world_points[:, 1] < float(max_corner[1]))
        & (world_points[:, 2] >= float(min_corner[2]))
        & (world_points[:, 2] < float(max_corner[2]))
    )
    if not bool(in_bounds.any()):
        return None

    uv_sel = uv_points[in_bounds].astype(np.int32, copy=False)
    world_sel = world_points[in_bounds].astype(np.float32, copy=False)
    local_sel = (world_sel - min_corner[None, :].astype(np.float32, copy=False)).astype(np.float32, copy=False)

    return {
        "bbox_id": int(record.get("bbox_id", -1)),
        "min_corner": min_corner.astype(np.int32, copy=False),
        "uv": uv_sel,
        "world": world_sel,
        "local": local_sel,
    }


def _gather_batch_items(
    batch_records,
    crop_size,
    world_points,
    uv_points,
    volume_arr,
    num_workers,
    point_index=None,
):
    if len(batch_records) == 0:
        return []
    if int(num_workers) <= 1 or len(batch_records) == 1:
        items = []
        for rec in batch_records:
            item = _prepare_bbox_item(rec, crop_size, world_points, uv_points, volume_arr, point_index=point_index)
            if item is not None:
                items.append(item)
        return items

    max_workers = min(int(num_workers), len(batch_records))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        prepared = executor.map(
            _prepare_bbox_item,
            batch_records,
            [crop_size] * len(batch_records),
            [world_points] * len(batch_records),
            [uv_points] * len(batch_records),
            [volume_arr] * len(batch_records),
            [point_index] * len(batch_records),
        )
        return [item for item in prepared if item is not None]


def _iter_prepared_batch_items(
    records,
    batch_size,
    crop_size,
    world_points,
    uv_points,
    volume_arr,
    num_workers,
    prefetch_batches=1,
    point_index=None,
):
    batch_records_iter = enumerate(_iter_bbox_batches(records, int(batch_size)), start=1)
    prefetch_batches = max(1, int(prefetch_batches))

    def _prepare(index_and_batch):
        batch_idx, (_, batch_records) = index_and_batch
        items = _gather_batch_items(
            batch_records=batch_records,
            crop_size=crop_size,
            world_points=world_points,
            uv_points=uv_points,
            volume_arr=volume_arr,
            num_workers=int(num_workers),
            point_index=point_index,
        )
        return batch_idx, items

    if prefetch_batches <= 1:
        for index_and_batch in batch_records_iter:
            yield _prepare(index_and_batch)
        return

    pending = deque()
    with ThreadPoolExecutor(max_workers=prefetch_batches) as executor:
        for _ in range(prefetch_batches):
            try:
                pending.append(executor.submit(_prepare, next(batch_records_iter)))
            except StopIteration:
                break

        while pending:
            future = pending.popleft()
            try:
                pending.append(executor.submit(_prepare, next(batch_records_iter)))
            except StopIteration:
                pass
            yield future.result()


def _gather_sample_items(
    batch_records,
    crop_size,
    world_points,
    uv_points,
    num_workers,
):
    if len(batch_records) == 0:
        return []
    if int(num_workers) <= 1 or len(batch_records) == 1:
        items = []
        for rec in batch_records:
            item = _prepare_bbox_sample_item(rec, crop_size, world_points, uv_points)
            if item is not None:
                items.append(item)
        return items

    max_workers = min(int(num_workers), len(batch_records))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        prepared = executor.map(
            _prepare_bbox_sample_item,
            batch_records,
            [crop_size] * len(batch_records),
            [world_points] * len(batch_records),
            [uv_points] * len(batch_records),
        )
        return [item for item in prepared if item is not None]


def _run_triplet_inference(
    args,
    model_state,
    records,
    crop_size,
    world_points,
    uv_points,
    volume_arr,
    shape_hw,
    input_normals,
    input_normals_valid,
):
    h, w = int(shape_hw[0]), int(shape_hw[1])
    sum_back = np.zeros((h, w, 3), dtype=np.float32)
    sum_front = np.zeros((h, w, 3), dtype=np.float32)
    count_back = np.zeros((h, w), dtype=np.float32)
    count_front = np.zeros((h, w), dtype=np.float32)
    normals_arr = np.asarray(input_normals, dtype=np.float32)
    normals_valid_arr = np.asarray(input_normals_valid, dtype=bool)
    if normals_arr.shape != (h, w, 3):
        raise RuntimeError(
            f"input_normals shape {tuple(normals_arr.shape)} does not match expected {(h, w, 3)}"
        )
    if normals_valid_arr.shape != (h, w):
        raise RuntimeError(
            f"input_normals_valid shape {tuple(normals_valid_arr.shape)} does not match expected {(h, w)}"
        )
    global_unit_normal = _estimate_global_unit_normal(normals_arr, normals_valid_arr)
    orientation_global_normals_points = int(np.count_nonzero(normals_valid_arr))

    volume_shape = np.asarray(np.shape(volume_arr), dtype=np.float64)
    if volume_shape.size < 3:
        raise RuntimeError(f"Expected 3D volume for triplet inference, got shape={tuple(np.shape(volume_arr))}")

    expected_in_channels = int(model_state["expected_in_channels"])
    if expected_in_channels != 8:
        raise RuntimeError(
            "Triplet wrap inference requires direction-conditioned checkpoints with in_channels=8 "
            "(volume + conditioning + 6 direction-prior channels); "
            f"checkpoint expects in_channels={expected_in_channels}."
        )
    model_config = dict(model_state.get("model_config") or {})
    triplet_direction_prior_mask = str(model_config.get("triplet_direction_prior_mask", "cond")).lower()
    if triplet_direction_prior_mask not in {"cond", "full"}:
        raise RuntimeError(
            "triplet_direction_prior_mask in checkpoint config must be 'cond' or 'full', "
            f"got {triplet_direction_prior_mask!r}."
        )

    n_total = len(records)
    n_batches = (n_total + int(args.batch_size) - 1) // int(args.batch_size)
    kept_bboxes = 0
    side_assignment_points = 0
    window_min, window_shape = _records_window(records, crop_size)
    _log(
        args.verbose,
        "using surface-point weighted merge: "
        f"window_min={window_min.tolist()} window_shape={list(window_shape)}",
    )
    point_index = _WorldPointSpatialIndex(world_points, cell_size=crop_size)
    _log(
        args.verbose,
        "built bbox point spatial index: "
        f"cells={int(point_index.unique_keys.shape[0])} "
        f"grid_shape={point_index.grid_shape.astype(int).tolist()}",
    )

    sample_kwargs = {
        "sample_radius": float(args.disp_sample_radius),
        "sample_spacing": float(args.disp_sample_spacing),
        "sample_min_count": int(args.disp_sample_min_count),
        "sample_reduce": str(args.disp_sample_reduce),
    }
    batch_iter = _iter_prepared_batch_items(
        records=records,
        batch_size=int(args.batch_size),
        crop_size=crop_size,
        world_points=world_points,
        uv_points=uv_points,
        volume_arr=volume_arr,
        num_workers=int(args.crop_input_workers),
        prefetch_batches=int(getattr(args, "crop_prefetch_batches", 1)),
        point_index=point_index,
    )
    batch_iter = tqdm(batch_iter, total=n_batches, desc="triplet_infer_surface", unit="batch")
    for batch_idx, items in batch_iter:
        if len(items) == 0:
            _log(args.verbose, f"batch {batch_idx}/{n_batches}: skipped (no points in batch bboxes)")
            continue

        kept_bboxes += len(items)
        d, h_c, w_c = crop_size
        real_batch_size = len(items)
        infer_batch_size = real_batch_size
        if bool(model_state.get("compiled", False)):
            infer_batch_size = int(args.batch_size)
        use_pinned_input = bool(str(args.device).startswith("cuda") and torch.cuda.is_available())
        batch_cpu = torch.empty(
            (infer_batch_size, 8, d, h_c, w_c),
            dtype=torch.float32,
            pin_memory=use_pinned_input,
        )
        batch_np = batch_cpu.numpy()
        for i, item in enumerate(items):
            uv = item["uv"].astype(np.int64, copy=False)
            batch_np[i, 0] = item["volume"]
            batch_np[i, 1] = item["cond_vox"]
            batch_np[i, 2:8] = _build_triplet_direction_priors_for_crop(
                crop_size=crop_size,
                cond_vox=item["cond_vox"],
                local_zyx=item["local"],
                local_normals=normals_arr[uv[:, 0], uv[:, 1]],
                local_normals_valid=normals_valid_arr[uv[:, 0], uv[:, 1]],
                fallback_unit_normal=global_unit_normal,
                mask_mode=triplet_direction_prior_mask,
            )
        if infer_batch_size > real_batch_size:
            batch_np[real_batch_size:] = 0.0

        model_inputs = batch_cpu.to(args.device, non_blocking=use_pinned_input)
        disp_pred = predict_displacement(args, model_state, model_inputs, use_tta=bool(args.tta), profiler=None)
        if disp_pred is None:
            raise RuntimeError("Model output did not contain 'displacement'.")
        disp_pred_t = disp_pred[:real_batch_size].detach().to(dtype=torch.float32)
        slot_a_batch, slot_b_batch = _split_triplet_displacement_channels(disp_pred_t)

        for i, item in enumerate(items):
            local = item["local"]
            uv = item["uv"]
            world = item["world"]
            point_weights = _crop_center_distance_weights_at_points(crop_size, local)

            slot_a_disp_t, slot_a_valid_t = _sample_trilinear_displacement_stack_tensor(
                slot_a_batch[i:i + 1],
                local,
                uv_rc=uv,
                **sample_kwargs,
            )
            slot_b_disp_t, slot_b_valid_t = _sample_trilinear_displacement_stack_tensor(
                slot_b_batch[i:i + 1],
                local,
                uv_rc=uv,
                **sample_kwargs,
            )
            slot_a_disp = slot_a_disp_t.cpu().numpy().astype(np.float32, copy=False)
            slot_b_disp = slot_b_disp_t.cpu().numpy().astype(np.float32, copy=False)
            slot_a_valid = slot_a_valid_t.cpu().numpy().astype(bool, copy=False)
            slot_b_valid = slot_b_valid_t.cpu().numpy().astype(bool, copy=False)

            world_front_all, front_valid, world_back_all, back_valid, n_side_assigned = (
                _assign_triplet_slots_to_chart_sides(
                    world=world,
                    uv_rc=uv,
                    slot_a_disp=slot_a_disp,
                    slot_a_valid=np.asarray(slot_a_valid, dtype=bool),
                    slot_b_disp=slot_b_disp,
                    slot_b_valid=np.asarray(slot_b_valid, dtype=bool),
                    normals=normals_arr,
                    normals_valid=normals_valid_arr,
                )
            )
            side_assignment_points += int(n_side_assigned)
            front_ok = (
                np.asarray(front_valid, dtype=bool)
                & np.isfinite(world_front_all).all(axis=1)
                & (point_weights > 0.0)
            )
            back_ok = (
                np.asarray(back_valid, dtype=bool)
                & np.isfinite(world_back_all).all(axis=1)
                & (point_weights > 0.0)
            )

            if bool(back_ok.any()):
                uv_b = uv[back_ok].astype(np.int32, copy=False)
                world_b = world_back_all[back_ok].astype(np.float32, copy=False)
                _accumulate_displaced(sum_back, count_back, uv_b, world_b, weights=point_weights[back_ok])

            if bool(front_ok.any()):
                uv_f = uv[front_ok].astype(np.int32, copy=False)
                world_f = world_front_all[front_ok].astype(np.float32, copy=False)
                _accumulate_displaced(sum_front, count_front, uv_f, world_f, weights=point_weights[front_ok])

        del batch_cpu
        del model_inputs
        del disp_pred
        del disp_pred_t

        _log(args.verbose, f"batch {batch_idx}/{n_batches}: inferred and merged {len(items)} bbox surface outputs")

    _log(args.verbose, f"merged bbox crops with points: {kept_bboxes}/{len(records)}")

    back_sparse_grid, back_sparse_valid = _finalize_sparse_prediction(sum_back, count_back, shape_hw=(h, w))
    front_sparse_grid, front_sparse_valid = _finalize_sparse_prediction(sum_front, count_front, shape_hw=(h, w))

    back_projected, back_projected_valid, back_proj_meta = _project_sparse_to_input_lattice(
        back_sparse_grid,
        back_sparse_valid,
    )
    front_projected, front_projected_valid, front_proj_meta = _project_sparse_to_input_lattice(
        front_sparse_grid,
        front_sparse_valid,
    )

    return {
        "back_grid": back_projected,
        "back_valid": back_projected_valid,
        "front_grid": front_projected,
        "front_valid": front_projected_valid,
        "back_projection_meta": back_proj_meta,
        "front_projection_meta": front_proj_meta,
        "n_back_cells": int(back_projected_valid.sum()),
        "n_front_cells": int(front_projected_valid.sum()),
        "orientation_mode": "uv_handedness_chart_normal",
        "orientation_global_normals_points": int(orientation_global_normals_points),
        "orientation_global_normal_zyx": [float(global_unit_normal[0]), float(global_unit_normal[1]), float(global_unit_normal[2])],
        "orientation_side_assignment_points": int(side_assignment_points),
        "orientation_convention": "front/back are assigned by signed displacement along normal = cross(col_tangent, row_tangent); names are arbitrary but chart-side consistent",
        "triplet_direction_prior_mask": str(triplet_direction_prior_mask),
        "triplet_slot_to_output": {"A": "dynamic", "B": "dynamic"},
        "triplet_slot_assignment": "per-point chart-side signed displacement",
        "triplet_output_direction_prior_sign": {"front": 1, "back": -1},
        "dense_overlap_merge": "center_distance_weighted_surface_point_accumulation",
        "dense_overlap_merge_window_min_zyx": [int(v) for v in window_min.tolist()],
        "dense_overlap_merge_window_shape_zyx": [int(v) for v in window_shape],
        "dense_overlap_merge_crop_count": int(kept_bboxes),
        "dense_overlap_merge_chunk_shape_zyx": [],
        "dense_overlap_merge_chunks": 0,
        "dense_overlap_merge_sparse_bytes": 0,
        "dense_overlap_merge_temp_parent": None,
        "bbox_point_membership_reuse": "single_pass_inference_items",
        "disp_sample_space": "parameterized_row_col",
        "disp_sample_radius": float(args.disp_sample_radius),
        "disp_sample_spacing": float(args.disp_sample_spacing),
        "disp_sample_min_count": int(args.disp_sample_min_count),
        "disp_sample_reduce": str(args.disp_sample_reduce),
    }


def _save_surface(
    grid,
    valid,
    out_dir,
    uuid,
    step_size,
    voxel_size_um,
    source,
    metadata,
    apply_mesh_cleanup=True,
):
    save_grid = np.asarray(grid, dtype=np.float32).copy()
    save_valid = np.asarray(valid, dtype=bool)
    metadata_out = dict(metadata) if isinstance(metadata, dict) else {}

    if bool(apply_mesh_cleanup):
        save_grid, cleanup_meta = _cleanup_surface_grid_before_save(
            save_grid,
            save_valid,
            target_step_size=float(step_size),
        )
        metadata_out.update(cleanup_meta)

    save_grid[~save_valid] = -1.0
    save_tifxyz(
        save_grid,
        out_dir,
        uuid,
        step_size=int(step_size),
        voxel_size_um=float(voxel_size_um),
        source=source,
        additional_metadata=metadata_out,
    )
    return str(Path(out_dir) / uuid)


def _build_valid_grid_triangles(valid_mask):
    valid = np.asarray(valid_mask, dtype=bool)
    h, w = valid.shape
    if h < 2 or w < 2:
        return np.zeros((0, 3), dtype=np.int64)

    quad_valid = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    qr, qc = np.where(quad_valid)
    if qr.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    index_grid = -np.ones_like(valid, dtype=np.int64)
    vr, vc = np.where(valid)
    index_grid[vr, vc] = np.arange(vr.size, dtype=np.int64)

    v00 = index_grid[qr, qc]
    v10 = index_grid[qr + 1, qc]
    v01 = index_grid[qr, qc + 1]
    v11 = index_grid[qr + 1, qc + 1]

    t0 = np.stack([v00, v10, v11], axis=1)
    t1 = np.stack([v00, v11, v01], axis=1)
    faces = np.concatenate([t0, t1], axis=0)
    return faces.astype(np.int64, copy=False)


def _build_valid_grid_edges(valid_mask):
    valid = np.asarray(valid_mask, dtype=bool)
    h, w = valid.shape
    if h == 0 or w == 0:
        return np.zeros((0, 2), dtype=np.int64)

    index_grid = -np.ones_like(valid, dtype=np.int64)
    vr, vc = np.where(valid)
    if vr.size < 2:
        return np.zeros((0, 2), dtype=np.int64)
    index_grid[vr, vc] = np.arange(vr.size, dtype=np.int64)

    horiz = valid[:, :-1] & valid[:, 1:]
    hr, hc = np.where(horiz)
    h_edges = np.zeros((0, 2), dtype=np.int64)
    if hr.size > 0:
        h_edges = np.stack([index_grid[hr, hc], index_grid[hr, hc + 1]], axis=1)

    vert = valid[:-1, :] & valid[1:, :]
    vr_e, vc_e = np.where(vert)
    v_edges = np.zeros((0, 2), dtype=np.int64)
    if vr_e.size > 0:
        v_edges = np.stack([index_grid[vr_e, vc_e], index_grid[vr_e + 1, vc_e]], axis=1)

    if h_edges.size == 0 and v_edges.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if h_edges.size == 0:
        return v_edges.astype(np.int64, copy=False)
    if v_edges.size == 0:
        return h_edges.astype(np.int64, copy=False)
    return np.concatenate([h_edges, v_edges], axis=0).astype(np.int64, copy=False)


def _resolve_duplicate_vertex_positions(vertices, edges, target_step):
    verts = np.asarray(vertices, dtype=np.float64).copy()
    if verts.shape[0] < 2:
        return verts, 0

    tol = max(1e-3, float(target_step) * 1e-4)
    max_shift = max(tol, 0.75 * float(target_step))
    original = verts.copy()
    quantized = np.round(verts / tol).astype(np.int64)
    _, inverse, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
    dup_groups = np.where(counts > 1)[0]
    if dup_groups.size == 0:
        return verts, 0

    edge_idx = np.asarray(edges, dtype=np.int64)
    if edge_idx.size == 0:
        return verts, 0
    src = np.concatenate([edge_idx[:, 0], edge_idx[:, 1]], axis=0)
    dst = np.concatenate([edge_idx[:, 1], edge_idx[:, 0]], axis=0)
    order = np.argsort(src, kind="stable")
    src_sorted = src[order]
    dst_sorted = dst[order]
    vertex_ids = np.arange(verts.shape[0], dtype=np.int64)
    nbr_start = np.searchsorted(src_sorted, vertex_ids, side="left")
    nbr_end = np.searchsorted(src_sorted, vertex_ids, side="right")

    fixed = 0
    for gid in dup_groups.tolist():
        members = np.where(inverse == gid)[0]
        if members.size <= 1:
            continue
        member_set = set(int(v) for v in members.tolist())
        for rank, vid in enumerate(members[1:], start=1):
            vid_i = int(vid)
            start = int(nbr_start[vid_i])
            end = int(nbr_end[vid_i])
            nbrs_all = dst_sorted[start:end]
            nbrs = [int(n) for n in nbrs_all.tolist() if int(n) not in member_set]
            if len(nbrs) > 0:
                new_pos = np.mean(verts[np.asarray(nbrs, dtype=np.int64)], axis=0)
            else:
                seed = float((vid_i + 1) * (rank + 3))
                direction = np.array(
                    [
                        np.sin(seed),
                        np.cos(2.0 * seed),
                        np.sin(3.0 * seed + 0.5),
                    ],
                    dtype=np.float64,
                )
                norm = np.linalg.norm(direction)
                if norm < 1e-8:
                    direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                else:
                    direction /= norm
                new_pos = verts[vid_i] + direction * tol
            delta = new_pos - original[vid_i]
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_shift:
                new_pos = original[vid_i] + (delta / delta_norm) * max_shift
            verts[vid_i] = new_pos
            fixed += 1

    return verts, int(fixed)


if njit is not None:
    @njit(cache=True)
    def _regularize_edge_lengths_numba_kernel(
        vertices,
        edges,
        target_step,
        iterations,
        relax_step,
        anchor_weight,
        max_displacement,
    ):
        verts = vertices.copy()
        original = vertices.copy()
        n_vertices = verts.shape[0]
        n_edges = edges.shape[0]
        eps = 1e-8

        for _ in range(iterations):
            disp = np.zeros((n_vertices, 3), dtype=np.float64)
            counts = np.zeros((n_vertices,), dtype=np.float64)

            for k in range(n_edges):
                i = edges[k, 0]
                j = edges[k, 1]

                dx0 = verts[j, 0] - verts[i, 0]
                dx1 = verts[j, 1] - verts[i, 1]
                dx2 = verts[j, 2] - verts[i, 2]

                length = np.sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2)
                if length <= eps:
                    continue

                inv_len = 1.0 / length
                err = length - target_step
                move0 = 0.5 * err * dx0 * inv_len
                move1 = 0.5 * err * dx1 * inv_len
                move2 = 0.5 * err * dx2 * inv_len

                disp[i, 0] += move0
                disp[i, 1] += move1
                disp[i, 2] += move2
                disp[j, 0] -= move0
                disp[j, 1] -= move1
                disp[j, 2] -= move2
                counts[i] += 1.0
                counts[j] += 1.0

            for i in range(n_vertices):
                if counts[i] > 0.0:
                    verts[i, 0] += relax_step * (disp[i, 0] / counts[i]) + anchor_weight * (original[i, 0] - verts[i, 0])
                    verts[i, 1] += relax_step * (disp[i, 1] / counts[i]) + anchor_weight * (original[i, 1] - verts[i, 1])
                    verts[i, 2] += relax_step * (disp[i, 2] / counts[i]) + anchor_weight * (original[i, 2] - verts[i, 2])
                else:
                    verts[i, 0] += anchor_weight * (original[i, 0] - verts[i, 0])
                    verts[i, 1] += anchor_weight * (original[i, 1] - verts[i, 1])
                    verts[i, 2] += anchor_weight * (original[i, 2] - verts[i, 2])

                dx0 = verts[i, 0] - original[i, 0]
                dx1 = verts[i, 1] - original[i, 1]
                dx2 = verts[i, 2] - original[i, 2]
                dist = np.sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2)
                if dist > max_displacement and dist > eps:
                    scale = max_displacement / dist
                    verts[i, 0] = original[i, 0] + dx0 * scale
                    verts[i, 1] = original[i, 1] + dx1 * scale
                    verts[i, 2] = original[i, 2] + dx2 * scale

        return verts
else:
    _regularize_edge_lengths_numba_kernel = None


def _regularize_edge_lengths(
    vertices,
    edges,
    target_step,
    iterations=8,
    relax_step=0.35,
    anchor_weight=0.08,
    max_displacement_ratio=0.5,
):
    verts = np.asarray(vertices, dtype=np.float64).copy()
    if verts.shape[0] == 0 or np.asarray(edges).size == 0:
        return verts

    edge_idx = np.asarray(edges, dtype=np.int64)
    if edge_idx.shape[0] == 0:
        return verts

    max_displacement = max(1e-6, float(target_step) * float(max_displacement_ratio))

    if _regularize_edge_lengths_numba_kernel is not None and edge_idx.shape[0] >= 256:
        return _regularize_edge_lengths_numba_kernel(
            verts,
            edge_idx,
            float(target_step),
            int(iterations),
            float(relax_step),
            float(anchor_weight),
            float(max_displacement),
        )

    e0 = edge_idx[:, 0]
    e1 = edge_idx[:, 1]
    original = verts.copy()
    target = float(target_step)
    eps = 1e-8

    for _ in range(int(iterations)):
        delta = verts[e1] - verts[e0]
        lengths = np.linalg.norm(delta, axis=1)
        good = lengths > eps
        if not bool(good.any()):
            break

        direction = np.zeros_like(delta)
        direction[good] = delta[good] / lengths[good, None]
        length_error = lengths - target
        move = 0.5 * length_error[:, None] * direction

        disp = np.zeros_like(verts)
        counts = np.zeros((verts.shape[0], 1), dtype=np.float64)
        np.add.at(disp, e0, move)
        np.add.at(disp, e1, -move)
        np.add.at(counts, e0, 1.0)
        np.add.at(counts, e1, 1.0)

        nonzero = counts[:, 0] > 0
        update = np.zeros_like(verts)
        update[nonzero] = disp[nonzero] / counts[nonzero]

        verts = verts + float(relax_step) * update + float(anchor_weight) * (original - verts)
        delta_from_original = verts - original
        delta_norm = np.linalg.norm(delta_from_original, axis=1)
        too_far = delta_norm > max_displacement
        if bool(too_far.any()):
            scale = (max_displacement / np.maximum(delta_norm[too_far], 1e-8))[:, None]
            verts[too_far] = original[too_far] + delta_from_original[too_far] * scale

    return verts


def _cleanup_surface_grid_before_save(grid, valid, target_step_size):
    grid_arr = np.asarray(grid, dtype=np.float32)
    valid_mask = np.asarray(valid, dtype=bool)
    out = grid_arr.copy()

    cleanup_meta = {
        "mesh_cleanup_enabled": bool(trimesh is not None),
        "mesh_cleanup_target_step": float(target_step_size),
        "mesh_cleanup_duplicate_vertices_fixed": 0,
        "mesh_cleanup_unique_edges": 0,
        "mesh_cleanup_relaxation_applied": False,
    }

    if trimesh is None:
        return out, cleanup_meta

    vr, vc = np.where(valid_mask)
    if vr.size < 3:
        return out, cleanup_meta

    faces = _build_valid_grid_triangles(valid_mask)
    if faces.shape[0] == 0:
        return out, cleanup_meta

    edges_unique = _build_valid_grid_edges(valid_mask)
    if edges_unique.shape[0] == 0:
        return out, cleanup_meta

    verts = out[vr, vc].astype(np.float64, copy=True)
    if not np.isfinite(verts).all():
        return out, cleanup_meta

    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False, validate=False)
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    except ValueError:
        return out, cleanup_meta

    cleanup_meta["mesh_cleanup_unique_edges"] = int(edges_unique.shape[0])

    verts, dup_fixed = _resolve_duplicate_vertex_positions(
        verts,
        edges_unique,
        target_step=float(target_step_size),
    )
    cleanup_meta["mesh_cleanup_duplicate_vertices_fixed"] = int(dup_fixed)

    if target_step_size is not None and float(target_step_size) > 0.0:
        verts = _regularize_edge_lengths(
            verts,
            edges_unique,
            target_step=float(target_step_size),
            iterations=8,
            relax_step=0.35,
            anchor_weight=0.08,
        )
        cleanup_meta["mesh_cleanup_relaxation_applied"] = True

    out[vr, vc] = verts.astype(np.float32, copy=False)
    return out, cleanup_meta


def _rescale_grid_for_save(grid, valid, scale_factor):
    out = np.asarray(grid, dtype=np.float32).copy()
    valid_mask = np.asarray(valid, dtype=bool)
    out[~valid_mask] = -1.0
    s = int(scale_factor)
    if s != 1:
        out[valid_mask] = out[valid_mask] * float(s)
    return out


def _append_iteration_suffix(uuid_base, iteration_index, iterative_mode):
    if not bool(iterative_mode):
        return str(uuid_base)
    return f"{uuid_base}_iteration_{int(iteration_index)}"


def _extract_surface_points_for_iteration(input_grid, input_valid, retarget_factor, verbose):
    rows, cols = np.where(input_valid)
    world_points = input_grid[rows, cols].astype(np.float32, copy=False)
    uv_points = np.stack([rows, cols], axis=-1).astype(np.int32, copy=False)
    if world_points.shape[0] == 0:
        raise RuntimeError("Input tifxyz has no valid points.")
    if verbose:
        world_min = np.min(world_points, axis=0).astype(np.float32, copy=False)
        world_max = np.max(world_points, axis=0).astype(np.float32, copy=False)
        _log(
            verbose,
            "input lattice world bounds after retarget: "
            f"retarget_factor={retarget_factor:g} "
            f"z=[{world_min[0]:.3f},{world_max[0]:.3f}] "
            f"y=[{world_min[1]:.3f},{world_max[1]:.3f}] "
            f"x=[{world_min[2]:.3f},{world_max[2]:.3f}]",
        )
    return world_points, uv_points


def _run_single_iteration(
    args,
    model_state,
    crop_size,
    volume_arr,
    input_tifxyz_path,
    out_dir,
    out_prefix,
    retarget_factor,
    tifxyz_step_size,
    tifxyz_voxel_size_um,
    stored_scale_rc,
    save_scale_factor,
    iteration_index,
    iterations_requested,
    iterative_mode,
    iter_direction,
    keep_previous_wrap,
    preloaded_input=None,
):
    if preloaded_input is None:
        _, input_grid, input_valid = _load_input_grid(input_tifxyz_path, retarget_factor=retarget_factor)
    else:
        _, input_grid, input_valid = preloaded_input
    input_path_resolved = Path(input_tifxyz_path).resolve()
    input_uuid = input_path_resolved.name

    world_points, uv_points = _extract_surface_points_for_iteration(
        input_grid=input_grid,
        input_valid=input_valid,
        retarget_factor=retarget_factor,
        verbose=bool(args.verbose),
    )
    input_normals, input_normals_valid = _compute_surface_normals_from_input_grid(input_grid, input_valid)
    records = _generate_cover_bboxes_from_points(
        world_points,
        tifxyz_uuid=input_uuid,
        crop_size=crop_size,
        overlap=float(args.bbox_overlap),
        prune_bboxes=bool(args.bbox_prune),
        prune_max_remove_per_band=args.bbox_prune_max_remove_per_band,
        band_workers=int(args.bbox_band_workers),
        show_progress=bool(args.verbose),
    )
    _log(args.verbose, f"generated bboxes (retargeted coords): {len(records)}")

    infer_out = _run_triplet_inference(
        args=args,
        model_state=model_state,
        records=records,
        crop_size=crop_size,
        world_points=world_points,
        uv_points=uv_points,
        volume_arr=volume_arr,
        shape_hw=input_valid.shape,
        input_normals=input_normals,
        input_normals_valid=input_normals_valid,
    )

    back_merged, back_merged_valid = _merge_with_original(
        input_grid,
        input_valid,
        infer_out["back_grid"],
        infer_out["back_valid"],
    )
    front_merged, front_merged_valid = _merge_with_original(
        input_grid,
        input_valid,
        infer_out["front_grid"],
        infer_out["front_valid"],
    )

    run_meta = {
        "checkpoint_path": str(args.checkpoint_path),
        "compile_model": bool(model_state.get("compiled", False)),
        "compile_requested": bool(model_state.get("compile_requested", False)),
        "compile_mode": str(model_state.get("compile_mode", "")),
        "crop_size": [int(v) for v in crop_size],
        "bbox_count": int(len(records)),
        "bbox_overlap": float(args.bbox_overlap),
        "bbox_prune": bool(args.bbox_prune),
        "bbox_band_workers": int(args.bbox_band_workers),
        "crop_input_workers": int(args.crop_input_workers),
        "crop_prefetch_batches": int(args.crop_prefetch_batches),
        "triplet_output_channels": 6,
        "retarget_factor": float(retarget_factor),
        "tta_transform_effective": str(args.tta_transform),
        "tta_merge_method_effective": str(args.tta_merge_method),
        "n_pred_back_cells": int(infer_out["n_back_cells"]),
        "n_pred_front_cells": int(infer_out["n_front_cells"]),
        "orientation_mode": str(infer_out.get("orientation_mode", "legacy")),
        "orientation_global_normals_points": int(infer_out.get("orientation_global_normals_points", 0)),
        "orientation_global_normal_zyx": list(infer_out.get("orientation_global_normal_zyx", [])),
        "orientation_side_assignment_points": int(infer_out.get("orientation_side_assignment_points", 0)),
        "orientation_convention": str(infer_out.get("orientation_convention", "")),
        "triplet_direction_prior_mask_effective": str(infer_out.get("triplet_direction_prior_mask", "cond")),
        "triplet_slot_to_output": dict(infer_out.get("triplet_slot_to_output", {"A": "front", "B": "back"})),
        "triplet_slot_assignment": str(infer_out.get("triplet_slot_assignment", "")),
        "triplet_output_direction_prior_sign": dict(infer_out.get("triplet_output_direction_prior_sign", {"front": 1, "back": -1})),
        "dense_overlap_merge": str(infer_out.get("dense_overlap_merge", "")),
        "dense_overlap_merge_window_min_zyx": list(infer_out.get("dense_overlap_merge_window_min_zyx", [])),
        "dense_overlap_merge_window_shape_zyx": list(infer_out.get("dense_overlap_merge_window_shape_zyx", [])),
        "dense_overlap_merge_crop_count": int(infer_out.get("dense_overlap_merge_crop_count", 0)),
        "dense_overlap_merge_chunk_shape_zyx": list(infer_out.get("dense_overlap_merge_chunk_shape_zyx", [])),
        "dense_overlap_merge_chunks": int(infer_out.get("dense_overlap_merge_chunks", 0)),
        "dense_overlap_merge_sparse_bytes": int(infer_out.get("dense_overlap_merge_sparse_bytes", 0)),
        "dense_overlap_merge_temp_parent": infer_out.get("dense_overlap_merge_temp_parent", None),
        "disp_sample_space": str(infer_out.get("disp_sample_space", "parameterized_row_col")),
        "disp_sample_radius": float(infer_out.get("disp_sample_radius", args.disp_sample_radius)),
        "disp_sample_spacing": float(infer_out.get("disp_sample_spacing", args.disp_sample_spacing)),
        "disp_sample_min_count": int(infer_out.get("disp_sample_min_count", args.disp_sample_min_count)),
        "disp_sample_reduce": str(infer_out.get("disp_sample_reduce", args.disp_sample_reduce)),
        "save_coordinate_scale_factor": int(save_scale_factor),
        "stored_scale_rc": None if stored_scale_rc is None else [float(stored_scale_rc[0]), float(stored_scale_rc[1])],
        "effective_step_size_used": int(tifxyz_step_size),
        "front_projection": infer_out["front_projection_meta"],
        "back_projection": infer_out["back_projection_meta"],
        "iteration_index": int(iteration_index),
        "iterations_requested": int(iterations_requested),
        "iter_direction": None if iter_direction is None else str(iter_direction),
        "keep_previous_wrap": bool(keep_previous_wrap),
        "iterative_mode": bool(iterative_mode),
        "run_argv": list(sys.argv[1:]),
    }

    outputs = {}
    source = str(args.checkpoint_path)
    input_grid_save = _rescale_grid_for_save(input_grid, input_valid, save_scale_factor)
    back_merged_save = _rescale_grid_for_save(back_merged, back_merged_valid, save_scale_factor)
    front_merged_save = _rescale_grid_for_save(front_merged, front_merged_valid, save_scale_factor)
    original_uuid = _append_iteration_suffix(out_prefix, iteration_index, iterative_mode)
    back_uuid = _append_iteration_suffix(f"{out_prefix}_back", iteration_index, iterative_mode)
    front_uuid = _append_iteration_suffix(f"{out_prefix}_front", iteration_index, iterative_mode)

    if args.save_original_copy:
        original_target = Path(out_dir) / original_uuid
        if original_target.resolve() != input_path_resolved:
            outputs["original"] = _save_surface(
                input_grid_save,
                input_valid,
                out_dir,
                original_uuid,
                tifxyz_step_size,
                tifxyz_voxel_size_um,
                source=source,
                metadata={**run_meta, "surface_role": "original"},
                apply_mesh_cleanup=False,
            )
        else:
            outputs["original"] = str(input_path_resolved)
            _log(
                args.verbose,
                "original output path equals input tifxyz path; skipping rewrite and reusing input as unchanged original.",
            )

    save_back = True
    save_front = True
    if bool(iterative_mode) and int(iteration_index) >= 2 and (not bool(keep_previous_wrap)):
        if str(iter_direction) == "front":
            save_back = False
        elif str(iter_direction) == "back":
            save_front = False

    if save_back:
        outputs["back"] = _save_surface(
            back_merged_save,
            back_merged_valid,
            out_dir,
            back_uuid,
            tifxyz_step_size,
            tifxyz_voxel_size_um,
            source=source,
            metadata={**run_meta, "surface_role": "back"},
        )
    if save_front:
        outputs["front"] = _save_surface(
            front_merged_save,
            front_merged_valid,
            out_dir,
            front_uuid,
            tifxyz_step_size,
            tifxyz_voxel_size_um,
            source=source,
            metadata={**run_meta, "surface_role": "front"},
        )

    chain_valid_cells = None
    if iter_direction in {"front", "back"}:
        chain_valid_cells = int(infer_out[f"n_{iter_direction}_cells"])

    return {
        "input_tifxyz_path": str(input_path_resolved),
        "outputs": outputs,
        "n_pred_back_cells": int(infer_out["n_back_cells"]),
        "n_pred_front_cells": int(infer_out["n_front_cells"]),
        "chain_valid_cells": chain_valid_cells,
    }


def run(args):
    args = _canonicalize_tta_settings(args)
    retarget_factor = float(2 ** int(args.volume_scale))
    surface, input_grid, input_valid = _load_input_grid(args.tifxyz_path, retarget_factor=retarget_factor)
    input_uuid = Path(args.tifxyz_path).resolve().name
    out_prefix = args.output_prefix if args.output_prefix else input_uuid

    model_state = load_model(args)
    model_config = model_state["model_config"]
    crop_size = _resolve_crop_size(args, model_config)
    tifxyz_step_size, tifxyz_voxel_size_um, stored_scale_rc = resolve_tifxyz_params(
        args,
        model_config,
        args.volume_scale,
        input_scale=surface.get_scale_tuple(),
    )
    if stored_scale_rc is not None:
        step_y = _scale_to_subsample_stride(stored_scale_rc[0])
        step_x = _scale_to_subsample_stride(stored_scale_rc[1])
        if step_y != step_x:
            raise RuntimeError(
                "Triplet wrap inference currently requires isotropic stored scale; "
                f"got scale={stored_scale_rc!r} -> steps ({step_y}, {step_x})."
            )
    volume_arr, resolved_volume_level = _open_vc_volume_level(
        args.volume_path,
        volume_scale=args.volume_scale,
        cache_dir=args.volume_cache_dir,
        chunk_cache_gb=args.volume_chunk_cache_gb,
    )
    _log(
        args.verbose,
        "resolved vc volume level "
        f"shape={tuple(int(v) for v in volume_arr.shape)} "
        f"volume_scale={int(args.volume_scale)} "
        f"resolved_level={int(resolved_volume_level)} "
        f"chunk_cache_gb={float(args.volume_chunk_cache_gb):.3g}",
    )

    out_dir = str(Path(args.out_dir).resolve()) if args.out_dir else str(Path(args.tifxyz_path).resolve().parent)
    os.makedirs(out_dir, exist_ok=True)
    save_scale_factor = int(2 ** int(args.volume_scale))
    iterative_mode = args.iterations is not None
    iterations_requested = int(args.iterations) if iterative_mode else 1
    iter_direction = str(args.iter_direction) if iterative_mode else None
    current_tifxyz_path = str(Path(args.tifxyz_path).resolve())
    outputs_by_iteration = {}
    iterations_completed = 0
    stop_reason = None

    for iteration_index in range(1, iterations_requested + 1):
        _log(
            args.verbose,
            f"[iteration {iteration_index}/{iterations_requested}] input={current_tifxyz_path}",
        )
        iter_result = _run_single_iteration(
            args=args,
            model_state=model_state,
            crop_size=crop_size,
            volume_arr=volume_arr,
            input_tifxyz_path=current_tifxyz_path,
            out_dir=out_dir,
            out_prefix=out_prefix,
            retarget_factor=retarget_factor,
            tifxyz_step_size=tifxyz_step_size,
            tifxyz_voxel_size_um=tifxyz_voxel_size_um,
            stored_scale_rc=stored_scale_rc,
            save_scale_factor=save_scale_factor,
            iteration_index=iteration_index,
            iterations_requested=iterations_requested,
            iterative_mode=iterative_mode,
            iter_direction=iter_direction,
            keep_previous_wrap=bool(args.keep_previous_wrap),
            preloaded_input=(surface, input_grid, input_valid) if iteration_index == 1 else None,
        )
        outputs_by_iteration[str(iteration_index)] = {
            "input_tifxyz_path": iter_result["input_tifxyz_path"],
            "outputs": iter_result["outputs"],
            "n_pred_back_cells": int(iter_result["n_pred_back_cells"]),
            "n_pred_front_cells": int(iter_result["n_pred_front_cells"]),
        }
        iterations_completed = int(iteration_index)

        if not iterative_mode:
            continue

        chain_valid_cells = int(iter_result["chain_valid_cells"])
        outputs_by_iteration[str(iteration_index)]["chain_valid_cells"] = chain_valid_cells
        outputs_by_iteration[str(iteration_index)]["chained_direction"] = str(iter_direction)
        if chain_valid_cells <= 0 and iteration_index < iterations_requested:
            stop_reason = f"no_valid_{iter_direction}_cells"
            _log(
                args.verbose,
                f"stopping iterative chaining early at iteration {iteration_index}: {stop_reason}",
            )
            break

        if iteration_index < iterations_requested:
            next_input_path = iter_result["outputs"].get(iter_direction, None)
            if next_input_path is None:
                raise RuntimeError(
                    f"Iteration {iteration_index} did not save chained direction output '{iter_direction}'."
                )
            current_tifxyz_path = str(next_input_path)

    if not iterative_mode:
        return outputs_by_iteration["1"]["outputs"]

    return {
        "iterations_requested": int(iterations_requested),
        "iterations_completed": int(iterations_completed),
        "iter_direction": str(iter_direction),
        "keep_previous_wrap": bool(args.keep_previous_wrap),
        "stopped_early": bool(iterations_completed < iterations_requested),
        "stop_reason": stop_reason,
        "outputs_by_iteration": outputs_by_iteration,
    }


def run_copy_displacement(copy_args):
    argv = _copy_args_to_argv(copy_args)
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        detail = str(exc)
        raise RuntimeError(f"Invalid copy args for infer_rowcol_triplet_wraps: {detail}") from exc
    return run(args)


def main(argv=None):
    args = parse_args(argv)
    outputs = run(args)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
