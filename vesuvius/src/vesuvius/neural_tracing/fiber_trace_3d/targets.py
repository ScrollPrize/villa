from __future__ import annotations

import math
import time
from dataclasses import replace

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace_3d.direction import (
    encode_lasagna_direction_3x2,
    projection_magnitude_weights_3x2,
)
from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    _TARGET_MODE_CP_ONLY,
    _TARGET_MODE_DENSE_LINE,
    FiberTrace3DBatch,
    FiberTrace3DConfig,
)


_GRID_CACHE: dict[tuple[tuple[int, int, int], str, torch.dtype], torch.Tensor] = {}


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _zyx_grid(
    patch_shape: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (tuple(int(v) for v in patch_shape), str(device), dtype)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    z = torch.arange(patch_shape[0], dtype=dtype, device=device)
    y = torch.arange(patch_shape[1], dtype=dtype, device=device)
    x = torch.arange(patch_shape[2], dtype=dtype, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    grid = torch.stack([zz, yy, xx], dim=-1)
    _GRID_CACHE[key] = grid
    return grid


def _materialize_cp_only_sample(
    *,
    batch: FiberTrace3DBatch,
    sample_index: int,
    grid: torch.Tensor,
    patch_shape: tuple[int, int, int],
    radius2: float,
    positive: torch.Tensor,
    direction_indices: list[torch.Tensor],
    direction_tangents: list[torch.Tensor],
) -> int:
    cp = batch.cp_local_zyx[sample_index].to(dtype=torch.float32)
    dist2 = torch.sum((grid - cp.view(1, 1, 1, 3)) ** 2, dim=-1)
    sample_positive = dist2 <= float(radius2)
    positive[sample_index] = sample_positive
    tangent = batch.target_tangent_zyx[sample_index].to(dtype=torch.float32)
    local_indices = torch.nonzero(sample_positive, as_tuple=False)
    if int(local_indices.numel()) == 0:
        return 0
    batch_column = torch.full(
        (int(local_indices.shape[0]), 1),
        int(sample_index),
        dtype=torch.long,
        device=local_indices.device,
    )
    direction_indices.append(torch.cat([batch_column, local_indices.to(dtype=torch.long)], dim=1))
    direction_tangents.append(tangent.view(1, 3).expand(int(local_indices.shape[0]), 3))
    return int(local_indices.shape[0])


def _unique_indices_and_tangents(
    indices_bzyx: torch.Tensor,
    tangents_zyx: torch.Tensor,
    *,
    patch_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(indices_bzyx.numel()) == 0:
        return indices_bzyx, tangents_zyx
    z_size, y_size, x_size = (int(v) for v in patch_shape)
    linear = (
        ((indices_bzyx[:, 0] * z_size + indices_bzyx[:, 1]) * y_size + indices_bzyx[:, 2])
        * x_size
        + indices_bzyx[:, 3]
    )
    order = torch.argsort(linear, stable=True)
    sorted_linear = linear[order]
    keep_sorted = torch.ones_like(sorted_linear, dtype=torch.bool)
    keep_sorted[1:] = sorted_linear[1:] != sorted_linear[:-1]
    keep = order[keep_sorted]
    return indices_bzyx[keep], tangents_zyx[keep]


def _materialize_dense_line_indices(
    *,
    batch: FiberTrace3DBatch,
    patch_shape: tuple[int, int, int],
    positive: torch.Tensor,
    validate: bool,
    coordinate_scale: int = 1,
    input_shape: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    device = batch.volume.device
    starts = batch.target_segment_starts_zyx.to(device=device, dtype=torch.float32)
    ends = batch.target_segment_ends_zyx.to(device=device, dtype=torch.float32)
    segment_count = int(starts.shape[0])
    if segment_count == 0:
        empty_indices = torch.zeros((0, 4), dtype=torch.long, device=device)
        empty_tangents = torch.zeros((0, 3), dtype=torch.float32, device=device)
        if validate and bool((batch.target_modes == _TARGET_MODE_DENSE_LINE).any()):
            raise ValueError("dense line target has no patch-overlapping segments")
        return empty_indices, empty_tangents, 0, 0

    counts = batch.target_segment_counts.to(device=device, dtype=torch.long)
    sample_ids = torch.repeat_interleave(
        torch.arange(int(batch.volume.shape[0]), dtype=torch.long, device=device),
        counts,
    )
    if int(sample_ids.shape[0]) != segment_count:
        raise ValueError(
            "target segment metadata is inconsistent: "
            f"sample_ids={int(sample_ids.shape[0])} segments={segment_count}"
        )
    dense_sample = batch.target_modes.to(device=device) == _TARGET_MODE_DENSE_LINE
    dense_segment = dense_sample[sample_ids]
    if not bool(dense_segment.any()):
        empty_indices = torch.zeros((0, 4), dtype=torch.long, device=device)
        empty_tangents = torch.zeros((0, 3), dtype=torch.float32, device=device)
        if validate and bool(dense_sample.any()):
            raise ValueError("dense line target has no patch-overlapping segments")
        return empty_indices, empty_tangents, 0, 0
    starts = starts[dense_segment]
    ends = ends[dense_segment]
    sample_ids = sample_ids[dense_segment]

    delta = ends - starts
    length = torch.linalg.vector_norm(delta, dim=1)
    valid_segment = torch.isfinite(starts).all(dim=1) & torch.isfinite(ends).all(dim=1) & (length > 1.0e-6)
    if not bool(valid_segment.any()):
        empty_indices = torch.zeros((0, 4), dtype=torch.long, device=device)
        empty_tangents = torch.zeros((0, 3), dtype=torch.float32, device=device)
        if validate and bool((batch.target_modes == _TARGET_MODE_DENSE_LINE).any()):
            raise ValueError("dense line target has no valid patch-overlapping segments")
        return empty_indices, empty_tangents, 0, 0

    starts = starts[valid_segment]
    ends = ends[valid_segment]
    delta = delta[valid_segment]
    sample_ids = sample_ids[valid_segment]
    steps = torch.ceil(torch.max(torch.abs(delta), dim=1).values).to(dtype=torch.long) + 1
    max_steps = int(steps.max().detach().cpu())
    step_index = torch.arange(max_steps, dtype=torch.float32, device=device)
    step_mask = step_index.view(1, -1) < steps.view(-1, 1)
    denom = (steps - 1).clamp_min(1).to(dtype=torch.float32).view(-1, 1)
    t = step_index.view(1, -1) / denom
    points = starts.view(-1, 1, 3) + t.unsqueeze(-1) * delta.view(-1, 1, 3)
    coords = torch.round(points).to(dtype=torch.long)
    bounds_shape = patch_shape if input_shape is None else input_shape
    in_bounds = (
        (coords[..., 0] >= 0)
        & (coords[..., 0] < int(bounds_shape[0]))
        & (coords[..., 1] >= 0)
        & (coords[..., 1] < int(bounds_shape[1]))
        & (coords[..., 2] >= 0)
        & (coords[..., 2] < int(bounds_shape[2]))
    )
    point_mask = step_mask & in_bounds
    if not bool(point_mask.any()):
        empty_indices = torch.zeros((0, 4), dtype=torch.long, device=device)
        empty_tangents = torch.zeros((0, 3), dtype=torch.float32, device=device)
        if validate and bool((batch.target_modes == _TARGET_MODE_DENSE_LINE).any()):
            raise ValueError("dense line target has no patch-overlapping positive voxels")
        return empty_indices, empty_tangents, int(valid_segment.sum().detach().cpu()), 0

    b = sample_ids.view(-1, 1).expand(-1, max_steps)[point_mask]
    zyx = coords[point_mask]
    if int(coordinate_scale) > 1:
        zyx = torch.div(zyx, int(coordinate_scale), rounding_mode="floor")
    tangents = delta.view(-1, 1, 3).expand(-1, max_steps, 3)[point_mask]
    indices = torch.cat([b.view(-1, 1), zyx], dim=1)
    indices, tangents = _unique_indices_and_tangents(
        indices,
        tangents,
        patch_shape=patch_shape,
    )
    positive[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = True
    return (
        indices,
        tangents,
        int(valid_segment.sum().detach().cpu()),
        int(indices.shape[0]),
    )


def _materialize_anisotropic_gaussian_presence(
    *,
    batch: FiberTrace3DBatch,
    config: FiberTrace3DConfig,
    patch_shape: tuple[int, int, int],
    coordinate_scale: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize a soft tube in the local fiber/sheet frame.

    The local surface normal is the thickness axis. The in-sheet axis
    perpendicular to the segment is the width axis. Finite segment caps use
    the width sigma so adjacent segments join without a hard endpoint seam.
    """

    if batch.target_segment_normals_zyx is None:
        raise ValueError("anisotropic Gaussian presence targets require segment normals")
    device = batch.volume.device
    starts = batch.target_segment_starts_zyx.to(device=device, dtype=torch.float32)
    ends = batch.target_segment_ends_zyx.to(device=device, dtype=torch.float32)
    normals = batch.target_segment_normals_zyx.to(device=device, dtype=torch.float32)
    if starts.shape != ends.shape or starts.shape != normals.shape:
        raise ValueError("Gaussian segment starts/ends/normals must have matching shapes")
    counts_cpu = batch.target_segment_counts.to(device="cpu", dtype=torch.long)
    sample_ids_cpu = torch.repeat_interleave(
        torch.arange(int(batch.volume.shape[0]), dtype=torch.long), counts_cpu
    )
    dense_cpu = batch.target_modes.to(device="cpu")[sample_ids_cpu] == _TARGET_MODE_DENSE_LINE
    selected_cpu = torch.nonzero(dense_cpu, as_tuple=False).flatten()
    selected = selected_cpu.to(device=device)
    starts, ends, normals = (value[selected] for value in (starts, ends, normals))
    sample_ids_cpu = sample_ids_cpu[selected_cpu]
    bbox_lo_cpu = batch.target_segment_bbox_lo_zyx.to(device="cpu", dtype=torch.long)[selected_cpu]
    bbox_hi_cpu = batch.target_segment_bbox_hi_zyx.to(device="cpu", dtype=torch.long)[selected_cpu]
    if int(dense_cpu.shape[0]) != int(batch.target_segment_starts_zyx.shape[0]):
        raise ValueError("Gaussian segment metadata count mismatch")
    target = torch.zeros(
        (int(batch.volume.shape[0]), *patch_shape), dtype=torch.float32, device=device
    )
    tangent_target = torch.zeros(
        (int(batch.volume.shape[0]), *patch_shape, 3), dtype=torch.float32, device=device
    )
    if int(starts.shape[0]) == 0:
        empty_indices = torch.zeros((0, 4), dtype=torch.long, device=device)
        empty_tangents = torch.zeros((0, 3), dtype=torch.float32, device=device)
        empty_weights = torch.zeros((0,), dtype=torch.float32, device=device)
        return target, empty_indices, empty_tangents, empty_weights

    scale = float(coordinate_scale)
    starts = starts / scale
    ends = ends / scale
    bbox_lo_cpu = torch.div(bbox_lo_cpu, int(coordinate_scale), rounding_mode="floor")
    bbox_hi_cpu = torch.div(
        bbox_hi_cpu + int(coordinate_scale) - 1,
        int(coordinate_scale),
        rounding_mode="floor",
    )
    sigma_thickness = float(config.presence_radius_voxels) / scale
    sigma_width = sigma_thickness * float(config.presence_gaussian_width_ratio)
    cutoff2 = float(config.presence_gaussian_truncate_sigma) ** 2
    grid = _zyx_grid(patch_shape, device=device, dtype=torch.float32)
    for segment in range(int(starts.shape[0])):
        lo = tuple(max(0, int(bbox_lo_cpu[segment, axis])) for axis in range(3))
        hi = tuple(min(patch_shape[axis], int(bbox_hi_cpu[segment, axis])) for axis in range(3))
        if any(hi[axis] <= lo[axis] for axis in range(3)):
            continue
        p0, p1 = starts[segment], ends[segment]
        vector = p1 - p0
        length = torch.linalg.vector_norm(vector).clamp_min(1.0e-6)
        tangent = vector / length
        normal = normals[segment] - torch.dot(normals[segment], tangent) * tangent
        normal = normal / torch.linalg.vector_norm(normal).clamp_min(1.0e-6)
        width = torch.linalg.cross(tangent, normal, dim=0)
        width = width / torch.linalg.vector_norm(width).clamp_min(1.0e-6)
        slices = tuple(slice(lo[axis], hi[axis]) for axis in range(3))
        coordinates = grid[slices]
        from_start = coordinates - p0
        along = torch.minimum(
            torch.sum(from_start * tangent, dim=-1).clamp_min(0.0),
            length,
        )
        delta = coordinates - (p0 + along.unsqueeze(-1) * tangent)
        standardized2 = (
            (torch.sum(delta * normal, dim=-1) / sigma_thickness) ** 2
            + (torch.sum(delta * width, dim=-1) / sigma_width) ** 2
            + (torch.sum(delta * tangent, dim=-1) / sigma_width) ** 2
        )
        probability = torch.where(
            standardized2 <= cutoff2,
            torch.exp(-0.5 * standardized2),
            torch.zeros_like(standardized2),
        )
        sample = int(sample_ids_cpu[segment])
        current = target[sample][slices]
        replace = probability > current
        target[sample][slices] = torch.where(replace, probability, current)
        tangent_view = tangent_target[sample][slices]
        tangent_target[sample][slices] = torch.where(
            replace.unsqueeze(-1), tangent.view(1, 1, 1, 3), tangent_view
        )
    support = target > 0.0
    indices_bzyx = torch.nonzero(support, as_tuple=False).to(dtype=torch.long)
    return target, indices_bzyx, tangent_target[support], target[support]


def _materialize_coarse_cp_only_sample(
    *,
    batch: FiberTrace3DBatch,
    sample_index: int,
    patch_shape: tuple[int, int, int],
    input_shape: tuple[int, int, int],
    output_stride: int,
    radius: float,
    positive: torch.Tensor,
    direction_indices: list[torch.Tensor],
    direction_tangents: list[torch.Tensor],
) -> int:
    device = batch.volume.device
    cp = batch.cp_local_zyx[sample_index].to(device=device, dtype=torch.float32)
    lo = torch.ceil(cp - float(radius)).to(dtype=torch.long)
    hi = torch.floor(cp + float(radius)).to(dtype=torch.long)
    axes = [
        torch.arange(
            max(0, int(lo[axis].item())),
            min(int(input_shape[axis]) - 1, int(hi[axis].item())) + 1,
            device=device,
        )
        for axis in range(3)
    ]
    if all(int(axis.numel()) > 0 for axis in axes):
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        source = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)
        source = source[
            torch.sum((source.to(dtype=torch.float32) - cp.view(1, 3)) ** 2, dim=1)
            <= float(radius) ** 2
        ]
    else:
        source = torch.empty((0, 3), dtype=torch.long, device=device)
    if int(source.shape[0]) == 0:
        nearest = torch.round(cp).to(dtype=torch.long)
        limits = torch.tensor(input_shape, dtype=torch.long, device=device) - 1
        source = torch.minimum(torch.maximum(nearest, torch.zeros_like(nearest)), limits).view(1, 3)
    coarse = torch.div(source, int(output_stride), rounding_mode="floor")
    linear = (coarse[:, 0] * patch_shape[1] + coarse[:, 1]) * patch_shape[2] + coarse[:, 2]
    order = torch.argsort(linear, stable=True)
    keep = torch.ones_like(order, dtype=torch.bool)
    keep[1:] = linear[order][1:] != linear[order][:-1]
    coarse = coarse[order[keep]]
    positive[sample_index, coarse[:, 0], coarse[:, 1], coarse[:, 2]] = True
    batch_column = torch.full(
        (int(coarse.shape[0]), 1), sample_index, dtype=torch.long, device=device
    )
    direction_indices.append(torch.cat([batch_column, coarse], dim=1))
    tangent = batch.target_tangent_zyx[sample_index].to(device=device, dtype=torch.float32)
    direction_tangents.append(tangent.view(1, 3).expand(int(coarse.shape[0]), 3))
    return int(coarse.shape[0])


def materialize_targets(
    batch: FiberTrace3DBatch,
    config: FiberTrace3DConfig,
    *,
    profile: bool = False,
    output_stride: int = 1,
) -> FiberTrace3DBatch:
    """Create sparse direction and dense presence targets from compact metadata."""

    if (
        batch.direction_indices_bzyx is not None
        and batch.direction_target_sparse is not None
        and batch.direction_weight_sparse is not None
        and batch.presence_target is not None
    ):
        return batch

    device = batch.volume.device
    output_stride = int(output_stride)
    if output_stride <= 0:
        raise ValueError("output_stride must be > 0")
    input_shape = tuple(int(v) for v in batch.volume.shape[-3:])
    patch_shape = tuple((size + output_stride - 1) // output_stride for size in input_shape)
    geometry_batch = batch
    radius = float(config.presence_radius_voxels)
    radius2 = radius * radius
    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    if profile:
        _sync_if_cuda(device)
    start = time.perf_counter()
    grid = _zyx_grid(patch_shape, device=device, dtype=torch.float32)
    positive = torch.zeros(
        (int(batch.volume.shape[0]), *patch_shape),
        dtype=torch.bool,
        device=device,
    )
    direction_indices: list[torch.Tensor] = []
    direction_tangents: list[torch.Tensor] = []
    if profile:
        _sync_if_cuda(device)
        timings["target_gpu_alloc_ms"] = (time.perf_counter() - start) * 1000.0
    line_start = time.perf_counter()
    line_indices, line_tangents, line_segments, line_points = _materialize_dense_line_indices(
        batch=geometry_batch,
        patch_shape=patch_shape,
        positive=positive,
        validate=bool(profile),
        coordinate_scale=output_stride,
        input_shape=input_shape,
    )
    if int(line_indices.shape[0]) > 0 and config.presence_target_mode != "anisotropic_gaussian":
        direction_indices.append(line_indices)
        direction_tangents.append(line_tangents)
    if profile:
        _sync_if_cuda(device)
        timings["target_line_index_ms"] = (time.perf_counter() - line_start) * 1000.0
        timings["target_line_segments"] = float(line_segments)
        timings["target_line_points"] = float(line_points)

    cp_start = time.perf_counter()
    cp_direction_points = 0
    for sample_index in range(int(batch.volume.shape[0])):
        mode = int(geometry_batch.target_modes[sample_index])
        if mode == _TARGET_MODE_CP_ONLY:
            if output_stride == 1:
                cp_direction_points += _materialize_cp_only_sample(
                    batch=geometry_batch,
                    sample_index=sample_index,
                    grid=grid,
                    patch_shape=patch_shape,
                    radius2=radius2,
                    positive=positive,
                    direction_indices=direction_indices,
                    direction_tangents=direction_tangents,
                )
            else:
                cp_direction_points += _materialize_coarse_cp_only_sample(
                    batch=batch,
                    sample_index=sample_index,
                    patch_shape=patch_shape,
                    input_shape=input_shape,
                    output_stride=output_stride,
                    radius=radius,
                    positive=positive,
                    direction_indices=direction_indices,
                    direction_tangents=direction_tangents,
                )
        elif mode == _TARGET_MODE_DENSE_LINE:
            continue
        else:
            raise ValueError(f"unknown 3D target mode {mode}")
    if profile:
        _sync_if_cuda(device)
        timings["target_cp_index_ms"] = (time.perf_counter() - cp_start) * 1000.0
        timings["target_cp_points"] = float(cp_direction_points)
    scatter_start = time.perf_counter()
    if config.presence_target_mode == "anisotropic_gaussian":
        presence, gaussian_indices, gaussian_tangents, gaussian_falloff = _materialize_anisotropic_gaussian_presence(
            batch=geometry_batch,
            config=config,
            patch_shape=patch_shape,
            coordinate_scale=output_stride,
        )
        if direction_indices:
            raise ValueError("Gaussian direction support currently requires dense-line-only batches")
        direction_indices.append(gaussian_indices)
        direction_tangents.append(gaussian_tangents)
    else:
        presence = positive.to(dtype=torch.float32)
        gaussian_falloff = None
    presence_target = presence.view(
        int(batch.volume.shape[0]),
        1,
        *patch_shape,
    )
    if profile:
        _sync_if_cuda(device)
        timings["target_presence_scatter_ms"] = (time.perf_counter() - scatter_start) * 1000.0

    encode_start = time.perf_counter()
    if direction_indices:
        indices_bzyx = torch.cat(direction_indices, dim=0)
        tangents_zyx = torch.cat(direction_tangents, dim=0)
        indices_bzyx, tangents_zyx = _unique_indices_and_tangents(
            indices_bzyx,
            tangents_zyx,
            patch_shape=patch_shape,
        )
        tangent_xyz = tangents_zyx[:, [2, 1, 0]]
        direction_target_sparse = encode_lasagna_direction_3x2(tangent_xyz)
        direction_weight_sparse = projection_magnitude_weights_3x2(tangent_xyz)
        if gaussian_falloff is not None:
            if int(gaussian_falloff.shape[0]) != int(direction_weight_sparse.shape[0]):
                raise ValueError("Gaussian direction falloff count mismatch")
            direction_weight_sparse = direction_weight_sparse * gaussian_falloff[:, None]
    else:
        indices_bzyx = torch.zeros((0, 4), dtype=torch.long, device=device)
        tangents_zyx = torch.zeros((0, 3), dtype=torch.float32, device=device)
        direction_target_sparse = torch.zeros((0, 6), dtype=torch.float32, device=device)
        direction_weight_sparse = torch.zeros((0, 6), dtype=torch.float32, device=device)
    if profile:
        _sync_if_cuda(device)
        timings["target_direction_encode_ms"] = (time.perf_counter() - encode_start) * 1000.0
    if profile:
        timings["target_gpu_positive_voxels"] = float(positive.sum().detach().cpu())
        timings["target_direction_points"] = float(indices_bzyx.shape[0])
        timings["target_gpu_total_ms"] = (time.perf_counter() - total_start) * 1000.0
    else:
        timings["target_gpu_submit_ms"] = (time.perf_counter() - total_start) * 1000.0

    profile_timings = dict(batch.profile_timings_ms or {})
    profile_timings.update(timings)
    return replace(
        batch,
        direction_target=None,
        direction_weight=None,
        direction_mask=None,
        direction_indices_bzyx=indices_bzyx,
        direction_target_sparse=direction_target_sparse,
        direction_weight_sparse=direction_weight_sparse,
        direction_tangent_sparse_zyx=tangents_zyx,
        presence_target=presence_target,
        presence_mask=None,
        profile_timings_ms=profile_timings if profile_timings else batch.profile_timings_ms,
    )


def require_materialized_targets(batch: FiberTrace3DBatch) -> None:
    missing = [
        name
        for name in (
            "direction_indices_bzyx",
            "direction_target_sparse",
            "direction_weight_sparse",
            "presence_target",
        )
        if getattr(batch, name) is None
    ]
    if missing:
        raise ValueError(
            "3D targets have not been materialized; missing "
            + ", ".join(missing)
        )
