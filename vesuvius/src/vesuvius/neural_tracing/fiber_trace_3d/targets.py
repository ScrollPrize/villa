from __future__ import annotations

import math
import time
from dataclasses import replace

import torch

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


def _presence_loss_mask(
    valid_mask: torch.Tensor,
    *,
    patch_shape: tuple[int, int, int],
    config: FiberTrace3DConfig,
    target_modes: torch.Tensor | None = None,
) -> torch.Tensor:
    presence_mask = valid_mask.clone()
    edge_margin = config.presence_negative_edge_margin_voxels
    if edge_margin is None:
        edge_margin = int(math.ceil(max(config.augment_shift_zyx)))
    if edge_margin > 0:
        device = valid_mask.device
        z = torch.arange(patch_shape[0], device=device)
        y = torch.arange(patch_shape[1], device=device)
        x = torch.arange(patch_shape[2], device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        interior = (
            (zz >= edge_margin)
            & (zz < patch_shape[0] - edge_margin)
            & (yy >= edge_margin)
            & (yy < patch_shape[1] - edge_margin)
            & (xx >= edge_margin)
            & (xx < patch_shape[2] - edge_margin)
        )
        interior = interior.view(1, 1, *patch_shape)
        if target_modes is None:
            presence_mask = presence_mask & interior
        else:
            edge_limited = (
                target_modes.to(device=device, dtype=torch.long) == _TARGET_MODE_CP_ONLY
            ).view(-1, 1, 1, 1, 1)
            presence_mask = presence_mask & (~edge_limited | interior)
    return presence_mask


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
    local_indices = torch.nonzero(sample_positive & batch.valid_mask[sample_index, 0], as_tuple=False)
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
    in_bounds = (
        (coords[..., 0] >= 0)
        & (coords[..., 0] < int(patch_shape[0]))
        & (coords[..., 1] >= 0)
        & (coords[..., 1] < int(patch_shape[1]))
        & (coords[..., 2] >= 0)
        & (coords[..., 2] < int(patch_shape[2]))
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
    tangents = delta.view(-1, 1, 3).expand(-1, max_steps, 3)[point_mask]
    indices = torch.cat([b.view(-1, 1), zyx], dim=1)
    indices, tangents = _unique_indices_and_tangents(
        indices,
        tangents,
        patch_shape=patch_shape,
    )
    positive[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = True
    valid = batch.valid_mask[indices[:, 0], 0, indices[:, 1], indices[:, 2], indices[:, 3]]
    return (
        indices[valid],
        tangents[valid],
        int(valid_segment.sum().detach().cpu()),
        int(indices.shape[0]),
    )


def materialize_targets(
    batch: FiberTrace3DBatch,
    config: FiberTrace3DConfig,
    *,
    profile: bool = False,
) -> FiberTrace3DBatch:
    """Create sparse direction and dense presence targets from compact metadata."""

    if (
        batch.direction_indices_bzyx is not None
        and batch.direction_target_sparse is not None
        and batch.direction_weight_sparse is not None
        and batch.presence_target is not None
        and batch.presence_mask is not None
    ):
        return batch

    device = batch.volume.device
    patch_shape = tuple(int(v) for v in batch.volume.shape[-3:])
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
        batch=batch,
        patch_shape=patch_shape,
        positive=positive,
        validate=bool(profile),
    )
    if int(line_indices.shape[0]) > 0:
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
        mode = int(batch.target_modes[sample_index])
        if mode == _TARGET_MODE_CP_ONLY:
            cp_direction_points += _materialize_cp_only_sample(
                batch=batch,
                sample_index=sample_index,
                grid=grid,
                patch_shape=patch_shape,
                radius2=radius2,
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
    presence_target = positive.to(dtype=torch.float32).view(
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
    else:
        indices_bzyx = torch.zeros((0, 4), dtype=torch.long, device=device)
        tangents_zyx = torch.zeros((0, 3), dtype=torch.float32, device=device)
        direction_target_sparse = torch.zeros((0, 6), dtype=torch.float32, device=device)
        direction_weight_sparse = torch.zeros((0, 6), dtype=torch.float32, device=device)
    if profile:
        _sync_if_cuda(device)
        timings["target_direction_encode_ms"] = (time.perf_counter() - encode_start) * 1000.0
    mask_start = time.perf_counter()
    presence_mask = _presence_loss_mask(
        batch.valid_mask.to(dtype=torch.bool),
        patch_shape=patch_shape,
        config=config,
        target_modes=batch.target_modes,
    )
    if profile:
        _sync_if_cuda(device)
        timings["target_gpu_mask_ms"] = (time.perf_counter() - mask_start) * 1000.0
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
        presence_mask=presence_mask,
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
            "presence_mask",
        )
        if getattr(batch, name) is None
    ]
    if missing:
        raise ValueError(
            "3D targets have not been materialized; missing "
            + ", ".join(missing)
        )
