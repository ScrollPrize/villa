from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate


def build_sample_offsets(
    neg_dist: float,
    pos_dist: float,
    sample_step: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    neg_dist = float(neg_dist)
    pos_dist = float(pos_dist)
    sample_step = float(sample_step)
    if sample_step <= 0:
        raise ValueError(f"sample_step must be > 0, got {sample_step!r}")
    if neg_dist < 0 or pos_dist < 0:
        raise ValueError(
            f"neg_dist and pos_dist must be >= 0, got {(neg_dist, pos_dist)!r}"
        )

    max_dist = neg_dist + pos_dist
    num_steps = max(1, int(round(max_dist / sample_step))) + 1
    return torch.linspace(
        -neg_dist,
        pos_dist,
        steps=num_steps,
        device=device,
        dtype=dtype,
    )


def local_points_zyx_to_normalized_grid(
    points_zyx: torch.Tensor,
    spatial_shape: Iterable[int],
    *,
    align_corners: bool = True,
) -> torch.Tensor:
    if not align_corners:
        raise ValueError("normal pooling currently requires align_corners=True")

    depth, height, width = (int(v) for v in spatial_shape)
    if min(depth, height, width) <= 0:
        raise ValueError(f"spatial_shape must be positive, got {(depth, height, width)!r}")

    z = points_zyx[..., 0]
    y = points_zyx[..., 1]
    x = points_zyx[..., 2]

    def _normalize(coord: torch.Tensor, size: int) -> torch.Tensor:
        if size == 1:
            return torch.zeros_like(coord)
        return (coord / float(size - 1)) * 2.0 - 1.0

    return torch.stack(
        [
            _normalize(x, width),
            _normalize(y, height),
            _normalize(z, depth),
        ],
        dim=-1,
    )


def build_normalized_grid_and_sample_valid(
    flat_points_local_zyx: torch.Tensor,
    flat_normals_local_zyx: torch.Tensor,
    flat_valid: torch.Tensor,
    offsets: torch.Tensor,
    spatial_shape: Iterable[int],
    *,
    align_corners: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not align_corners:
        raise ValueError("normal pooling currently requires align_corners=True")

    depth, height, width = (int(v) for v in spatial_shape)
    if min(depth, height, width) <= 0:
        raise ValueError(f"spatial_shape must be positive, got {(depth, height, width)!r}")

    offset_view = offsets.view(1, 1, 1, -1)
    grid_shape = (*flat_points_local_zyx.shape[:3], int(offsets.numel()), 3)
    grid = torch.empty(grid_shape, device=flat_points_local_zyx.device, dtype=flat_points_local_zyx.dtype)
    sample_valid = flat_valid.to(dtype=torch.bool).unsqueeze(-1).expand(*grid_shape[:-1]).clone()

    def _normalize(coord: torch.Tensor, size: int) -> torch.Tensor:
        if size == 1:
            return torch.zeros_like(coord)
        return (coord / float(size - 1)) * 2.0 - 1.0

    z = flat_points_local_zyx[..., 0].unsqueeze(-1) + flat_normals_local_zyx[..., 0].unsqueeze(-1) * offset_view
    sample_valid &= torch.isfinite(z)
    sample_valid &= (z >= 0) & (z <= depth - 1)
    grid[..., 2] = _normalize(z, depth)

    y = flat_points_local_zyx[..., 1].unsqueeze(-1) + flat_normals_local_zyx[..., 1].unsqueeze(-1) * offset_view
    sample_valid &= torch.isfinite(y)
    sample_valid &= (y >= 0) & (y <= height - 1)
    grid[..., 1] = _normalize(y, height)

    x = flat_points_local_zyx[..., 2].unsqueeze(-1) + flat_normals_local_zyx[..., 2].unsqueeze(-1) * offset_view
    sample_valid &= torch.isfinite(x)
    sample_valid &= (x >= 0) & (x <= width - 1)
    grid[..., 0] = _normalize(x, width)

    return grid, sample_valid


def pool_logits_along_normals(
    logits_3d: torch.Tensor,
    flat_points_local_zyx: torch.Tensor,
    flat_normals_local_zyx: torch.Tensor,
    flat_valid: torch.Tensor,
    *,
    neg_dist: float,
    pos_dist: float,
    sample_step: float,
    align_corners: bool = True,
    timer=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits_3d.ndim != 5:
        raise ValueError(
            f"Expected logits_3d with shape [B, C, Z, Y, X], got {tuple(logits_3d.shape)}"
        )
    if logits_3d.shape[1] != 1:
        raise ValueError(
            f"Expected a single-channel logit volume, got shape {tuple(logits_3d.shape)}"
        )

    if flat_points_local_zyx.ndim != 4 or flat_points_local_zyx.shape[-1] != 3:
        raise ValueError(
            "flat_points_local_zyx must have shape [B, H, W, 3], "
            f"got {tuple(flat_points_local_zyx.shape)}"
        )
    if flat_normals_local_zyx.ndim != 4 or flat_normals_local_zyx.shape[-1] != 3:
        raise ValueError(
            "flat_normals_local_zyx must have shape [B, H, W, 3], "
            f"got {tuple(flat_normals_local_zyx.shape)}"
        )

    if flat_valid.ndim == 4:
        if flat_valid.shape[1] != 1:
            raise ValueError(
                f"flat_valid must have shape [B, 1, H, W] or [B, H, W], got {tuple(flat_valid.shape)}"
            )
        flat_valid = flat_valid[:, 0]
    elif flat_valid.ndim != 3:
        raise ValueError(
            f"flat_valid must have shape [B, 1, H, W] or [B, H, W], got {tuple(flat_valid.shape)}"
        )

    batch_size, _, depth, height, width = logits_3d.shape
    if flat_points_local_zyx.shape[0] != batch_size or flat_normals_local_zyx.shape[0] != batch_size:
        raise ValueError("Batch size mismatch between logits and flat geometry tensors")

    def timed(section_name: str):
        if timer is None:
            return nullcontext()
        return timer(section_name, logits_3d.device)

    with timed('normal_pooling/build_sample_offsets'):
        offsets = build_sample_offsets(
            neg_dist,
            pos_dist,
            sample_step,
            device=logits_3d.device,
            dtype=logits_3d.dtype,
        )
    with timed('normal_pooling/build_grid_and_valid'):
        grid, sample_valid = build_normalized_grid_and_sample_valid(
            flat_points_local_zyx,
            flat_normals_local_zyx,
            flat_valid,
            offsets,
            (depth, height, width),
            align_corners=align_corners,
        )
    with timed('normal_pooling/grid_sample'):
        sampled_logits = F.grid_sample(
            logits_3d,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        )

    with timed('normal_pooling/reduce_samples'):
        invalid_fill = torch.finfo(sampled_logits.dtype).min
        sampled_logits = sampled_logits.masked_fill(~sample_valid.unsqueeze(1), invalid_fill)

        pooled_logits = sampled_logits.amax(dim=-1)
        pooled_valid = sample_valid.any(dim=-1, keepdim=False).unsqueeze(1)
        pooled_logits = torch.where(pooled_valid, pooled_logits, torch.zeros_like(pooled_logits))
    return pooled_logits, pooled_valid.to(dtype=logits_3d.dtype)


def collate_normal_pooled_batch(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    profile_timings = None
    if any('profile_timings' in sample for sample in batch):
        profile_timings = [sample.get('profile_timings', {}) for sample in batch]

    max_height = max(int(sample['flat_target'].shape[-2]) for sample in batch)
    max_width = max(int(sample['flat_target'].shape[-1]) for sample in batch)

    padded_batch = []
    for sample in batch:
        padded = dict(sample)
        padded.pop('profile_timings', None)
        pad_h = max_height - int(sample['flat_target'].shape[-2])
        pad_w = max_width - int(sample['flat_target'].shape[-1])
        pad = (0, pad_w, 0, pad_h)

        for key in ('flat_target', 'flat_supervision', 'flat_valid'):
            padded[key] = F.pad(sample[key], pad)

        for key in ('flat_points_local_zyx', 'flat_normals_local_zyx'):
            value = sample[key].permute(2, 0, 1)
            value = F.pad(value, pad)
            padded[key] = value.permute(1, 2, 0)

        padded_batch.append(padded)

    collated = default_collate(padded_batch)
    if profile_timings is not None:
        collated['profile_timings'] = profile_timings
    return collated
