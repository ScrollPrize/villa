import torch
import torch.nn.functional as F
import numpy as np

def _build_normal_pool_offsets(distance_pair, sample_step, *, device, dtype):
    pos_distance = max(0.0, float(distance_pair[0]))
    neg_distance = max(0.0, float(distance_pair[1]))
    sample_step = float(sample_step)
    if sample_step <= 0.0:
        raise ValueError(f"normal_pool_step must be > 0, got {sample_step}")

    offsets = np.arange(-neg_distance, pos_distance + (0.5 * sample_step), sample_step, dtype=np.float32)
    anchors = np.asarray([-neg_distance, 0.0, pos_distance], dtype=np.float32)
    offsets = np.unique(np.round(np.concatenate([offsets, anchors], axis=0), decimals=6))
    offsets.sort()
    return torch.as_tensor(offsets, device=device, dtype=dtype)


def _get_normal_pool_offsets(offset_cache, distance_pair, sample_step, *, device, dtype):
    cache_key = (device.type, device.index, dtype)
    offsets = offset_cache.get(cache_key)
    if offsets is None:
        offsets = _build_normal_pool_offsets(
            distance_pair,
            sample_step,
            device=device,
            dtype=dtype,
        )
        offset_cache[cache_key] = offsets
    return offsets

def _pool_logits_along_surface_normals(
    logits,
    surface_points_zyx,
    surface_normals_zyx,
    offsets,
    *,
    reduction,
    temperature,
):
    if logits.ndim != 5:
        raise ValueError(
            f"mode='normal_pooled_3d' expects 3D logits with shape [B, C, Z, Y, X], got {tuple(logits.shape)}"
        )
    if surface_points_zyx.ndim != 4 or surface_points_zyx.shape[-1] != 3:
        raise ValueError(
            f"Expected surface_points_zyx shape [B, H, W, 3], got {tuple(surface_points_zyx.shape)}"
        )
    if surface_normals_zyx.shape != surface_points_zyx.shape:
        raise ValueError(
            f"surface_normals_zyx shape {tuple(surface_normals_zyx.shape)} must match surface_points_zyx "
            f"{tuple(surface_points_zyx.shape)}"
        )

    logits = logits.float()
    surface_points_zyx = surface_points_zyx.to(device=logits.device, dtype=logits.dtype)
    surface_normals_zyx = surface_normals_zyx.to(device=logits.device, dtype=logits.dtype)
    offsets = offsets.to(device=logits.device, dtype=logits.dtype)

    ray_points = (
        surface_points_zyx[..., None, :]
        + (surface_normals_zyx[..., None, :] * offsets.view(1, 1, 1, -1, 1))
    )
    z = ray_points[..., 0]
    y = ray_points[..., 1]
    x = ray_points[..., 2]

    size_z, size_y, size_x = logits.shape[-3:]
    ray_valid = (
        (z >= 0.0) & (z <= float(size_z - 1))
        & (y >= 0.0) & (y <= float(size_y - 1))
        & (x >= 0.0) & (x <= float(size_x - 1))
    )
    denom_z = max(size_z - 1, 1)
    denom_y = max(size_y - 1, 1)
    denom_x = max(size_x - 1, 1)
    z_norm = (2.0 * z / denom_z) - 1.0
    y_norm = (2.0 * y / denom_y) - 1.0
    x_norm = (2.0 * x / denom_x) - 1.0
    grid = torch.stack([x_norm, y_norm, z_norm], dim=-1).permute(0, 3, 1, 2, 4)

    ray_logits = F.grid_sample(
        logits,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    ray_valid = ray_valid.permute(0, 3, 1, 2).unsqueeze(1)
    masked_ray_logits = torch.where(
        ray_valid,
        ray_logits,
        torch.full_like(ray_logits, float("-inf")),
    )
    any_valid = ray_valid.any(dim=2)
    if reduction == "max":
        pooled = masked_ray_logits.amax(dim=2)
        pooled = torch.where(any_valid, pooled, torch.zeros_like(pooled))
        return pooled, any_valid
    if reduction == "logsumexp":
        if temperature <= 0.0:
            raise ValueError(
                f"normal_pool_temperature must be > 0 when using logsumexp pooling, got {temperature}"
            )
        pooled = temperature * torch.logsumexp(masked_ray_logits / temperature, dim=2)
        pooled = torch.where(any_valid, pooled, torch.zeros_like(pooled))
        return pooled, any_valid
    raise ValueError(
        f"Unsupported normal_pool_reduction {reduction!r}; expected 'logsumexp' or 'max'"
    )
