import random
from typing import Any, Mapping

import numpy as np


def compute_surface_normals(surface_zyxs: np.ndarray) -> np.ndarray:
    """Estimate per-point unit normals from local row/col tangents."""
    h, w, _ = surface_zyxs.shape
    if h < 2 or w < 2:
        return np.zeros_like(surface_zyxs, dtype=np.float32)

    surface = surface_zyxs.astype(np.float32, copy=False)
    row_tangent = np.empty_like(surface)
    col_tangent = np.empty_like(surface)

    row_tangent[1:-1] = surface[2:] - surface[:-2]
    row_tangent[0] = surface[1] - surface[0]
    row_tangent[-1] = surface[-1] - surface[-2]

    col_tangent[:, 1:-1] = surface[:, 2:] - surface[:, :-2]
    col_tangent[:, 0] = surface[:, 1] - surface[:, 0]
    col_tangent[:, -1] = surface[:, -1] - surface[:, -2]

    normals = np.cross(col_tangent, row_tangent)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)
    normals[norms[..., 0] <= 1e-6] = 0.0
    return normals.astype(np.float32, copy=False)


def maybe_perturb_conditioning_surface(
    cond_zyxs: np.ndarray,
    config: Mapping[str, Any],
    apply_augmentation: bool,
) -> np.ndarray:
    """Apply local normal-direction pushes with Gaussian falloff on small regions."""
    cfg = config.get("cond_local_perturb", {})
    apply_without_aug = bool(cfg.get("apply_without_augmentation", False))
    if (not apply_augmentation and not apply_without_aug) or not cfg.get("enabled", True):
        return cond_zyxs
    if random.random() >= float(cfg.get("probability", 0.35)):
        return cond_zyxs

    cond_h, cond_w, _ = cond_zyxs.shape
    if cond_h < 2 or cond_w < 2:
        return cond_zyxs

    normals = compute_surface_normals(cond_zyxs)
    valid_normal_idx = np.argwhere(np.linalg.norm(normals, axis=-1) > 1e-6)
    if len(valid_normal_idx) == 0:
        return cond_zyxs

    blob_cfg = cfg.get("num_blobs", [1, 3])
    if isinstance(blob_cfg, (list, tuple)) and len(blob_cfg) == 2:
        min_blobs = max(1, int(blob_cfg[0]))
        max_blobs = max(min_blobs, int(blob_cfg[1]))
    else:
        min_blobs = max_blobs = 1
    n_blobs = random.randint(min_blobs, max_blobs)

    sigma_cfg = cfg.get("sigma_fraction_range", [0.04, 0.10])
    if isinstance(sigma_cfg, (list, tuple)) and len(sigma_cfg) == 2:
        sigma_lo_frac = max(0.01, float(sigma_cfg[0]))
        sigma_hi_frac = max(sigma_lo_frac, float(sigma_cfg[1]))
    else:
        sigma_lo_frac, sigma_hi_frac = 0.04, 0.10
    sigma_scale = float(min(cond_h, cond_w))
    sigma_lo = max(0.3, sigma_lo_frac * sigma_scale)
    sigma_hi = max(sigma_lo, sigma_hi_frac * sigma_scale)

    amp_cfg = cfg.get("amplitude_range", [0.25, 1.25])
    if isinstance(amp_cfg, (list, tuple)) and len(amp_cfg) == 2:
        amp_lo = max(0.0, float(amp_cfg[0]))
        amp_hi = max(amp_lo, float(amp_cfg[1]))
    else:
        amp_lo, amp_hi = 0.25, 1.25

    radius_sigma_mult = max(0.5, float(cfg.get("radius_sigma_mult", 2.5)))
    max_total_disp = max(0.0, float(cfg.get("max_total_displacement", 1.5)))
    if max_total_disp <= 0.0:
        return cond_zyxs

    rr, cc = np.meshgrid(np.arange(cond_h), np.arange(cond_w), indexing="ij")
    disp_along_normal = np.zeros((cond_h, cond_w), dtype=np.float32)
    points_affected = int(cfg.get("points_affected", 10))
    use_k_neighborhood = points_affected > 0

    for _ in range(n_blobs):
        seed_r, seed_c = valid_normal_idx[np.random.randint(len(valid_normal_idx))]

        dr = rr - float(seed_r)
        dc = cc - float(seed_c)
        dist2 = dr * dr + dc * dc

        if use_k_neighborhood:
            flat_dist2 = dist2.reshape(-1)
            k = min(points_affected, flat_dist2.size)
            if k <= 0:
                continue
            kth_idx = np.argpartition(flat_dist2, k - 1)[k - 1]
            radius2 = float(flat_dist2[kth_idx])
            local_mask = dist2 <= radius2
            sigma = max(0.3, np.sqrt(max(radius2, 1e-6)) / max(radius_sigma_mult, 1e-3))
        else:
            sigma = random.uniform(sigma_lo, sigma_hi)
            radius2 = (radius_sigma_mult * sigma) ** 2
            local_mask = dist2 <= radius2

        if not np.any(local_mask):
            continue

        amp = random.uniform(amp_lo, amp_hi)
        signed_amp = amp if random.random() < 0.5 else -amp
        falloff = np.exp(-0.5 * dist2 / max(sigma * sigma, 1e-6))
        disp_along_normal[local_mask] += (signed_amp * falloff[local_mask]).astype(np.float32)

    if not np.any(disp_along_normal):
        return cond_zyxs

    disp_along_normal = np.clip(disp_along_normal, -max_total_disp, max_total_disp)
    perturbed = cond_zyxs.astype(np.float32, copy=True)
    perturbed += normals * disp_along_normal[..., None]
    return perturbed.astype(cond_zyxs.dtype, copy=False)
