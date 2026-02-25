import random

import numpy as np


def _estimate_mean_unit_direction_from_field(disp_np: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Estimate one robust mean unit direction from a 3-channel dense field."""
    disp = np.asarray(disp_np, dtype=np.float32)
    if disp.ndim != 4 or disp.shape[0] != 3:
        raise ValueError(f"disp_np must have shape (3, D, H, W), got {tuple(disp.shape)}")
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != tuple(disp.shape[1:]):
        raise ValueError(
            "mask shape must match displacement spatial dims "
            f"{tuple(disp.shape[1:])}, got {tuple(mask_bool.shape)}"
        )
    if not bool(mask_bool.any()):
        return None

    vecs = disp[:, mask_bool].T  # [N, 3]
    if vecs.size == 0:
        return None
    finite = np.isfinite(vecs).all(axis=1)
    vecs = vecs[finite]
    if vecs.shape[0] == 0:
        return None

    mags = np.linalg.norm(vecs, axis=1)
    vecs = vecs[mags > 1e-6]
    if vecs.shape[0] == 0:
        return None

    unit_vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-6)
    mean_vec = np.mean(unit_vecs, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(mean_vec))
    if not np.isfinite(norm) or norm <= 1e-6:
        return None
    return (mean_vec / norm).astype(np.float32, copy=False)


def _compute_surface_tangent_axis(surface_grid: np.ndarray, surface_valid: np.ndarray, axis: int):
    """Estimate local tangent vectors along one grid axis."""
    grid = np.asarray(surface_grid, dtype=np.float32)
    valid = np.asarray(surface_valid, dtype=bool)
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise ValueError(f"surface_grid must have shape (H, W, 3), got {tuple(grid.shape)}")
    if valid.shape != grid.shape[:2]:
        raise ValueError(
            f"surface_valid shape {tuple(valid.shape)} must match grid shape {tuple(grid.shape[:2])}"
        )
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis!r}")

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


def estimate_global_unit_normal_from_surface_grid(surface_grid: np.ndarray) -> np.ndarray:
    """Estimate a global unit normal from one ordered surface grid.

    Returns a zero vector when no stable estimate can be formed.
    """
    grid = np.asarray(surface_grid, dtype=np.float32)
    if grid.ndim != 3 or grid.shape[2] != 3:
        return np.zeros((3,), dtype=np.float32)

    valid = np.isfinite(grid).all(axis=2)
    if not bool(valid.any()):
        return np.zeros((3,), dtype=np.float32)

    row_tangent, row_tangent_valid = _compute_surface_tangent_axis(grid, valid, axis=0)
    col_tangent, col_tangent_valid = _compute_surface_tangent_axis(grid, valid, axis=1)
    normals = np.cross(col_tangent, row_tangent)
    norms = np.linalg.norm(normals, axis=2)
    finite = np.isfinite(normals).all(axis=2) & np.isfinite(norms)
    normals_valid = valid & row_tangent_valid & col_tangent_valid & finite & (norms > 1e-6)
    if not bool(normals_valid.any()):
        return np.zeros((3,), dtype=np.float32)

    unit_normals = normals[normals_valid] / norms[normals_valid, None]
    mean_vec = np.mean(unit_normals, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    mean_norm = float(np.linalg.norm(mean_vec))
    if not np.isfinite(mean_norm) or mean_norm <= 1e-6:
        return np.zeros((3,), dtype=np.float32)
    return (mean_vec / mean_norm).astype(np.float32, copy=False)


def estimate_triplet_unit_direction(disp_np: np.ndarray, cond_mask: np.ndarray) -> np.ndarray:
    """Estimate one robust unit direction from dense displacement on conditioning voxels."""
    disp = np.asarray(disp_np, dtype=np.float32)
    mask = np.asarray(cond_mask, dtype=bool)
    if disp.ndim != 4 or disp.shape[0] != 3:
        raise ValueError(f"disp_np must have shape (3, D, H, W), got {tuple(disp.shape)}")
    if mask.shape != tuple(disp.shape[1:]):
        raise ValueError(
            f"cond_mask shape must match displacement spatial dims {tuple(disp.shape[1:])}, got {tuple(mask.shape)}"
        )
    if not bool(mask.any()):
        return np.zeros((3,), dtype=np.float32)

    vecs = disp[:, mask].T  # [N, 3]
    if vecs.size == 0:
        return np.zeros((3,), dtype=np.float32)
    finite = np.isfinite(vecs).all(axis=1)
    vecs = vecs[finite]
    if vecs.shape[0] == 0:
        return np.zeros((3,), dtype=np.float32)

    mags = np.linalg.norm(vecs, axis=1)
    vecs = vecs[mags > 1e-6]
    if vecs.shape[0] == 0:
        return np.zeros((3,), dtype=np.float32)

    unit_vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-6)
    mean_vec = np.mean(unit_vecs, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(mean_vec))
    if not np.isfinite(norm) or norm <= 1e-6:
        return np.zeros((3,), dtype=np.float32)
    return (mean_vec / norm).astype(np.float32, copy=False)


def build_triplet_direction_priors(
    crop_size,
    cond_mask: np.ndarray,
    ch0_dir: np.ndarray,
    ch1_dir: np.ndarray,
    mask_mode: str = "cond",
) -> np.ndarray:
    """Build broadcast direction priors for 2 triplet displacement branches."""
    crop_size = tuple(int(v) for v in crop_size)
    if len(crop_size) != 3:
        raise ValueError(f"crop_size must be length 3, got {crop_size}")
    priors = np.zeros((6, *crop_size), dtype=np.float32)
    v0 = np.asarray(ch0_dir, dtype=np.float32).reshape(3)
    v1 = np.asarray(ch1_dir, dtype=np.float32).reshape(3)
    for axis in range(3):
        priors[axis, ...] = v0[axis]
        priors[axis + 3, ...] = v1[axis]

    mode = str(mask_mode).lower()
    if mode == "cond":
        mask = np.asarray(cond_mask, dtype=np.float32)
        if mask.shape != crop_size:
            raise ValueError(f"cond_mask shape must match crop_size {crop_size}, got {tuple(mask.shape)}")
        priors *= mask[None]
    elif mode != "full":
        raise ValueError(f"mask_mode must be 'cond' or 'full', got {mask_mode!r}")
    return priors


def build_triplet_direction_priors_from_displacements(
    crop_size,
    cond_mask: np.ndarray,
    behind_disp_np: np.ndarray,
    front_disp_np: np.ndarray,
    mask_mode: str = "cond",
) -> np.ndarray:
    ch0_dir = estimate_triplet_unit_direction(behind_disp_np, cond_mask)
    ch1_dir = estimate_triplet_unit_direction(front_disp_np, cond_mask)
    return build_triplet_direction_priors(
        crop_size,
        cond_mask,
        ch0_dir,
        ch1_dir,
        mask_mode=mask_mode,
    )


def build_triplet_direction_priors_from_conditioning_surface(
    crop_size,
    cond_mask: np.ndarray,
    cond_surface_local: np.ndarray,
    mask_mode: str = "cond",
) -> np.ndarray | None:
    """Build triplet priors from conditioning-surface geometry only (no neighbor GT)."""
    global_unit_normal = estimate_global_unit_normal_from_surface_grid(cond_surface_local)
    if float(np.linalg.norm(global_unit_normal)) <= 1e-6:
        return None
    return build_triplet_direction_priors(
        crop_size,
        cond_mask,
        global_unit_normal,
        -global_unit_normal,
        mask_mode=mask_mode,
    )


def swap_triplet_branch_channels(
    dense_gt_np: np.ndarray,
    dir_priors_np: np.ndarray = None,
):
    """Swap branch channel groups [0:3] and [3:6] for GT and optional priors."""
    dense = np.asarray(dense_gt_np, dtype=np.float32)
    if dense.ndim != 4 or dense.shape[0] < 6:
        raise ValueError(f"dense_gt_np must have at least 6 channels, got shape {tuple(dense.shape)}")
    swapped_dense = np.concatenate([dense[3:6], dense[0:3]], axis=0).astype(np.float32, copy=False)
    if dir_priors_np is None:
        return swapped_dense, None
    priors = np.asarray(dir_priors_np, dtype=np.float32)
    if priors.ndim != 4 or priors.shape[0] != 6:
        raise ValueError(f"dir_priors_np must have shape (6, D, H, W), got {tuple(priors.shape)}")
    swapped_priors = np.concatenate([priors[3:6], priors[0:3]], axis=0).astype(np.float32, copy=False)
    return swapped_dense, swapped_priors


def maybe_swap_triplet_branch_channels(
    dense_gt_np: np.ndarray,
    dir_priors_np: np.ndarray = None,
    swap_prob: float = 0.0,
    rng=None,
):
    """Randomly swap triplet branch channels, returning updated tensors and channel order."""
    p = float(swap_prob)
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"swap_prob must satisfy 0 <= p <= 1, got {swap_prob!r}")
    if rng is None:
        rng = random

    triplet_channel_order_np = np.array([0, 1], dtype=np.int64)
    if p > 0.0 and float(rng.random()) < p:
        dense_gt_np, dir_priors_np = swap_triplet_branch_channels(dense_gt_np, dir_priors_np)
        triplet_channel_order_np = np.array([1, 0], dtype=np.int64)
    return dense_gt_np, dir_priors_np, triplet_channel_order_np


def align_triplet_branch_channels_to_priors(
    dense_gt_np: np.ndarray,
    dir_priors_np: np.ndarray,
    cond_mask: np.ndarray | None = None,
):
    """Deterministically align triplet GT branch order to prior slots.

    Returns:
        dense_gt_np: possibly swapped dense GT channels
        dir_priors_np: unchanged
        channel_order_np: mapping from current channels to original branch ids
            ([0, 1] for no swap, [1, 0] when swapped)
    """
    dense = np.asarray(dense_gt_np, dtype=np.float32)
    priors = np.asarray(dir_priors_np, dtype=np.float32)
    if dense.ndim != 4 or dense.shape[0] < 6:
        raise ValueError(f"dense_gt_np must have at least 6 channels, got shape {tuple(dense.shape)}")
    if priors.ndim != 4 or priors.shape[0] != 6:
        raise ValueError(f"dir_priors_np must have shape (6, D, H, W), got {tuple(priors.shape)}")
    if dense.shape[1:] != priors.shape[1:]:
        raise ValueError(
            "dense_gt_np and dir_priors_np must share spatial shape, got "
            f"{tuple(dense.shape[1:])} vs {tuple(priors.shape[1:])}"
        )

    if cond_mask is None:
        mask = np.any(np.abs(priors) > 0, axis=0)
    else:
        mask = np.asarray(cond_mask, dtype=bool)
        if mask.shape != tuple(dense.shape[1:]):
            raise ValueError(
                "cond_mask shape must match spatial shape "
                f"{tuple(dense.shape[1:])}, got {tuple(mask.shape)}"
            )
    if not bool(mask.any()):
        return dense_gt_np, dir_priors_np, np.array([0, 1], dtype=np.int64)

    g0 = _estimate_mean_unit_direction_from_field(dense[0:3], mask)
    g1 = _estimate_mean_unit_direction_from_field(dense[3:6], mask)
    p0 = _estimate_mean_unit_direction_from_field(priors[0:3], mask)
    p1 = _estimate_mean_unit_direction_from_field(priors[3:6], mask)
    if any(v is None for v in (g0, g1, p0, p1)):
        return dense_gt_np, dir_priors_np, np.array([0, 1], dtype=np.int64)

    keep_score = float(np.dot(g0, p0) + np.dot(g1, p1))
    swap_score = float(np.dot(g0, p1) + np.dot(g1, p0))
    if swap_score > keep_score:
        dense_swapped = np.concatenate([dense[3:6], dense[0:3]], axis=0).astype(np.float32, copy=False)
        return dense_swapped, dir_priors_np, np.array([1, 0], dtype=np.int64)
    return dense_gt_np, dir_priors_np, np.array([0, 1], dtype=np.int64)
