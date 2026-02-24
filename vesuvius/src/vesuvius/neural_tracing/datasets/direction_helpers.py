import random

import numpy as np


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
