from __future__ import annotations

import numpy as np

from vesuvius.neural_tracing.autoreg_mesh.serialization import (
    IGNORE_INDEX,
    decode_local_xyz,
    quantize_local_xyz,
)


def serialize_fiber_example(
    points_zyx_local: np.ndarray,
    *,
    prompt_length: int,
    target_length: int | None,
    volume_shape,
    patch_size,
    offset_num_bins,
) -> dict:
    points = np.asarray(points_zyx_local, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_zyx_local must have shape (N, 3), got {points.shape}")
    prompt_len = int(prompt_length)
    if prompt_len <= 0:
        raise ValueError("prompt_length must be positive")
    if points.shape[0] <= prompt_len:
        raise ValueError("fiber window must contain at least one target point")
    if target_length is None:
        target_len = int(points.shape[0] - prompt_len)
    else:
        target_len = int(target_length)
    if target_len <= 0:
        raise ValueError("target_length must be positive")
    if points.shape[0] < prompt_len + target_len:
        raise ValueError(
            f"fiber window has {points.shape[0]} points but needs prompt+target={prompt_len + target_len}"
        )

    prompt_xyz = points[:prompt_len]
    target_xyz = points[prompt_len:prompt_len + target_len]
    prompt_positions = np.arange(prompt_len, dtype=np.int64)
    target_positions = np.arange(prompt_len, prompt_len + target_len, dtype=np.int64)

    prompt_coarse_ids, prompt_offset_bins, prompt_valid_mask = quantize_local_xyz(
        prompt_xyz,
        volume_shape=volume_shape,
        patch_size=patch_size,
        offset_num_bins=offset_num_bins,
    )
    target_coarse_ids, target_offset_bins, target_valid_mask = quantize_local_xyz(
        target_xyz,
        volume_shape=volume_shape,
        patch_size=patch_size,
        offset_num_bins=offset_num_bins,
    )
    target_bin_center_xyz = decode_local_xyz(
        np.where(target_valid_mask, target_coarse_ids, 0),
        np.where(target_valid_mask[:, None], target_offset_bins, 0),
        volume_shape=volume_shape,
        patch_size=patch_size,
        offset_num_bins=offset_num_bins,
    )
    target_stop = np.zeros((target_len,), dtype=np.float32)
    valid_indices = np.flatnonzero(target_valid_mask)
    if valid_indices.size > 0:
        target_stop[int(valid_indices[-1])] = 1.0

    return {
        "prompt_tokens": {
            "coarse_ids": prompt_coarse_ids.astype(np.int64, copy=False),
            "offset_bins": prompt_offset_bins.astype(np.int64, copy=False),
            "xyz": prompt_xyz.astype(np.float32, copy=False),
            "positions": prompt_positions,
            "valid_mask": prompt_valid_mask.astype(bool, copy=False),
        },
        "prompt_anchor_xyz": prompt_xyz[-1].astype(np.float32, copy=False),
        "prompt_anchor_valid": bool(prompt_valid_mask[-1]),
        "target_coarse_ids": target_coarse_ids.astype(np.int64, copy=False),
        "target_offset_bins": target_offset_bins.astype(np.int64, copy=False),
        "target_valid_mask": target_valid_mask.astype(bool, copy=False),
        "target_stop": target_stop,
        "target_xyz": target_xyz.astype(np.float32, copy=False),
        "target_bin_center_xyz": target_bin_center_xyz.astype(np.float32, copy=False),
        "target_positions": target_positions,
        "target_length": int(target_len),
        "prompt_length": int(prompt_len),
    }


__all__ = ["IGNORE_INDEX", "serialize_fiber_example"]
