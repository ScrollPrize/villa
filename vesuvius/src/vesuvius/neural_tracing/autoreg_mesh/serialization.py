from __future__ import annotations

from typing import Literal

import numpy as np


Direction = Literal["left", "right", "up", "down"]

DIRECTION_TO_ID = {
    "left": 0,
    "right": 1,
    "up": 2,
    "down": 3,
}
ID_TO_DIRECTION = {value: key for key, value in DIRECTION_TO_ID.items()}

IGNORE_INDEX = -100


def _as_3tuple(value) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value), int(value), int(value))
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"expected an int or length-3 sequence, got {value!r}")
    return tuple(int(v) for v in value)


def compute_patch_grid_shape(volume_shape, patch_size) -> tuple[int, int, int]:
    volume_shape = _as_3tuple(volume_shape)
    patch_size = _as_3tuple(patch_size)
    if any(size % patch != 0 for size, patch in zip(volume_shape, patch_size)):
        raise ValueError(f"volume_shape {volume_shape!r} must be divisible by patch_size {patch_size!r}")
    return tuple(int(size // patch) for size, patch in zip(volume_shape, patch_size))


def _broadcast_downsample_factor(surface_downsample_factor) -> tuple[int, int]:
    if isinstance(surface_downsample_factor, int):
        factor = int(surface_downsample_factor)
        return factor, factor
    if not isinstance(surface_downsample_factor, (list, tuple)):
        raise ValueError(f"surface_downsample_factor must be an int or length-2 sequence, got {surface_downsample_factor!r}")
    if len(surface_downsample_factor) != 2:
        raise ValueError(
            f"surface_downsample_factor sequence must have length 2 for row/col factors, got {surface_downsample_factor!r}"
        )
    return int(surface_downsample_factor[0]), int(surface_downsample_factor[1])


def downsample_surface_grid(
    surface_zyxs: np.ndarray,
    *,
    direction: Direction,
    surface_downsample_factor=1,
) -> np.ndarray:
    factor_r, factor_c = _broadcast_downsample_factor(surface_downsample_factor)
    if factor_r <= 1 and factor_c <= 1:
        return np.asarray(surface_zyxs, dtype=np.float32, copy=True)

    surface = np.asarray(surface_zyxs, dtype=np.float32)
    if surface.ndim != 3 or surface.shape[-1] != 3:
        raise ValueError(f"surface_zyxs must have shape [H, W, 3], got {surface.shape!r}")

    h, w = surface.shape[:2]
    row_indices = np.arange(0, h, max(1, factor_r), dtype=np.int64)
    col_indices = np.arange(0, w, max(1, factor_c), dtype=np.int64)
    if row_indices[-1] != h - 1:
        row_indices = np.concatenate([row_indices, np.asarray([h - 1], dtype=np.int64)])
    if col_indices[-1] != w - 1:
        col_indices = np.concatenate([col_indices, np.asarray([w - 1], dtype=np.int64)])

    # Preserve the true split frontier so prompt extraction still references
    # the conditioning edge after downsampling.
    if direction == "left":
        col_indices = np.unique(np.concatenate([col_indices, np.asarray([w - 1], dtype=np.int64)]))
    elif direction == "right":
        col_indices = np.unique(np.concatenate([np.asarray([0], dtype=np.int64), col_indices]))
    elif direction == "up":
        row_indices = np.unique(np.concatenate([row_indices, np.asarray([h - 1], dtype=np.int64)]))
    elif direction == "down":
        row_indices = np.unique(np.concatenate([np.asarray([0], dtype=np.int64), row_indices]))

    return surface[np.ix_(row_indices, col_indices)].astype(np.float32, copy=False)


def extract_frontier_prompt_band(
    cond_zyxs: np.ndarray,
    *,
    direction: Direction,
    frontier_band_width: int = 1,
) -> np.ndarray:
    cond = np.asarray(cond_zyxs, dtype=np.float32)
    if cond.ndim != 3 or cond.shape[-1] != 3:
        raise ValueError(f"cond_zyxs must have shape [H, W, 3], got {cond.shape!r}")
    band = int(frontier_band_width)
    if band <= 0:
        raise ValueError(f"frontier_band_width must be positive, got {frontier_band_width!r}")

    if direction == "left":
        band = min(band, cond.shape[1])
        return cond[:, -band:, :].copy()
    if direction == "right":
        band = min(band, cond.shape[1])
        return cond[:, :band, :].copy()
    if direction == "up":
        band = min(band, cond.shape[0])
        return cond[-band:, :, :].copy()
    if direction == "down":
        band = min(band, cond.shape[0])
        return cond[:band, :, :].copy()
    raise ValueError(f"unsupported direction {direction!r}")


def _iter_strip_indices(size: int, *, direction: Direction, region: Literal["prompt", "target"]) -> list[int]:
    if region == "target":
        if direction in {"left", "up"}:
            return list(range(size))
        return list(range(size - 1, -1, -1))
    if direction in {"left", "up"}:
        return list(range(size - 1, -1, -1))
    return list(range(size))


def _serialize_surface(
    surface_zyxs: np.ndarray,
    *,
    direction: Direction,
    region: Literal["prompt", "target"],
) -> dict:
    surface = np.asarray(surface_zyxs, dtype=np.float32)
    if surface.ndim != 3 or surface.shape[-1] != 3:
        raise ValueError(f"surface_zyxs must have shape [H, W, 3], got {surface.shape!r}")

    h, w = surface.shape[:2]
    seq_xyz: list[np.ndarray] = []
    strip_positions: list[tuple[int, int]] = []
    lattice_rc: list[tuple[int, int]] = []

    if direction in {"left", "right"}:
        strip_indices = _iter_strip_indices(w, direction=direction, region=region)
        for strip_idx, col_idx in enumerate(strip_indices):
            for within_idx in range(h):
                seq_xyz.append(surface[within_idx, col_idx])
                strip_positions.append((strip_idx, within_idx))
                lattice_rc.append((within_idx, col_idx))
        strip_length = h
        num_strips = w
    else:
        strip_indices = _iter_strip_indices(h, direction=direction, region=region)
        for strip_idx, row_idx in enumerate(strip_indices):
            for within_idx in range(w):
                seq_xyz.append(surface[row_idx, within_idx])
                strip_positions.append((strip_idx, within_idx))
                lattice_rc.append((row_idx, within_idx))
        strip_length = w
        num_strips = h

    strip_positions_arr = np.asarray(strip_positions, dtype=np.int64)
    strip_coords = np.zeros((len(strip_positions), 2), dtype=np.float32)
    if len(strip_positions) > 0:
        strip_den = max(num_strips - 1, 1)
        within_den = max(strip_length - 1, 1)
        strip_coords[:, 0] = strip_positions_arr[:, 0].astype(np.float32) / float(strip_den)
        strip_coords[:, 1] = strip_positions_arr[:, 1].astype(np.float32) / float(within_den)
    return {
        "xyz": np.asarray(seq_xyz, dtype=np.float32),
        "strip_positions": strip_positions_arr,
        "strip_coords": strip_coords,
        "lattice_rc": np.asarray(lattice_rc, dtype=np.int64),
        "strip_length": int(strip_length),
        "num_strips": int(num_strips),
        "grid_shape": (int(h), int(w)),
    }


def flatten_coarse_ids(
    cell_indices_zyx: np.ndarray,
    *,
    grid_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    gz, gy, gx = grid_shape_zyx
    cells = np.asarray(cell_indices_zyx, dtype=np.int64)
    return (cells[:, 0] * (gy * gx)) + (cells[:, 1] * gx) + cells[:, 2]


def unflatten_coarse_ids(
    coarse_ids: np.ndarray,
    *,
    grid_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    coarse = np.asarray(coarse_ids, dtype=np.int64)
    gz, gy, gx = grid_shape_zyx
    z = coarse // (gy * gx)
    rem = coarse % (gy * gx)
    y = rem // gx
    x = rem % gx
    return np.stack([z, y, x], axis=-1).astype(np.int64, copy=False)


def quantize_local_xyz(
    xyz: np.ndarray,
    *,
    volume_shape,
    patch_size,
    offset_num_bins,
) -> tuple[np.ndarray, np.ndarray]:
    volume_shape_zyx = _as_3tuple(volume_shape)
    patch_size_zyx = _as_3tuple(patch_size)
    offset_bins_zyx = _as_3tuple(offset_num_bins)
    grid_shape = compute_patch_grid_shape(volume_shape_zyx, patch_size_zyx)

    coords = np.asarray(xyz, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {coords.shape!r}")

    max_coord = np.asarray(volume_shape_zyx, dtype=np.float32) - 1e-4
    coords = np.clip(coords, 0.0, max_coord)

    patch = np.asarray(patch_size_zyx, dtype=np.float32)
    cell_indices = np.floor(coords / patch[None, :]).astype(np.int64)
    for axis, dim in enumerate(grid_shape):
        cell_indices[:, axis] = np.clip(cell_indices[:, axis], 0, dim - 1)

    coarse_ids = flatten_coarse_ids(cell_indices, grid_shape_zyx=grid_shape)

    cell_starts = cell_indices.astype(np.float32) * patch[None, :]
    offsets = np.clip(coords - cell_starts, 0.0, patch[None, :] - 1e-4)
    offset_bins = np.zeros_like(cell_indices, dtype=np.int64)
    for axis, bins in enumerate(offset_bins_zyx):
        bin_width = float(patch_size_zyx[axis]) / float(bins)
        offset_bins[:, axis] = np.floor(offsets[:, axis] / max(bin_width, 1e-6)).astype(np.int64)
        offset_bins[:, axis] = np.clip(offset_bins[:, axis], 0, bins - 1)
    return coarse_ids.astype(np.int64, copy=False), offset_bins.astype(np.int64, copy=False)


def decode_local_xyz(
    coarse_ids: np.ndarray,
    offset_bins: np.ndarray,
    *,
    volume_shape,
    patch_size,
    offset_num_bins,
) -> np.ndarray:
    volume_shape_zyx = _as_3tuple(volume_shape)
    patch_size_zyx = _as_3tuple(patch_size)
    offset_bins_zyx = _as_3tuple(offset_num_bins)
    grid_shape = compute_patch_grid_shape(volume_shape_zyx, patch_size_zyx)

    coarse = np.asarray(coarse_ids, dtype=np.int64)
    offsets = np.asarray(offset_bins, dtype=np.int64)
    if coarse.ndim != 1:
        raise ValueError(f"coarse_ids must have shape [N], got {coarse.shape!r}")
    if offsets.shape != (coarse.shape[0], 3):
        raise ValueError(f"offset_bins must have shape [N, 3], got {offsets.shape!r}")

    cell_indices = unflatten_coarse_ids(coarse, grid_shape_zyx=grid_shape).astype(np.float32)
    cell_starts = cell_indices * np.asarray(patch_size_zyx, dtype=np.float32)[None, :]

    coords = np.zeros((coarse.shape[0], 3), dtype=np.float32)
    for axis, bins in enumerate(offset_bins_zyx):
        bin_width = float(patch_size_zyx[axis]) / float(bins)
        coords[:, axis] = cell_starts[:, axis] + (offsets[:, axis].astype(np.float32) + 0.5) * bin_width
    return np.clip(coords, 0.0, np.asarray(volume_shape_zyx, dtype=np.float32) - 1e-4)


def deserialize_continuation_grid(
    xyz_sequence: np.ndarray,
    *,
    direction: Direction,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    h, w = int(grid_shape[0]), int(grid_shape[1])
    xyz = np.asarray(xyz_sequence, dtype=np.float32)
    expected = h * w
    if xyz.shape != (expected, 3):
        raise ValueError(f"xyz_sequence must have shape [{expected}, 3], got {xyz.shape!r}")

    grid = np.zeros((h, w, 3), dtype=np.float32)
    if direction in {"left", "right"}:
        strip_order = _iter_strip_indices(w, direction=direction, region="target")
        cursor = 0
        for col_idx in strip_order:
            for row_idx in range(h):
                grid[row_idx, col_idx] = xyz[cursor]
                cursor += 1
    else:
        strip_order = _iter_strip_indices(h, direction=direction, region="target")
        cursor = 0
        for row_idx in strip_order:
            for col_idx in range(w):
                grid[row_idx, col_idx] = xyz[cursor]
                cursor += 1
    return grid


def deserialize_full_grid(
    prompt_grid: np.ndarray,
    continuation_grid: np.ndarray,
    *,
    direction: Direction,
) -> np.ndarray:
    prompt = np.asarray(prompt_grid, dtype=np.float32)
    continuation = np.asarray(continuation_grid, dtype=np.float32)
    if prompt.ndim != 3 or continuation.ndim != 3 or prompt.shape[-1] != 3 or continuation.shape[-1] != 3:
        raise ValueError("prompt_grid and continuation_grid must have shape [H, W, 3]")

    if direction == "left":
        if prompt.shape[0] != continuation.shape[0]:
            raise ValueError("prompt and continuation heights must match for left/right directions")
        return np.concatenate([prompt, continuation], axis=1)
    if direction == "right":
        if prompt.shape[0] != continuation.shape[0]:
            raise ValueError("prompt and continuation heights must match for left/right directions")
        return np.concatenate([continuation, prompt], axis=1)
    if direction == "up":
        if prompt.shape[1] != continuation.shape[1]:
            raise ValueError("prompt and continuation widths must match for up/down directions")
        return np.concatenate([prompt, continuation], axis=0)
    if prompt.shape[1] != continuation.shape[1]:
        raise ValueError("prompt and continuation widths must match for up/down directions")
    return np.concatenate([continuation, prompt], axis=0)


def serialize_split_conditioning_example(
    *,
    cond_zyxs_local: np.ndarray,
    masked_zyxs_local: np.ndarray,
    direction: Direction,
    volume_shape,
    patch_size,
    offset_num_bins,
    frontier_band_width: int = 1,
    surface_downsample_factor=1,
    use_stored_resolution_only: bool = False,
) -> dict:
    del use_stored_resolution_only
    cond = np.asarray(cond_zyxs_local, dtype=np.float32)
    masked = np.asarray(masked_zyxs_local, dtype=np.float32)
    if cond.ndim != 3 or masked.ndim != 3 or cond.shape[-1] != 3 or masked.shape[-1] != 3:
        raise ValueError("cond_zyxs_local and masked_zyxs_local must have shape [H, W, 3]")

    cond = downsample_surface_grid(cond, direction=direction, surface_downsample_factor=surface_downsample_factor)
    masked = downsample_surface_grid(masked, direction=direction, surface_downsample_factor=surface_downsample_factor)

    prompt_grid = extract_frontier_prompt_band(cond, direction=direction, frontier_band_width=frontier_band_width)
    prompt_serialized = _serialize_surface(prompt_grid, direction=direction, region="prompt")
    target_serialized = _serialize_surface(masked, direction=direction, region="target")

    prompt_coarse_ids, prompt_offset_bins = quantize_local_xyz(
        prompt_serialized["xyz"],
        volume_shape=volume_shape,
        patch_size=patch_size,
        offset_num_bins=offset_num_bins,
    )
    target_coarse_ids, target_offset_bins = quantize_local_xyz(
        target_serialized["xyz"],
        volume_shape=volume_shape,
        patch_size=patch_size,
        offset_num_bins=offset_num_bins,
    )
    target_stop = np.zeros((target_serialized["xyz"].shape[0],), dtype=np.float32)
    if target_stop.size > 0:
        target_stop[-1] = 1.0

    prompt_anchor_xyz = np.zeros((3,), dtype=np.float32)
    if prompt_serialized["xyz"].shape[0] > 0:
        prompt_anchor_xyz = prompt_serialized["xyz"][0].astype(np.float32, copy=False)

    return {
        "prompt_tokens": {
            "coarse_ids": prompt_coarse_ids,
            "offset_bins": prompt_offset_bins,
            "xyz": prompt_serialized["xyz"],
            "strip_positions": prompt_serialized["strip_positions"],
            "strip_coords": prompt_serialized["strip_coords"],
            "valid_mask": np.ones((prompt_serialized["xyz"].shape[0],), dtype=bool),
        },
        "prompt_meta": {
            "direction": direction,
            "prompt_grid_shape": prompt_serialized["grid_shape"],
            "conditioning_grid_shape": tuple(int(v) for v in cond.shape[:2]),
            "masked_grid_shape": target_serialized["grid_shape"],
            "frontier_band_width": int(frontier_band_width),
            "surface_downsample_factor": surface_downsample_factor,
        },
        "conditioning_grid_local": cond,
        "prompt_grid_local": prompt_grid,
        "prompt_anchor_xyz": prompt_anchor_xyz,
        "target_coarse_ids": target_coarse_ids,
        "target_offset_bins": target_offset_bins,
        "target_stop": target_stop,
        "target_xyz": target_serialized["xyz"],
        "target_strip_positions": target_serialized["strip_positions"],
        "target_strip_coords": target_serialized["strip_coords"],
        "target_grid_local": masked,
        "strip_length": int(target_serialized["strip_length"]),
        "num_strips": int(target_serialized["num_strips"]),
        "target_grid_shape": tuple(int(v) for v in target_serialized["grid_shape"]),
        "direction": direction,
        "direction_id": int(DIRECTION_TO_ID[direction]),
    }
