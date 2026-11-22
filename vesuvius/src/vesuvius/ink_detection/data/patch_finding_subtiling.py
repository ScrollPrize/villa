"""Filter-empty-tile subtiling patch discovery."""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np

from vesuvius.ink_detection.types import Patch, Segment


def _label_tile_is_empty(label_tile: np.ndarray) -> bool:
    tile = np.asarray(label_tile)
    if tile.size == 0:
        return True
    if np.issubdtype(tile.dtype, np.floating):
        return bool(np.all(tile < 0.01))
    if np.issubdtype(tile.dtype, np.integer):
        return bool(np.all(tile < 3))
    return bool(np.all(tile.astype(np.float32, copy=False) < 0.01))


def _patch_labeled_coverage(label_patch: np.ndarray, *, unsigned_int: bool) -> float:
    if label_patch.size == 0:
        return 0.0
    ph, pw = label_patch.shape
    if unsigned_int:
        row_any = label_patch.max(axis=1) != 0
    else:
        row_any = np.any(label_patch, axis=1)
    if not row_any.any():
        return 0.0
    y_first = int(np.argmax(row_any))
    y_last = ph - 1 - int(np.argmax(row_any[::-1]))
    sub = label_patch[y_first : y_last + 1]
    if unsigned_int:
        col_any = sub.max(axis=0) != 0
    else:
        col_any = np.any(sub, axis=0)
    x_first = int(np.argmax(col_any))
    x_last = pw - 1 - int(np.argmax(col_any[::-1]))
    return float((y_last - y_first + 1) * (x_last - x_first + 1)) / float(label_patch.size)


def _validation_patch_flags(
    validation_plane: np.ndarray,
    xyxys: np.ndarray,
    *,
    size: int,
    stride: int,
) -> np.ndarray:
    """Return a boolean array indicating whether each patch bbox has any nonzero validation pixel."""
    n = xyxys.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.bool_)
    h, w = validation_plane.shape
    val_unsigned = np.issubdtype(validation_plane.dtype, np.unsignedinteger)
    val_bool = validation_plane.dtype == np.bool_
    g = math.gcd(size, stride)
    if g >= 4 and int(xyxys[:, 0].max()) + size <= w and int(xyxys[:, 1].max()) + size <= h:
        # Check alignment to g grid (always true since x1 = x_tile + xi, both multiples of g)
        if not bool(((xyxys[:, 0] % g).any() or (xyxys[:, 1] % g).any())):
            bg_h = h // g
            bg_w = w // g
            y_lim = bg_h * g
            x_lim = bg_w * g
            coarse = np.empty((bg_h, bg_w), dtype=np.bool_)
            strip_blocks = max(1, 512 // g)
            for by0 in range(0, bg_h, strip_blocks):
                by1 = min(bg_h, by0 + strip_blocks)
                nb = by1 - by0
                blk = validation_plane[by0 * g : by1 * g, :x_lim].reshape(nb, g, bg_w, g)
                if val_unsigned:
                    coarse[by0:by1] = blk.max(axis=3).max(axis=1) != 0
                elif val_bool:
                    coarse[by0:by1] = blk.any(axis=3).any(axis=1)
                else:
                    coarse[by0:by1] = (blk != 0).any(axis=3).any(axis=1)
            if not coarse.any():
                return np.zeros(n, dtype=np.bool_)
            sat = np.zeros((bg_h + 1, bg_w + 1), dtype=np.int32)
            np.cumsum(coarse, axis=0, dtype=np.int32, out=sat[1:, 1:])
            np.cumsum(sat[1:, 1:], axis=1, out=sat[1:, 1:])
            bt = size // g
            bx = xyxys[:, 0] // g
            by = xyxys[:, 1] // g
            sums = sat[by + bt, bx + bt] - sat[by, bx + bt] - sat[by + bt, bx] + sat[by, bx]
            return sums > 0

    flags = np.empty(n, dtype=np.bool_)
    for i, (x1, y1, _, _) in enumerate(xyxys.tolist()):
        patch = validation_plane[int(y1) : int(y1) + size, int(x1) : int(x1) + size]
        flags[i] = bool(patch.size and np.any(patch))
    return flags


def build_patch_index(
    mask: np.ndarray,
    fragment_mask: np.ndarray,
    *,
    size: int,
    tile_size: int,
    stride: int,
    filter_empty_tile: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the full-grid index after filtering empty label tiles."""
    size, tile_size, stride = int(size), int(tile_size), int(stride)
    if size <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError("size, tile_size, and stride must be positive")
    if tile_size % size:
        raise ValueError(f"tile_size={tile_size} must be divisible by size={size}")
    if not filter_empty_tile:
        raise ValueError(
            "patch_finding_type='subtiling' does not support "
            "patch_finding_filter_empty_tile=false; set "
            "patch_finding_filter_empty_tile=true"
        )
    label_mask = np.asarray(mask)
    valid_mask = np.asarray(fragment_mask)
    if label_mask.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError(
            f"mask and fragment_mask must be 2D, got {label_mask.shape!r} and {valid_mask.shape!r}"
        )
    if label_mask.shape != valid_mask.shape:
        raise ValueError(
            "mask and fragment_mask must have the same shape, "
            f"got {label_mask.shape!r} and {valid_mask.shape!r}"
        )

    h, w = valid_mask.shape
    max_y = h - tile_size
    max_x = w - tile_size
    if max_y < 0 or max_x < 0:
        empty = np.zeros((0, 4), dtype=np.int64)
        return (
            np.zeros((0, 4), dtype=np.int32),
            empty,
            np.full((0,), -1, dtype=np.int32),
        )

    is_int = np.issubdtype(label_mask.dtype, np.integer)
    is_float = np.issubdtype(label_mask.dtype, np.floating)
    val_is_bool = valid_mask.dtype == np.bool_
    val_is_unsigned = np.issubdtype(valid_mask.dtype, np.unsignedinteger)

    g = math.gcd(tile_size, stride)
    ny_tiles = max_y // stride + 1
    nx_tiles = max_x // stride + 1

    if g >= 4:
        y_limit = (ny_tiles - 1) * stride + tile_size
        x_limit = (nx_tiles - 1) * stride + tile_size
        bg_h = y_limit // g
        bg_w = x_limit // g
        strip_blocks = max(1, 512 // g)
        coarse_ne = np.empty((bg_h, bg_w), dtype=np.bool_)
        coarse_inv = np.empty((bg_h, bg_w), dtype=np.bool_)
        for by0 in range(0, bg_h, strip_blocks):
            by1 = min(bg_h, by0 + strip_blocks)
            nb = by1 - by0
            y0, y1 = by0 * g, by1 * g
            lbl = label_mask[y0:y1, :x_limit].reshape(nb, g, bg_w, g)
            if is_int:
                coarse_ne[by0:by1] = lbl.max(axis=3).max(axis=1) >= 3
            elif is_float:
                # Propagates NaN identically to ~np.all(tile < 0.01): max([0, nan]) is nan and ~(nan < 0.01) is True.
                coarse_ne[by0:by1] = ~(lbl.max(axis=3).max(axis=1) < 0.01)
            else:
                coarse_ne[by0:by1] = ~(lbl.astype(np.float32, copy=False).max(axis=3).max(axis=1) < 0.01)

            val = valid_mask[y0:y1, :x_limit].reshape(nb, g, bg_w, g)
            if val_is_bool:
                coarse_inv[by0:by1] = ~val.all(axis=3).all(axis=1)
            elif val_is_unsigned:
                coarse_inv[by0:by1] = val.min(axis=3).min(axis=1) == 0
            else:
                coarse_inv[by0:by1] = ~(val != 0).all(axis=3).all(axis=1)

        bt = tile_size // g
        bs = stride // g
        if bt == 1:
            accepted = coarse_ne[::bs, ::bs][:ny_tiles, :nx_tiles] & ~coarse_inv[::bs, ::bs][:ny_tiles, :nx_tiles]
        else:
            sat_ne = np.zeros((bg_h + 1, bg_w + 1), dtype=np.int32)
            np.cumsum(coarse_ne, axis=0, dtype=np.int32, out=sat_ne[1:, 1:])
            np.cumsum(sat_ne[1:, 1:], axis=1, out=sat_ne[1:, 1:])
            sat_inv = np.zeros((bg_h + 1, bg_w + 1), dtype=np.int32)
            np.cumsum(coarse_inv, axis=0, dtype=np.int32, out=sat_inv[1:, 1:])
            np.cumsum(sat_inv[1:, 1:], axis=1, out=sat_inv[1:, 1:])
            by = np.arange(0, ny_tiles * bs, bs)
            bx = np.arange(0, nx_tiles * bs, bs)
            ne_sum = (
                sat_ne[by[:, None] + bt, bx[None, :] + bt]
                - sat_ne[by[:, None], bx[None, :] + bt]
                - sat_ne[by[:, None] + bt, bx[None, :]]
                + sat_ne[by[:, None], bx[None, :]]
            )
            inv_sum = (
                sat_inv[by[:, None] + bt, bx[None, :] + bt]
                - sat_inv[by[:, None], bx[None, :] + bt]
                - sat_inv[by[:, None] + bt, bx[None, :]]
                + sat_inv[by[:, None], bx[None, :]]
            )
            accepted = (ne_sum > 0) & (inv_sum == 0)

        ty_idx, tx_idx = np.nonzero(accepted)
        y_tiles = (ty_idx * stride).astype(np.int64)
        x_tiles = (tx_idx * stride).astype(np.int64)
    else:
        valid_bool = valid_mask.astype(bool, copy=False)
        xs = np.arange(0, max_x + 1, stride, dtype=np.int64)
        y_parts: list[np.ndarray] = []
        x_parts: list[np.ndarray] = []
        p_ne = np.empty(w + 1, dtype=np.int32)
        p_inv = np.empty(w + 1, dtype=np.int32)
        p_ne[0] = 0
        p_inv[0] = 0
        for y_tile in range(0, max_y + 1, stride):
            row_lbl = label_mask[y_tile : y_tile + tile_size]
            if is_int:
                col_ne = row_lbl.max(axis=0) >= 3
            elif is_float:
                col_ne = ~(row_lbl.max(axis=0) < 0.01)
            else:
                col_ne = ~(row_lbl.astype(np.float32, copy=False).max(axis=0) < 0.01)
            if not col_ne.any():
                continue
            col_inv = ~valid_bool[y_tile : y_tile + tile_size].all(axis=0)
            np.cumsum(col_ne, dtype=np.int32, out=p_ne[1:])
            np.cumsum(col_inv, dtype=np.int32, out=p_inv[1:])
            ok = ((p_ne[xs + tile_size] - p_ne[xs]) > 0) & ((p_inv[xs + tile_size] - p_inv[xs]) == 0)
            if ok.any():
                x_ok = xs[ok]
                y_parts.append(np.full(x_ok.size, y_tile, dtype=np.int64))
                x_parts.append(x_ok)
        if y_parts:
            y_tiles = np.concatenate(y_parts)
            x_tiles = np.concatenate(x_parts)
        else:
            y_tiles = np.empty(0, dtype=np.int64)
            x_tiles = np.empty(0, dtype=np.int64)

    if y_tiles.size == 0:
        empty = np.zeros((0, 4), dtype=np.int64)
        return (
            np.zeros((0, 4), dtype=np.int32),
            empty,
            np.full((0,), -1, dtype=np.int32),
        )

    if tile_size == size:
        xyxys = np.column_stack((x_tiles, y_tiles, x_tiles + size, y_tiles + size))
    else:
        sub = np.arange(0, tile_size, size, dtype=np.int64)
        k = sub.size
        yy = (y_tiles[:, None, None] + sub[None, :, None] + np.zeros((1, 1, k), dtype=np.int64)).ravel()
        xx = (x_tiles[:, None, None] + np.zeros((1, k, 1), dtype=np.int64) + sub[None, None, :]).ravel()
        # Preserve exact first-seen row-major tile/subtile emission order.
        keys = yy * (w + 1) + xx
        _, first_idx = np.unique(keys, return_index=True)
        first_idx.sort()
        xx = xx[first_idx]
        yy = yy[first_idx]
        xyxys = np.column_stack((xx, yy, xx + size, yy + size))
    return (
        np.zeros((0, 4), dtype=np.int32),
        xyxys,
        np.full((xyxys.shape[0],), -1, dtype=np.int32),
    )


def find_segment_patches(
    segment: Segment, open_volume: Callable[[str, int], object]
) -> tuple[list[Patch], list[Patch]]:
    """Find labeled patches using filter-empty-tile subtiling."""
    if segment.patch_size[1] != segment.patch_size[2]:
        raise ValueError(
            "subtiling patch finding requires square y/x patch_size, "
            f"got {segment.patch_size!r}"
        )
    if segment.supervision_mask is None or segment.inklabels is None:
        raise ValueError("subtiling requires supervision and inklabels")
    supervision = open_volume(segment.supervision_mask, segment.scale)
    inklabels = open_volume(segment.inklabels, segment.scale)
    validation = (
        None
        if segment.validation_mask is None
        else open_volume(segment.validation_mask, segment.scale)
    )
    # Each label store is read once at its own mid-plane (they may hold a different number of planes).
    surface = int(supervision.shape[0] // 2)
    ink_surface = int(inklabels.shape[0] // 2)
    validation_surface = None if validation is None else int(validation.shape[0] // 2)
    patch_size = segment.patch_size
    finding = segment.data_config.patch_finding
    size = patch_size[1]
    tile_size = size if finding.tile_size is None else finding.tile_size
    default_stride = int(size * finding.overlap)
    stride = default_stride if finding.stride is None else finding.stride

    ink_plane = np.asarray(inklabels[ink_surface])
    supervision_plane = np.asarray(supervision[surface])
    validation_plane = None if validation is None else np.asarray(validation[validation_surface])

    _, xyxys, _ = build_patch_index(
        ink_plane,
        supervision_plane,
        size=size,
        tile_size=tile_size,
        stride=stride,
        filter_empty_tile=finding.filter_empty_tile,
    )
    training: list[Patch] = []
    held_out: list[Patch] = []
    # The patch window is in image coordinates; the labels sit at their own mid-plane.
    image_surface = int(open_volume(segment.image_volume, segment.scale).shape[0] // 2)
    z0 = image_surface - patch_size[0] // 2
    z1 = z0 + patch_size[0]
    ph, pw = patch_size[1], patch_size[2]
    min_cov = finding.min_labeled_coverage
    ink_unsigned = np.issubdtype(ink_plane.dtype, np.unsignedinteger)

    val_flags = (
        None
        if validation_plane is None
        else _validation_patch_flags(validation_plane, xyxys, size=size, stride=stride)
    )

    for idx, (x1, y1, _, _) in enumerate(xyxys.tolist()):
        y1 = int(y1)
        x1 = int(x1)
        bbox = (z0, y1, x1, z1, y1 + ph, x1 + pw)
        # build_patch_index only emits patches fully covered by nonzero supervision_plane.
        has_training = True
        has_validation = False if val_flags is None else bool(val_flags[idx])
        if has_validation:
            supervision_patch = supervision_plane[y1 : y1 + ph, x1 : x1 + pw]
            validation_patch = validation_plane[y1 : y1 + ph, x1 : x1 + pw]
            has_training = bool(np.any(supervision_patch & ~validation_patch))
            held_out.append(
                Patch(
                    segment=segment,
                    bbox=bbox,
                    is_validation=True,
                    supervision_mask_override=segment.validation_mask,
                )
            )
        if has_training:
            label_patch = ink_plane[y1 : y1 + ph, x1 : x1 + pw]
            coverage = _patch_labeled_coverage(label_patch, unsigned_int=ink_unsigned)
            if coverage >= min_cov:
                training.append(Patch(segment=segment, bbox=bbox))
    if not training and not held_out:
        raise ValueError(f"{segment.inklabels} produced no valid patches")
    return training, held_out
