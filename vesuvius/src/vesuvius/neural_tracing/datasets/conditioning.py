import random

import numpy as np


def create_centered_conditioning(dataset, idx: int, patch_idx: int, wrap_idx: int, patch):
    _ = idx
    triplet_meta = dataset._triplet_neighbor_lookup.get((patch_idx, wrap_idx))
    if triplet_meta is None:
        return None

    center_zyxs = dataset._extract_wrap_world_surface_cached(patch_idx, wrap_idx, require_all_valid=True)
    behind_zyxs = dataset._extract_wrap_world_surface_cached(
        patch_idx, triplet_meta["behind_wrap_idx"], require_all_valid=True
    )
    front_zyxs = dataset._extract_wrap_world_surface_cached(
        patch_idx, triplet_meta["front_wrap_idx"], require_all_valid=True
    )
    if center_zyxs is None or behind_zyxs is None or front_zyxs is None:
        return None

    center_zyxs_unperturbed = center_zyxs

    crop_size = dataset.crop_size
    z_min, _, y_min, _, x_min, _ = patch.world_bbox
    min_corner = np.round([z_min, y_min, x_min]).astype(np.int64)
    max_corner = min_corner + np.array(crop_size)
    vol_cache_key = (
        patch_idx,
        int(min_corner[0]), int(min_corner[1]), int(min_corner[2]),
        int(crop_size[0]), int(crop_size[1]), int(crop_size[2]),
    )

    return {
        "center_zyxs_unperturbed": center_zyxs_unperturbed,
        "behind_zyxs": behind_zyxs,
        "front_zyxs": front_zyxs,
        "min_corner": min_corner,
        "max_corner": max_corner,
        "vol_cache_key": vol_cache_key,
    }


def create_split_conditioning(dataset, idx: int, patch_idx: int, wrap_idx: int, patch):
    _ = idx
    crop_size = dataset.crop_size
    # in wrap mode, use the indexed wrap; in chunk mode, choose randomly (legacy behavior)
    wrap = patch.wraps[wrap_idx] if wrap_idx is not None else random.choice(patch.wraps)
    seg = wrap["segment"]
    r_min, r_max, c_min, c_max = wrap["bbox_2d"]

    # clamp bbox to segment bounds (bbox is inclusive in stored resolution)
    seg_h, seg_w = seg._valid_mask.shape
    r_min = max(0, r_min)
    r_max = min(seg_h - 1, r_max)
    c_min = max(0, c_min)
    c_max = min(seg_w - 1, c_max)
    if r_max < r_min or c_max < c_min:
        return None

    seg.use_stored_resolution()
    scale_y, scale_x = seg._scale
    x_full_s, y_full_s, z_full_s, valid_full_s = seg[r_min:r_max + 1, c_min:c_max + 1]

    # if any sample contains an invalid point, just grab a new one
    if not valid_full_s.all():
        return None

    # upsampling here instead of in the tifxyz module because of the annoyances with
    # handling coords in dif scales
    x_full, y_full, z_full = dataset._upsample_world_triplet(x_full_s, y_full_s, z_full_s, scale_y, scale_x)
    trimmed = dataset._trim_to_world_bbox(x_full, y_full, z_full, patch.world_bbox)
    if trimmed is None:
        return None
    x_full, y_full, z_full = trimmed
    h_up, w_up = x_full.shape  # update dimensions after crop

    # split into cond and mask on the upsampled grid
    conditioning_percent = random.uniform(dataset._cond_percent_min, dataset._cond_percent_max)
    if h_up < 2 and w_up < 2:
        return None

    valid_directions = []
    if w_up >= 2:
        valid_directions.extend(["left", "right"])
    if h_up >= 2:
        valid_directions.extend(["up", "down"])
    if not valid_directions:
        return None

    r_cond_up = int(round(h_up * conditioning_percent))
    c_cond_up = int(round(w_up * conditioning_percent))
    if h_up >= 2:
        r_cond_up = min(max(r_cond_up, 1), h_up - 1)
    if w_up >= 2:
        c_cond_up = min(max(c_cond_up, 1), w_up - 1)

    # Split boundaries measured from top/left in the upsampled frame.
    r_split_up_top = r_cond_up
    c_split_up_left = c_cond_up

    cond_direction = random.choice(valid_directions)

    if cond_direction == "left":
        # conditioning is left, mask the right
        x_cond, y_cond, z_cond = x_full[:, :c_split_up_left], y_full[:, :c_split_up_left], z_full[:, :c_split_up_left]
        x_mask, y_mask, z_mask = x_full[:, c_split_up_left:], y_full[:, c_split_up_left:], z_full[:, c_split_up_left:]
        cond_row_off, cond_col_off = 0, 0
        mask_row_off, mask_col_off = 0, c_split_up_left
    elif cond_direction == "right":
        # conditioning is right, mask the left
        c_split_up_left = w_up - c_cond_up
        x_cond, y_cond, z_cond = x_full[:, c_split_up_left:], y_full[:, c_split_up_left:], z_full[:, c_split_up_left:]
        x_mask, y_mask, z_mask = x_full[:, :c_split_up_left], y_full[:, :c_split_up_left], z_full[:, :c_split_up_left]
        cond_row_off, cond_col_off = 0, c_split_up_left
        mask_row_off, mask_col_off = 0, 0
    elif cond_direction == "up":
        # conditioning is up, mask the bottom
        x_cond, y_cond, z_cond = x_full[:r_split_up_top, :], y_full[:r_split_up_top, :], z_full[:r_split_up_top, :]
        x_mask, y_mask, z_mask = x_full[r_split_up_top:, :], y_full[r_split_up_top:, :], z_full[r_split_up_top:, :]
        cond_row_off, cond_col_off = 0, 0
        mask_row_off, mask_col_off = r_split_up_top, 0
    elif cond_direction == "down":
        # conditioning is down, mask the top
        r_split_up_top = h_up - r_cond_up
        x_cond, y_cond, z_cond = x_full[r_split_up_top:, :], y_full[r_split_up_top:, :], z_full[r_split_up_top:, :]
        x_mask, y_mask, z_mask = x_full[:r_split_up_top, :], y_full[:r_split_up_top, :], z_full[:r_split_up_top, :]
        cond_row_off, cond_col_off = r_split_up_top, 0
        mask_row_off, mask_col_off = 0, 0
    else:
        return None

    cond_h, cond_w = x_cond.shape
    mask_h, mask_w = x_mask.shape
    if cond_h == 0 or cond_w == 0 or mask_h == 0 or mask_w == 0:
        return None

    uv_cond = np.stack(np.meshgrid(
        np.arange(cond_h) + cond_row_off,
        np.arange(cond_w) + cond_col_off,
        indexing='ij'
    ), axis=-1)

    uv_mask = np.stack(np.meshgrid(
        np.arange(mask_h) + mask_row_off,
        np.arange(mask_w) + mask_col_off,
        indexing='ij'
    ), axis=-1)

    cond_zyxs = np.stack([z_cond, y_cond, x_cond], axis=-1)
    masked_zyxs = np.stack([z_mask, y_mask, x_mask], axis=-1)
    cond_zyxs_unperturbed = cond_zyxs.copy()

    # use world_bbox directly as crop position, this is the crop returned by find_patches
    z_min, _, y_min, _, x_min, _ = patch.world_bbox
    min_corner = np.round([z_min, y_min, x_min]).astype(np.int64)
    max_corner = min_corner + np.array(crop_size)
    vol_cache_key = (
        patch_idx,
        int(min_corner[0]), int(min_corner[1]), int(min_corner[2]),
        int(crop_size[0]), int(crop_size[1]), int(crop_size[2]),
    )

    return {
        "wrap": wrap,
        "seg": seg,
        "r_min": r_min,
        "r_max": r_max,
        "c_min": c_min,
        "c_max": c_max,
        "conditioning_percent": conditioning_percent,
        "cond_direction": cond_direction,
        "uv_cond": uv_cond,
        "uv_mask": uv_mask,
        "cond_zyxs_unperturbed": cond_zyxs_unperturbed,
        "masked_zyxs": masked_zyxs,
        "min_corner": min_corner,
        "max_corner": max_corner,
        "vol_cache_key": vol_cache_key,
    }
