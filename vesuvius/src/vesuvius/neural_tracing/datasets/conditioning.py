import random

import numpy as np


def create_split_conditioning(dataset, idx: int, patch_idx: int, wrap_idx: int, patch):
    _ = idx
    crop_size = dataset.crop_size
    wrap = patch.wraps[wrap_idx]
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

    surface_zyx = dataset._extract_wrap_world_surface(patch, wrap, require_all_valid=True)
    if surface_zyx is None:
        return None

    z_full = surface_zyx[..., 0]
    y_full = surface_zyx[..., 1]
    x_full = surface_zyx[..., 2]
    h_up, w_up = surface_zyx.shape[:2]

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
    elif cond_direction == "right":
        # conditioning is right, mask the left
        c_split_up_left = w_up - c_cond_up
        x_cond, y_cond, z_cond = x_full[:, c_split_up_left:], y_full[:, c_split_up_left:], z_full[:, c_split_up_left:]
        x_mask, y_mask, z_mask = x_full[:, :c_split_up_left], y_full[:, :c_split_up_left], z_full[:, :c_split_up_left]
    elif cond_direction == "up":
        # conditioning is up, mask the bottom
        x_cond, y_cond, z_cond = x_full[:r_split_up_top, :], y_full[:r_split_up_top, :], z_full[:r_split_up_top, :]
        x_mask, y_mask, z_mask = x_full[r_split_up_top:, :], y_full[r_split_up_top:, :], z_full[r_split_up_top:, :]
    elif cond_direction == "down":
        # conditioning is down, mask the top
        r_split_up_top = h_up - r_cond_up
        x_cond, y_cond, z_cond = x_full[r_split_up_top:, :], y_full[r_split_up_top:, :], z_full[r_split_up_top:, :]
        x_mask, y_mask, z_mask = x_full[:r_split_up_top, :], y_full[:r_split_up_top, :], z_full[:r_split_up_top, :]
    else:
        return None

    cond_h, cond_w = x_cond.shape
    mask_h, mask_w = x_mask.shape
    if cond_h == 0 or cond_w == 0 or mask_h == 0 or mask_w == 0:
        return None

    cond_zyxs = np.stack([z_cond, y_cond, x_cond], axis=-1)
    masked_zyxs = np.stack([z_mask, y_mask, x_mask], axis=-1)
    cond_zyxs_unperturbed = cond_zyxs.copy()

    # use world_bbox directly as crop position, this is the crop returned by find_patches
    z_min, _, y_min, _, x_min, _ = patch.world_bbox
    min_corner = np.round([z_min, y_min, x_min]).astype(np.int64)
    max_corner = min_corner + np.array(crop_size)

    return {
        "wrap": wrap,
        "cond_direction": cond_direction,
        "cond_zyxs_unperturbed": cond_zyxs_unperturbed,
        "masked_zyxs": masked_zyxs,
        "min_corner": min_corner,
        "max_corner": max_corner,
    }
