import math
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import cv2

import fit_mask
from fit_loss_gradmag import _grid_segment_length_x, _gradmag_period_core

def _to_uint8(arr: "np.ndarray") -> "np.ndarray":
    """
    Convert a float or integer image array to uint8 [0,255].

    - For float arrays: per-image min/max normalization to [0,1] then *255.
    - For integer arrays: pass through if already uint8, otherwise scale by
        the dtype max to fit into [0,255].
    """
    import numpy as np

    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax > vmin:
            norm = (arr - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(arr, dtype="float32")
        return (np.clip(norm, 0.0, 1.0) * 255.0).astype("uint8")

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if info.max > 0:
            norm = arr.astype("float32") / float(info.max)
        else:
            norm = np.zeros_like(arr, dtype="float32")
        return (np.clip(norm, 0.0, 1.0) * 255.0).astype("uint8")

    # Fallback: convert via float path.
    arr_f = arr.astype("float32")
    vmin = float(arr_f.min())
    vmax = float(arr_f.max())
    if vmax > vmin:
        norm = (arr_f - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(arr_f, dtype="float32")
    return (np.clip(norm, 0.0, 1.0) * 255.0).astype("uint8")

def _draw_grid_vis(
    *,
    scale_factor: int = 4,
    mask_coarse: torch.Tensor | None = None,
    h_img: int,
    w_img: int,
    for_video: bool,
    image: torch.Tensor,
    model,
    cosine_periods: float,
) -> "np.ndarray":
    """
    Draw the coarse sample-space grid mapped into (an upscaled) image space.

    We render points at each coarse grid corner and lines between neighbors.
    By default this is drawn on top of an upscaled version of the source
    image; in video mode we draw on a black background instead.
    Vertical lines whose coarse x-position coincides with cosine peaks in
    the ground-truth design are highlighted in a different color.

    If a coarse-grid mask is provided (1,1,gh,gw), edge brightness is scaled
    linearly with the corresponding edge weights:

        w_edge in [0,1] -> brightness = 0.1 + 0.9 * w_edge

    so edges in low-weight regions appear darker while still remaining visible.
    """
    import numpy as np
    import cv2

    with torch.no_grad():
        sf = max(1, int(scale_factor))
        h_vis = int(h_img * sf)
        w_vis = int(w_img * sf)

        if for_video:
            # Plain black background for video overlays.
            bg_color = np.zeros((h_vis, w_vis, 3), dtype="uint8")
        else:
            # Base grayscale background from the source image, but rendered
            # only in the central half in each dimension of the visualization
            # frame. The outer regions remain black so grid points that map
            # outside the image but still inside the frame are visible.
            img_np = image[0, 0].detach().cpu().numpy()
            if img_np.max() > img_np.min():
                img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            else:
                img_norm = img_np
            img_u8 = (img_norm * 255.0).astype("uint8")

            # Create black canvas at full visualization size.
            bg_color = np.zeros((h_vis, w_vis, 3), dtype="uint8")

            # Resize image to occupy the central half in each dimension.
            w_im = max(1, w_vis // 2)
            h_im = max(1, h_vis // 2)
            img_resized = cv2.resize(img_u8, (w_im, h_im), interpolation=cv2.INTER_LINEAR)
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

            # Paste resized image into centered rectangle.
            x0 = (w_vis - w_im) // 2
            y0 = (h_vis - h_im) // 2
            x1 = x0 + w_im
            y1 = y0 + h_im
            bg_color[y0:y1, x0:x1, :] = img_resized

        # Coarse coordinates in sample space (u,v).
        coords = model.base_grid + model.offset_coarse()  # (1,2,gh,gw)
        u = coords[:, 0:1]
        v = coords[:, 1:2]

        # Apply the same global x-only scale-then-rotation as in _build_sampling_grid.
        x_norm, y_norm = model._apply_global_transform(u, v)

        # Map normalized coords so that the *image domain* x_norm,y_norm ∈ [-1,1]
        # occupies only the central half of the visualization frame in each
        # dimension, while points farther out (|x_norm|,|y_norm| > 1) are
        # still drawn towards the outer frame.
        #
        # Mapping:
        #   x_norm ∈ [-1,1] -> x_vis ∈ [0.25, 0.75] (central half)
        # extended to:
        #   x_norm ∈ [-2,2] -> x_vis ∈ [0.0, 1.0] (full frame)
        #
        # This keeps the grid scale consistent with the shrunken image in
        # the center, but still shows grid points that lie outside the image
        # domain within the overall visualization frame.
        x_pix = (0.5 + 0.25 * x_norm) * (w_vis - 1)
        y_pix = (0.5 + 0.25 * y_norm) * (h_vis - 1)

        x_pix = x_pix[0, 0].detach().cpu().numpy().astype(np.float32)
        y_pix = y_pix[0, 0].detach().cpu().numpy().astype(np.float32)

        gh, gw = x_pix.shape

        # Optional coarse mask to modulate edge brightness.
        mask_arr = None
        m_h = None
        m_v = None
        if mask_coarse is not None:
            mask_arr = mask_coarse[0, 0].detach().cpu().numpy().astype(np.float32)
            mask_arr = np.clip(mask_arr, 0.0, 1.0)
            if mask_arr.shape == x_pix.shape:
                if gw > 1:
                    # Horizontal edges: average of endpoint weights so that edges
                    # with at least one in-image vertex still contribute.
                    m_h = 0.5 * (mask_arr[:, :-1] + mask_arr[:, 1:])  # (gh,gw-1)
                if gh > 1:
                    # Vertical edges: average of endpoint weights.
                    m_v = 0.5 * (mask_arr[:-1, :] + mask_arr[1:, :])  # (gh-1,gw)

        def edge_brightness(w_edge: float) -> float:
            """
            Map edge/vertex weight w ∈ [0,1] to brightness in [0.2,1.0].
            """
            return 1.0
            # w = max(0.0, min(1.0, float(w_edge)))
            # return 0.2 + 0.8 * w

        def in_bounds(px: int, py: int) -> bool:
            return 0 <= px < w_vis and 0 <= py < h_vis

        # Draw corner points (small green dots) with brightness from the
        # per-vertex mask when available.
        for iy in range(gh):
            for ix in range(gw):
                px = int(round(float(x_pix[iy, ix])))
                py = int(round(float(y_pix[iy, ix])))
                if not in_bounds(px, py):
                    continue
                if mask_arr is not None:
                    w_v = mask_arr[iy, ix]
                    b_v = edge_brightness(w_v)
                else:
                    b_v = 1.0
                color_v = (0, int(255 * b_v), 0)
                cv2.circle(bg_color, (px, py), 1, color_v, -1)

        # Draw horizontal lines (base color blue, brightness from m_h).
        for iy in range(gh):
            for ix in range(gw - 1):
                x0 = int(round(float(x_pix[iy, ix])))
                y0 = int(round(float(y_pix[iy, ix])))
                x1 = int(round(float(x_pix[iy, ix + 1])))
                y1 = int(round(float(y_pix[iy, ix + 1])))
                if in_bounds(x0, y0) and in_bounds(x1, y1):
                    if m_h is not None:
                        b = edge_brightness(m_h[iy, ix])
                    else:
                        b = 1.0
                    color = (int(0 * b), int(0 * b), int(255 * b))
                    cv2.line(bg_color, (x0, y0), (x1, y1), color, 1)

        # Visualize new horizontal connectivity with line offsets.
        #
        # We work in the same visualization coordinate space (x_pix,y_pix)
        # and use _coarse_x_line_pairs on a 2-channel coordinate field to
        # obtain both directed connections for each horizontal edge:
        #   0: left -> right  (cyan)
        #   1: right -> left  (yellow)
        coords_vis_t = torch.from_numpy(
            np.stack([x_pix, y_pix], axis=0)
        ).unsqueeze(0)  # (1,2,gh,gw)
        src_conn, nbr_conn = model._coarse_x_line_pairs(coords_vis_t)  # (1,2,2,gh,gw-1)
        src_conn_np = src_conn[0].detach().cpu().numpy()  # (2,2,gh,gw-1)
        nbr_conn_np = nbr_conn[0].detach().cpu().numpy()  # (2,2,gh,gw-1)

        for iy in range(gh):
            for ix in range(gw - 1):
                if m_h is not None:
                    b_edge = edge_brightness(m_h[iy, ix])
                else:
                    b_edge = 1.0

                # Left -> right (dir=0): cyan
                x0_lr = int(round(float(src_conn_np[0, 0, iy, ix])))
                y0_lr = int(round(float(src_conn_np[1, 0, iy, ix])))
                x1_lr = int(round(float(nbr_conn_np[0, 0, iy, ix])))
                y1_lr = int(round(float(nbr_conn_np[1, 0, iy, ix])))
                if in_bounds(x0_lr, y0_lr) and in_bounds(x1_lr, y1_lr):
                    color_lr = (int(255 * b_edge), int(255 * b_edge), int(0 * b_edge))
                    cv2.line(bg_color, (x0_lr, y0_lr), (x1_lr, y1_lr), color_lr, 1)

                # Right -> left (dir=1): yellow
                x0_rl = int(round(float(src_conn_np[0, 1, iy, ix])))
                y0_rl = int(round(float(src_conn_np[1, 1, iy, ix])))
                x1_rl = int(round(float(nbr_conn_np[0, 1, iy, ix])))
                y1_rl = int(round(float(nbr_conn_np[1, 1, iy, ix])))
                if in_bounds(x0_rl, y0_rl) and in_bounds(x1_rl, y1_rl):
                    color_rl = (int(0 * b_edge), int(255 * b_edge), int(255 * b_edge))
                    cv2.line(bg_color, (x0_rl, y0_rl), (x1_rl, y1_rl), color_rl, 1)

        # Determine "cos-peak" columns in the coarse canonical grid based on
        # the ground-truth cosine design. The cosine target is defined over
        # sample-space x in [-1,1], with `cosine_periods` periods across the
        # width. Peaks (cos=1) occur at normalized positions
        # x_norm_k = 2 * k / cosine_periods - 1, mapped to coarse u.
        base_u_row = model.base_grid[0, 0, 0].detach().cpu().numpy()  # (gw,)
        peak_cols: set[int] = set()
        periods_int = max(1, int(round(float(cosine_periods))))
        for k in range(periods_int + 1):
            x_norm_k = 2.0 * float(k) / float(periods_int) - 1.0
            ix_peak = int(np.argmin(np.abs(base_u_row - x_norm_k)))
            peak_cols.add(ix_peak)

        # Draw vertical lines: highlight peak columns in red, brightness from m_v.
        for iy in range(gh - 1):
            for ix in range(gw):
                x0 = int(round(float(x_pix[iy, ix])))
                y0 = int(round(float(y_pix[iy, ix])))
                x1 = int(round(float(x_pix[iy + 1, ix])))
                y1 = int(round(float(y_pix[iy + 1, ix])))
                if not (in_bounds(x0, y0) and in_bounds(x1, y1)):
                    continue
                if m_v is not None:
                    b = edge_brightness(m_v[iy, ix])
                else:
                    b = 1.0
                if ix in peak_cols:
                    # Cos-peak line: draw in red and slightly thicker.
                    base_color = (255, 0, 0)
                    thickness = 2
                else:
                    base_color = (0, 0, 255)
                    thickness = 1
                color = (int(base_color[0] * b), int(base_color[1] * b), int(base_color[2] * b))
                cv2.line(bg_color, (x0, y0), (x1, y1), color, thickness)

        return bg_color

def _save_snapshot(
    stage: int,
    step_stage: int,
    total_stage_steps: int,
    global_step_idx: int,
    *,
    snapshot: int | None,
    output_prefix: str | None,
    image: torch.Tensor,
    model,
    modulated_target_fn,
    unet_dir0_img: torch.Tensor | None,
    unet_dir1_img: torch.Tensor | None,
    unet_mag_img: torch.Tensor | None,
    img_downscale_factor: float,
    cosine_periods: float,
    h_img: int,
    w_img: int,
    output_scale: int,
    eff_output_scale: int,
    dbg: bool,
    for_video: bool,
    mask_full: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> None:
    """
    Save a snapshot for a given (stage, step) and global step index.

    The Gaussian mask schedule is driven directly from `stage` and the
    normalized `stage_progress = step_stage / (total_stage_steps-1)`,
    matching the training loop exactly (no reconstruction from the
    global step index).
    """
    if snapshot is None or snapshot <= 0 or output_prefix is None:
        return
    with torch.no_grad():
        # Native-resolution prediction and diff in sample space.
        pred_hr = model(image).clamp(0.0, 1.0)
        target_mod_hr = modulated_target_fn()
        diff_hr = pred_hr - target_mod_hr

        pred_out = pred_hr
        diff_out = diff_hr
        modgt_out = target_mod_hr
        if eff_output_scale is not None and eff_output_scale > 1:
            pred_out = F.interpolate(
                pred_out,
                scale_factor=eff_output_scale,
                mode="bicubic",
                align_corners=True,
            )
            modgt_out = F.interpolate(
                modgt_out,
                scale_factor=eff_output_scale,
                mode="bicubic",
                align_corners=True,
            )

        pred_np = pred_out.cpu().squeeze(0).squeeze(0).numpy()
        diff_np = diff_out.cpu().squeeze(0).squeeze(0).numpy()
        modgt_np = modgt_out.cpu().squeeze(0).squeeze(0).numpy()

        # Warped Gaussian weight mask and direction maps in sample space, for debugging.
        mask_np = None
        valid_np = None
        grid_vis_np = None
        dir_model_np = None
        dir_unet_np = None
        dir1_model_np = None
        dir1_unet_np = None
        gradmag_vis_np = None
        gradmag_raw_np = None
        if dbg:
            grid_dbg = model._build_sampling_grid()

            # Exact mask schedule for this snapshot step using the same
            # normalized progress convention as in training.
            if total_stage_steps > 0:
                stage_progress = float(step_stage) / float(max(total_stage_steps - 1, 1))
            else:
                stage_progress = 0.0
            cos_v_eff_dbg = fit_mask._current_cos_v_extent(stage, step_stage, total_stage_steps)

            if valid_mask is not None:
                valid_np = valid_mask.detach().cpu().squeeze(0).squeeze(0).numpy()
            if mask_full is not None:
                mask_np = mask_full.detach().cpu().squeeze(0).squeeze(0).numpy()

            # Direction maps (model vs UNet) in sample space (dir0 & dir1).
            if unet_dir0_img is not None or unet_dir1_img is not None:
                (
                    dir0_v_hr,
                    _dir0_u_lr_hr,
                    _dir0_u_rl_hr,
                    dir1_v_hr,
                    _dir1_u_lr_hr,
                    _dir1_u_rl_hr,
                ) = model._direction_maps(grid_dbg, unet_dir0_img, unet_dir1_img)

                dir0_unet_hr: torch.Tensor | None = None
                dir1_unet_hr: torch.Tensor | None = None
                if unet_dir0_img is not None:
                    dir0_unet_hr = F.grid_sample(
                        unet_dir0_img,
                        grid_dbg,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )
                if unet_dir1_img is not None:
                    dir1_unet_hr = F.grid_sample(
                        unet_dir1_img,
                        grid_dbg,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )

                dir0_model_hr = dir0_v_hr
                dir1_model_hr = dir1_v_hr

                if dir0_model_hr is not None and dir0_unet_hr is not None:
                    dir0_model_vis = dir0_model_hr
                    dir0_unet_vis = dir0_unet_hr
                    if eff_output_scale is not None and eff_output_scale > 1:
                        dir0_model_vis = F.interpolate(
                            dir0_model_vis,
                            scale_factor=eff_output_scale,
                            mode="bicubic",
                            align_corners=True,
                        )
                        dir0_unet_vis = F.interpolate(
                            dir0_unet_vis,
                            scale_factor=eff_output_scale,
                            mode="bicubic",
                            align_corners=True,
                        )
                    dir_model_np = dir0_model_vis.cpu().squeeze(0).squeeze(0).numpy()
                    dir_unet_np = dir0_unet_vis.cpu().squeeze(0).squeeze(0).numpy()

                if dir1_model_hr is not None and dir1_unet_hr is not None:
                    dir1_model_vis = dir1_model_hr
                    dir1_unet_vis = dir1_unet_hr
                    if eff_output_scale is not None and eff_output_scale > 1:
                        dir1_model_vis = F.interpolate(
                            dir1_model_vis,
                            scale_factor=eff_output_scale,
                            mode="bicubic",
                            align_corners=True,
                        )
                        dir1_unet_vis = F.interpolate(
                            dir1_unet_vis,
                            scale_factor=eff_output_scale,
                            mode="bicubic",
                            align_corners=True,
                        )
                    dir1_model_np = dir1_model_vis.cpu().squeeze(0).squeeze(0).numpy()
                    dir1_unet_np = dir1_unet_vis.cpu().squeeze(0).squeeze(0).numpy()

            # Gradient-magnitude visualizations in sample space:
            # - raw resampled UNet magnitude (no period-averaging),
            # - period-sum map (reusing the same core as the loss).
            if unet_mag_img is not None:
                mag_hr_dbg = F.grid_sample(
                    unet_mag_img,
                    grid_dbg,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )  # (1,1,hr,wr)

                # Per-sample distance along the horizontal index direction for debug grid.
                dist_x_dbg = _grid_segment_length_x(grid_dbg, w_img=w_img, h_img=h_img)

                # Period-sum visualization using the same core as the loss (also gives samples_per).
                (
                    sum_period_scaled_dbg,
                    samples_per_dbg,
                    max_cols_dbg,
                    hh_gm,
                    ww_gm,
                    _,
            ) = _gradmag_period_core(mag_hr_dbg, dist_x_dbg, img_downscale_factor, cosine_periods)

                # Raw magnitude in sample space, scaled by 0.5 * samples_per
                # (the size of the summed sub-dimension) for comparability with
                # the period-sum map, then optionally upscaled for output.
                gradmag_raw = mag_hr_dbg
                if samples_per_dbg > 0:
                    scale_raw = 0.5 * float(samples_per_dbg)
                    gradmag_raw = gradmag_raw * scale_raw
                if eff_output_scale is not None and eff_output_scale > 1:
                    gradmag_raw = F.interpolate(
                        gradmag_raw,
                        scale_factor=eff_output_scale,
                        mode="bicubic",
                        align_corners=True,
                    )
                gradmag_raw_np = gradmag_raw.cpu().squeeze(0).squeeze(0).numpy()

                if sum_period_scaled_dbg is not None and samples_per_dbg > 0 and max_cols_dbg > 0:
                    # Broadcast per-period values back to samples within each period.
                    # sum_broadcast_dbg = sum_period_scaled_dbg.repeat_interleave(
                    #     samples_per_dbg, dim=-1
                    # )  # (1,1,hh,max_cols)
                    # gradmag_vis = mag_hr_dbg.new_zeros(1, 1, hh_gm, ww_gm)
                    # gradmag_vis[:, :, :, :max_cols_dbg] = sum_broadcast_dbg
                    # if eff_output_scale is not None and eff_output_scale > 1:
                    #     gradmag_vis = F.interpolate(
                    #         gradmag_vis,
                    #         scale_factor=eff_output_scale,
                    #         mode="bicubic",
                    #         align_corners=True,
                    #     )
                    # gradmag_vis_np = gradmag_vis.cpu().squeeze(0).squeeze(0).numpy()
                    gradmag_vis_np = sum_period_scaled_dbg

            # Grid visualization: heavily upscaled relative to output_scale,
            # always using the same large size; in video mode the background
            # is black instead of the image. Edge brightness is modulated
            # by the same coarse-grid mask used for geometry losses so that
            # low-weight regions appear darker.
            base_scale = int(output_scale)
            vis_scale = base_scale * 2
            geom_mask_vis, _, _ = fit_mask._coarse_geom_mask(stage, stage_progress, cos_v_eff_dbg)
            grid_vis_np = _draw_grid_vis(
                scale_factor=vis_scale,
                mask_coarse=geom_mask_vis,
                h_img=h_img,
                w_img=w_img,
                for_video=for_video,
                image=image,
                model=model,
                cosine_periods=cosine_periods,
            )

    p = Path(output_prefix)

    # In video mode, save the background image once at step 0 for compositing.
    if for_video and global_step_idx == 0:

        img_np = image[0, 0].detach().cpu().numpy()
        if img_np.max() > img_np.min():
            img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        else:
            img_norm = img_np
        bg_np = _to_uint8(img_norm)
        bg_path = f"{p}_bg.tif"
        tifffile.imwrite(bg_path, bg_np, compression="lzw")

    recon_path = f"{p}_arecon_step{global_step_idx:06d}.tif"
    tifffile.imwrite(recon_path, _to_uint8(pred_np), compression="lzw")
    modgt_path = f"{p}_modgt_step{global_step_idx:06d}.tif"
    tifffile.imwrite(modgt_path, _to_uint8(modgt_np), compression="lzw")
    if dbg:
        if mask_np is not None:
            if for_video:
                mask_u8 = (np.clip(mask_np, 0.0, 1.0) * 255.0).astype("uint8")
                mask_path = f"{p}_mask_step{global_step_idx:06d}.jpg"
                cv2.imwrite(mask_path, mask_u8)
            else:
                mask_path = f"{p}_mask_step{global_step_idx:06d}.tif"
                # tifffile.imwrite(mask_path, _to_uint8(mask_np), compression="lzw")
                cv2.imwrite(mask_path, _to_uint8(mask_np))

        if valid_np is not None:
            if for_video:
                valid_u8 = (np.clip(valid_np, 0.0, 1.0) * 255.0).astype("uint8")
                valid_path = f"{p}_valid_step{global_step_idx:06d}.jpg"
                cv2.imwrite(valid_path, valid_u8)
            else:
                valid_path = f"{p}_valid_step{global_step_idx:06d}.tif"
                cv2.imwrite(valid_path, _to_uint8(valid_np))
        # Save diff as |diff|, with 0 -> black and max |diff| -> white.
        diff_abs = np.abs(diff_np)
        maxv = float(diff_abs.max())
        if maxv > 0.0:
            diff_u8 = (np.clip(diff_abs / maxv, 0.0, 1.0) * 255.0).astype("uint8")
        else:
            diff_u8 = np.zeros_like(diff_abs, dtype="uint8")
        diff_path = f"{p}_diff_step{global_step_idx:06d}.tif"
        tifffile.imwrite(diff_path, diff_u8, compression="lzw")
        if grid_vis_np is not None:
            grid_path = f"{p}_grid_step{global_step_idx:06d}.jpg"
            # tifffile.imwrite(grid_path, grid_vis_np, compression="lzw")
            cv2.imwrite(grid_path, np.flip(grid_vis_np, -1))
        # Save direction maps (model vs UNet) if available.
        if dir_model_np is not None and dir_unet_np is not None:
            # Primary encoding (dir0): keep legacy filenames for compatibility.
            dir_model_path = f"{p}_dir_model_step{global_step_idx:06d}.tif"
            dir_unet_path = f"{p}_dir_unet_step{global_step_idx:06d}.tif"
            tifffile.imwrite(dir_model_path, _to_uint8(dir_model_np), compression="lzw")
            tifffile.imwrite(dir_unet_path, _to_uint8(dir_unet_np), compression="lzw")
        if dir1_model_np is not None and dir1_unet_np is not None:
            # Secondary encoding (dir1) with explicit names.
            dir1_model_path = f"{p}_dir1_model_step{global_step_idx:06d}.tif"
            dir1_unet_path = f"{p}_dir1_unet_step{global_step_idx:06d}.tif"
            tifffile.imwrite(dir1_model_path, _to_uint8(dir1_model_np), compression="lzw")
            tifffile.imwrite(dir1_unet_path, _to_uint8(dir1_unet_np), compression="lzw")
        # Save raw gradient-magnitude (resampled UNet mag) as 8-bit JPG.
        if gradmag_raw_np is not None:
            graw = np.clip(gradmag_raw_np, 0.0, 1.0) * 255.0
            graw_u8 = graw.astype("uint8")
            graw_path = f"{p}_gmag_raw_step{global_step_idx:06d}.jpg"
            cv2.imwrite(graw_path, graw_u8)
        # Save gradient-magnitude period-sum visualization as 8-bit JPG.
        print(type(gradmag_vis_np))
        if gradmag_vis_np is not None:
            # gradmag_vis_np may be either a NumPy array (from the upsampled
            # visualization) or a Torch tensor (e.g. sum_period_scaled_dbg on
            # CUDA for debugging). Normalize such that 0.0 -> 0 and 1.0 -> 127.
            if isinstance(gradmag_vis_np, torch.Tensor):
                gradmag_vis_arr = gradmag_vis_np.detach().cpu().numpy()
            else:
                gradmag_vis_arr = gradmag_vis_np
            # gmag = np.clip(gradmag_vis_arr, 0.0, 1.0) * 127.0
            # gmag_u8 = gmag.astype("uint8")
            # gmag_u8 = np.require(gmag_u8, requirements=['C']).copy()
            gmag_path = f"{p}_gmag_step{global_step_idx:06d}.tif"
            # print("write ", gmag_path, gmag_u8.shape, gmag_u8.strides, gmag_u8.dtype)
            cv2.imwrite(gmag_path, gradmag_vis_arr.squeeze(0).squeeze(0))
