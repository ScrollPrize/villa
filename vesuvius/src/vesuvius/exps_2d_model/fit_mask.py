import torch
import torch.nn.functional as F

def _gaussian_mask(stage: int, stage_progress: float) -> torch.Tensor | None:
    """
    Generate a Gaussian mask in image space (1,1,H,W) for the given stage
    and normalized progress value, or None when full-image loss is used
    (no Gaussian).

    When `use_image_mask` is False, this function always returns None so
    that only the cosine-domain sample-space mask is active.

    Args:
        stage:
            1,2: global stages without Gaussian masking (always return None).
            3+: masked stages using the progressive Gaussian schedule.
        stage_progress:
            Float in [0,1] indicating normalized progress within the current
            stage. 0 = stage start, 1 = stage end.
    """
    # Stages 1 and 2 use global optimization without a Gaussian mask.
    # When use_image_mask is False we disable the Gaussian entirely.
    # if (not use_image_mask) or stage < 3:
        # return None

    # Stage 3: grow sigma from sigma_min to sigma_max over the first 90% of
    # the stage, then disable the Gaussian mask (equivalent to full-image
    # loss over the valid region).
    stage_progress = float(max(0.0, min(1.0, stage_progress)))
    if stage_progress >= 0.9:
        return None

    # Map progress in [0,0.9] to [0,1] with quadratic ramp for slower
    # initial growth.
    frac = stage_progress / 0.9
    t = max(0.0, min(1.0, frac * frac))
    sigma = sigma_min + (sigma_max - sigma_min) * t
    if abs(sigma - sigma_min) < 1e-8:
        return gauss_min_img

    return torch.exp(-0.5 * window_r2 / (sigma * sigma))

def _coarse_geom_mask(
    stage: int,
    stage_progress: float,
    cos_v_extent: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build per-vertex coarse-grid masks in [0,1] for geometry losses.

    The image-space mask is obtained by sampling either an all-ones map or
    the current Gaussian loss mask at the mapped coarse-grid coordinates.

    A separate cosine-domain mask is defined in coarse sample space using
    the canonical (u,v) grid. The combined mask is the product of the image
    mask and the cosine-domain mask.

    Returns:
        mask_coarse:      (1,1,gh,gw) image * cosine mask (for most geometry losses)
        mask_img_coarse:  (1,1,gh,gw) image mask only
        mask_cosine_coarse:(1,1,gh,gw) cosine-domain mask only
    """
    with torch.no_grad():
        coords = model.base_grid + model.offset_coarse()  # (1,2,gh,gw)
        u = coords[:, 0:1]
        v = coords[:, 1:2]

        # Map to normalized image coordinates.
        x_norm, y_norm = model._apply_global_transform(u, v)
        grid_coarse = torch.stack(
            [x_norm.squeeze(1), y_norm.squeeze(1)],
            dim=-1,
        )  # (1,gh,gw,2)

        # Image-space loss mask:
        # - stages 1 & 2: all-ones
        # - stage 3: current Gaussian schedule, or all-ones when disabled.
        gauss_img = _gaussian_mask(stage, stage_progress)
        if gauss_img is None:
            mask_img = torch.ones(
                (1, 1, h_img, w_img),
                device=torch_device,
                dtype=torch.float32,
            )
        else:
            mask_img = gauss_img

        mask_img_coarse = F.grid_sample(
            mask_img,
            grid_coarse,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (1,1,gh,gw)

        # Cosine-domain band mask in coarse sample space: same definition as
        # the high-resolution mask, but evaluated on the canonical (u,v) grid.
        base_coords = model.base_grid  # (1,2,gh,gw)
        u_c = base_coords[:, 0:1]      # [-1,1] horizontally
        # v spans [-2,2]; normalize to [-1,1] for the vertical extent test.
        v_c = base_coords[:, 1:2] * 0.5

        # Horizontal cosine-domain mask with linear ramps over two periods,
        # matching the high-resolution definition.
        abs_u_c = u_c.abs()
        u_ramp_c = 2.0 * period_u
        if u_ramp_c > 0.0:
            dist_c = abs_u_c - u_band_half
            mask_x_c = torch.clamp(1.0 - dist_c / u_ramp_c, min=0.0, max=1.0)
        else:
            mask_x_c = (abs_u_c <= u_band_half).float()

        # Clamp vertical extent to [0,1] per step.
        cos_v = float(max(0.0, min(1.0, cos_v_extent)))
        if cos_v >= 1.0:
            mask_y_c = torch.ones_like(v_c, dtype=torch.float32)
        else:
            abs_v_c = v_c.abs()
            dist_v_c = abs_v_c - cos_v
            mask_y_c = torch.clamp(1.0 - dist_v_c / cos_mask_v_ramp_f, min=0.0, max=1.0)
        mask_cosine_coarse = mask_x_c * mask_y_c

        mask_coarse = mask_img_coarse * mask_cosine_coarse

    return mask_coarse, mask_img_coarse, mask_cosine_coarse


def _build_mask_cosine_hr(cos_v_extent: float) -> torch.Tensor:
        """
        Build the current cosine-domain mask in sample space for a given vertical
        half-extent cos_v_extent in normalized v ∈ [0,1].
        """
        cos_v = float(max(0.0, min(1.0, cos_v_extent)))
        if cos_v >= 1.0:
            mask_y = torch.ones_like(v_hr, dtype=torch.float32)
        else:
            abs_v = v_hr.abs()
            dist_v = abs_v - cos_v
            mask_y = torch.clamp(1.0 - dist_v / cos_mask_v_ramp_f, min=0.0, max=1.0)
        return mask_x_hr * mask_y

def _current_cos_v_extent(stage: int, step_stage: int, total_stage_steps: int) -> float:
    """
    Effective vertical half-extent for the cosine-domain mask for a given
    stage and per-stage step index.

    - Stages 1–3: keep the initial extent cos_mask_v_extent_f constant.
    - Stage 4: start from cos_mask_v_extent_f and increase by 0.001 per
        optimization step within stage 4, clamped to 1.0.
    """
    if stage == 4:
        return float(cos_mask_v_extent_f + 0.0001 * float(step_stage))
    return float(cos_mask_v_extent_f)
