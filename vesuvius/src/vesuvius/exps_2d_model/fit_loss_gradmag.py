import torch
import torch.nn.functional as F


def _grid_segment_length_x(grid: torch.Tensor, w_img: int, h_img: int) -> torch.Tensor:
    """
    Per-sample distance along the horizontal index direction of the sampling grid.

    We compute segment lengths between neighbors along x (last axis) in image
    pixel space, then assign to each sample the average of its left/right
    neighbor segments (using only the available side at the edges).
    """
    # grid: (1, H, W, 2) in normalized image coordinates.
    x_norm = grid[..., 0].unsqueeze(1)  # (1,1,H,W)
    y_norm = grid[..., 1].unsqueeze(1)

    w_eff = float(max(1, w_img - 1))
    h_eff = float(max(1, h_img - 1))
    x_pix = (x_norm + 1.0) * 0.5 * w_eff
    y_pix = (y_norm + 1.0) * 0.5 * h_eff

    # Segment lengths between neighbors along x index.
    dx = x_pix[:, :, :, 1:] - x_pix[:, :, :, :-1]
    dy = y_pix[:, :, :, 1:] - y_pix[:, :, :, :-1]
    seg = torch.sqrt(dx * dx + dy * dy + 1e-12)  # (1,1,H,W-1)

    dist = torch.zeros_like(x_pix)
    if x_pix.shape[-1] == 1:
        # Degenerate case: single column, assign unit length.
        dist[:] = 1.0
        return dist

    # Interior: average of left/right segments.
    dist[:, :, :, 1:-1] = 0.5 * (seg[:, :, :, 1:] + seg[:, :, :, :-1])
    # Edges: only one neighboring segment.
    dist[:, :, :, 0] = seg[:, :, :, 0]
    dist[:, :, :, -1] = seg[:, :, :, -1]
    return dist

def _gradmag_period_core(
    mag_hr: torch.Tensor,
    dist_x_hr: torch.Tensor,
    img_downscale_factor: float,
    cosine_periods: float,
) -> tuple[torch.Tensor | None, int, int, int, int, int]:
    """
    Shared core for gradient-magnitude period handling with distance weighting.

    Each sample's magnitude is first weighted by the distance it covers along
    the horizontal index direction (dist_x_hr) before summing over periods.

    Args:
        mag_hr:     (1,1,H,W) magnitude sampled in sample space.
        dist_x_hr:  (1,1,H,W) per-sample distance along x (same shape as mag_hr).
        img_downscale_factor: image downscale factor used in fitting.

    Returns:
        sum_period_scaled: (1,1,H,periods) scaled period sums, or None if invalid.
        samples_per:       samples per period along x.
        max_cols:          number of valid columns (periods * samples_per).
        hh, ww:            height & width of mag_hr.
        periods_int:       integer number of periods.
    """
    _, _, hh, ww = mag_hr.shape
    periods_int = max(1, int(round(float(cosine_periods))))
    if ww < periods_int:
        return None, 0, 0, hh, ww, periods_int

    if dist_x_hr.shape != mag_hr.shape:
        raise ValueError(f"dist_x_hr shape {dist_x_hr.shape} != mag_hr shape {mag_hr.shape}")

    samples_per = ww // periods_int
    if samples_per <= 0:
        return None, 0, 0, hh, ww, periods_int

    max_cols = samples_per * periods_int
    mag_use = mag_hr[:, :, :, :max_cols]
    dist_use = dist_x_hr[:, :, :, :max_cols]
    weighted = mag_use * dist_use  # (1,1,hh,max_cols)

    weighted_reshaped = weighted.view(1, 1, hh, periods_int, samples_per)
    sum_period = weighted_reshaped.sum(dim=-1)  # (1,1,hh,periods_int)

    # Account for image downscale: each sample corresponds to 1/s^2 original
    # pixels, so the integral over a period should be ~1 (up to a global scale).
    m = float(img_downscale_factor) if img_downscale_factor is not None else 1.0
    sum_period_scaled = m * sum_period
    return sum_period_scaled, samples_per, max_cols, hh, ww, periods_int

def _gradmag_period_loss(
    grid: torch.Tensor,
    img_downscale_factor: float,
    cosine_periods: float,
    unet_mag_img: torch.Tensor | None,
    torch_device: torch.device,
    *,
    w_img: int,
    h_img: int,
    mask_sample: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Period-sum loss on sampled gradient magnitude in sample space.

    We sample the UNet magnitude channel (gradient magnitude) into sample
    space using the current grid, then, for each vertical row, group the
    x-dimension into cosine periods and enforce that the sum of magnitudes
    from peak to peak (one period) is close to 1.

    If a sample-space mask is provided (1,1,H,W), we reshape it in the same
    way as the magnitude (per row, per period, per-sample) and use the
    *minimum* mask value within each (row, period) group as its weight.
    """
    if unet_mag_img is None:
        return torch.zeros((), device=torch_device, dtype=torch.float32)

    # Sample UNet magnitude into sample space.
    mag_hr = F.grid_sample(
        unet_mag_img,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (1,1,hr,wr)

    # Per-sample distance along the horizontal index direction of the sampling grid.
    # Use the *source image* size for pixel scaling (matches original implementation).
    dist_x_hr = _grid_segment_length_x(grid, w_img=w_img, h_img=h_img)

    # If a sample-space mask is provided with matching spatial size, apply it
    # directly to the magnitude so that samples outside the mask contribute
    # zero to the period sums, independent of later weighting.
    if mask_sample is not None and mask_sample.shape[-2:] == mag_hr.shape[-2:]:
        mag_hr = mag_hr * mask_sample

    sum_period_scaled, samples_per, max_cols, hh, ww, periods_int = _gradmag_period_core(
        mag_hr, dist_x_hr, img_downscale_factor, cosine_periods
    )
    if sum_period_scaled is None:
        return torch.zeros((), device=torch_device, dtype=torch.float32)

    # Base per-(row,period) squared error (already scaled).
    err = (sum_period_scaled - 1.0) * (sum_period_scaled - 1.0)  # (1,1,hh,periods_int)

    if mask_sample is None:
        return err.mean()

    # Mask is already in sample space (same coords as mag_hr), e.g. weight_full.
    if mask_sample.shape[-2:] != (hh, ww):
        # Safety: fall back to unweighted if sizes mismatch.
        return err.mean()

    # Reshape mask exactly like magnitude: (N,1,hh,periods,samples_per).
    mask_use = mask_sample[:, :, :, :max_cols]
    mask_reshaped = mask_use.view(1, 1, hh, periods_int, samples_per)

    # For each (row,period), take the MIN mask over that interval to define
    # its weight.
    w_period, _ = mask_reshaped.min(dim=-1)  # (1,1,hh,periods_int)

    w_sum = w_period.sum()
    if w_sum <= 0:
        return err.mean()

    # Apply mask directly when forming the weighted error.
    return (err * w_period).sum() / w_sum
