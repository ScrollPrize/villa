import torch
import torch.nn.functional as F

def _directional_alignment_loss(
    grid: torch.Tensor,
    mask_sample: torch.Tensor | None = None,
    *,
    model,
    unet_dir0_img: torch.Tensor | None,
    unet_dir1_img: torch.Tensor | None,
    torch_device: torch.device,
) -> torch.Tensor:
    """
    Directional alignment loss between model direction estimates and UNet
    direction branches (dir0 & dir1) in sample space.

    For now we weight all available components equally.
    """
    if unet_dir0_img is None and unet_dir1_img is None:
        return torch.zeros((), device=torch_device, dtype=torch.float32)

    device = torch_device
    dtype = torch.float32

    (
        dir0_v,
        dir0_u_lr,
        dir0_u_rl,
        dir1_v,
        dir1_u_lr,
        dir1_u_rl,
    ) = model._direction_maps(grid, unet_dir0_img, unet_dir1_img)

    def _masked_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a - b
        diff2 = diff * diff
        if mask_sample is None:
            return diff2.mean()
        if mask_sample.shape[-2:] != diff2.shape[-2:]:
            return diff2.mean()
        w = mask_sample
        wsum = w.sum()
        if wsum > 0:
            return (diff2 * w).sum() / wsum
        return diff2.mean()

    # Warp UNet direction channels into sample space using the same grid.
    dir0_unet_hr: torch.Tensor | None = None
    dir1_unet_hr: torch.Tensor | None = None
    if unet_dir0_img is not None:
        dir0_unet_hr = F.grid_sample(
            unet_dir0_img,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
    if unet_dir1_img is not None:
        dir1_unet_hr = F.grid_sample(
            unet_dir1_img,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    losses: list[torch.Tensor] = []

    if dir0_unet_hr is not None and dir0_v is not None and dir0_u_lr is not None and dir0_u_rl is not None:
        losses.append(_masked_mse(dir0_v, dir0_unet_hr))
        losses.append(0.1*_masked_mse(dir0_u_lr, dir0_unet_hr))
        losses.append(0.1*_masked_mse(dir0_u_rl, dir0_unet_hr))

    if dir1_unet_hr is not None and dir1_v is not None and dir1_u_lr is not None and dir1_u_rl is not None:
        losses.append(_masked_mse(dir1_v, dir1_unet_hr))
        losses.append(0.1*_masked_mse(dir1_u_lr, dir1_unet_hr))
        losses.append(0.1*_masked_mse(dir1_u_rl, dir1_unet_hr))

    if len(losses) == 0:
        return torch.zeros((), device=device, dtype=dtype)

    return torch.stack(losses, dim=0).mean()

def _gradient_data_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Gradient matching term between the sampled image data and the *plain*
    cosine target in sample space.

    We penalize differences in forward x/y gradients:

        L = 0.5 * ( ||∂x pred - ∂x target||_2^2 + ||∂y pred - ∂y target||_2^2 )

    If a weight map is provided, we use it (averaged onto the gradient
    positions) as a spatial weighting for both directions.
    """
    # Forward differences along x (width).
    gx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
    diff_gx = gx_pred - gx_tgt

    # Forward differences along y (height).
    gy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
    diff_gy = gy_pred - gy_tgt

    if weight is None:
        loss_x = (diff_gx * diff_gx).mean()
        loss_y = (diff_gy * diff_gy).mean()
        return 0.5 * (loss_x + loss_y)

    # Average weights onto gradient locations.
    wx = torch.minimum(weight[:, :, :, 1:], weight[:, :, :, :-1])
    wy = torch.minimum(weight[:, :, 1:, :], weight[:, :, :-1, :])

    wsum_x = wx.sum()
    wsum_y = wy.sum()

    if wsum_x > 0:
        loss_x = (wx * (diff_gx * diff_gx)).sum() / wsum_x
    else:
        loss_x = (diff_gx * diff_gx).mean()

    if wsum_y > 0:
        loss_y = (wy * (diff_gy * diff_gy)).sum() / wsum_y
    else:
        loss_y = (diff_gy * diff_gy).mean()

    return 0.5 * (loss_x + loss_y)
