"""Visualization utilities for neural tracing training."""

import math
import torch
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt


def print_training_config(config, accelerator):
    """Print training configuration summary."""
    accelerator.print("\n=== Training Configuration ===")
    accelerator.print(f"Optimizer: {config.get('optimizer', 'adamw')}")
    accelerator.print(f"Scheduler: {config.get('scheduler', 'diffusers_cosine_warmup')}")
    accelerator.print(f"Initial LR: {config.get('optimizer_kwargs', {}).get('learning_rate', 1e-3)}")
    accelerator.print(f"Weight Decay: {config.get('optimizer_kwargs', {}).get('weight_decay', 1e-4)}")
    accelerator.print(f"Grad Clip: {config.get('grad_clip', 5)}")
    accelerator.print(f"Deep Supervision: {config.get('enable_deep_supervision', False)}")
    accelerator.print(f"Binary: {config.get('binary', False)}")
    accelerator.print("")
    accelerator.print("Point Perturbation:")
    pp = config.get('point_perturbation', {})
    if pp:
        accelerator.print(f"  perturb_probability: {pp.get('perturb_probability', 'not set')}")
        accelerator.print(f"  uv_max_perturbation: {pp.get('uv_max_perturbation', 'not set')}")
        accelerator.print(f"  w_max_perturbation: {pp.get('w_max_perturbation', 'not set')}")
        accelerator.print(f"  main_component_distance_factor: {pp.get('main_component_distance_factor', 'not set')}")
    else:
        accelerator.print("  (not configured)")
    accelerator.print("")
    accelerator.print("Step Settings:")
    accelerator.print(f"  step_size: {config.get('step_size', 'not set')}")
    accelerator.print(f"  step_count: {config.get('step_count', 1)}")
    accelerator.print("==============================\n")


def make_canvas(
    inputs,
    targets,
    target_pred,
    config,
    seg=None,
    seg_pred=None,
    normals=None,
    normals_pred=None,
    normals_mask=None,
    cond_channel_start=2,
    save_path=None,
):
    """
    Create visualization canvas for training/validation images.

    Args:
        inputs: Input tensor [B, C, D, H, W]
        targets: Target tensor [B, C, D, H, W]
        target_pred: Predicted tensor [B, C, D, H, W]
        config: Config dict with visualization settings
        seg: Optional segmentation ground truth
        seg_pred: Optional segmentation prediction
        normals: Optional normals ground truth
        normals_pred: Optional normals prediction
        normals_mask: Optional normals mask
        cond_channel_start: Starting channel index for conditioning visualization
        save_path: Optional path to save the canvas image

    Returns:
        Grid tensor ready for saving with plt.imsave
    """
    config.setdefault('log_image_max_samples', 4)
    config.setdefault('log_image_grid_cols', 2)
    config.setdefault('log_image_ext', 'jpg')
    config.setdefault('log_image_quality', 80)

    log_image_max_samples = config['log_image_max_samples']
    log_image_grid_cols = config['log_image_grid_cols']
    multistep_count = int(config.get('multistep_count', 1))

    sample_count = min(inputs.shape[0], log_image_max_samples)
    inputs = inputs[:sample_count]
    targets = targets[:sample_count]
    target_pred = target_pred[:sample_count]
    if seg is not None:
        seg = seg[:sample_count]
    if seg_pred is not None:
        seg_pred = seg_pred[:sample_count]
    if normals is not None:
        normals = normals[:sample_count]
    if normals_pred is not None:
        normals_pred = normals_pred[:sample_count]
    if normals_mask is not None:
        normals_mask = normals_mask[:sample_count]

    if multistep_count > 1:
        targets = rearrange(targets, 'b (uv s) z y x -> b uv s z y x', uv=2).amax(dim=2)
        target_pred = rearrange(target_pred, 'b (uv s) z y x -> b uv s z y x', uv=2).amax(dim=2)

    colours_by_step = torch.rand([targets.shape[1], 3], device=inputs.device) * 0.7 + 0.2
    colours_by_step = torch.cat([torch.ones([3, 3], device=inputs.device), colours_by_step], dim=0)

    def overlay_crosshair(x):
        x = x.clone()
        red = torch.tensor([0.8, 0, 0], device=x.device)
        x[:, x.shape[1] // 2 - 7 : x.shape[1] // 2 - 1, x.shape[2] // 2, :] = red
        x[:, x.shape[1] // 2 + 2 : x.shape[1] // 2 + 8, x.shape[2] // 2, :] = red
        x[:, x.shape[1] // 2, x.shape[2] // 2 - 7 : x.shape[2] // 2 - 1, :] = red
        x[:, x.shape[1] // 2, x.shape[2] // 2 + 2 : x.shape[2] // 2 + 8, :] = red
        return x

    def inputs_slice(dim):
        return overlay_crosshair(inputs[:, 0].select(dim=dim + 1, index=inputs.shape[(dim + 2)] // 2)[..., None].expand(-1, -1, -1, 3) * 0.5 + 0.5)

    def projections(x):
        cond_end = min(cond_channel_start + 3, inputs.shape[1])
        x = torch.cat([inputs[:, cond_channel_start:cond_end], x], dim=1)
        coloured = x[..., None] * colours_by_step[None, :x.shape[1], None, None, None, :]
        return torch.cat([overlay_crosshair(coloured.amax(dim=(1, dim + 2))) for dim in range(3)], dim=1)

    def seg_overlay(mask, colour, alpha=0.6):
        views = []
        # Accept masks in shape [B, Z, Y, X] or [B, 1, Z, Y, X]
        if mask.ndim == 5:
            mask_vol = mask[:, 0]
        elif mask.ndim == 4:
            mask_vol = mask
        else:
            raise ValueError(f"Unexpected seg mask shape {tuple(mask.shape)}; expected 4D or 5D.")
        volume = inputs[:, 0]
        for dim in range(3):
            vol_slice = volume.select(dim=dim + 1, index=volume.shape[dim + 1] // 2)[..., None].expand(-1, -1, -1, 3) * 0.5 + 0.5
            mask_slice = mask_vol.select(dim=dim + 1, index=mask_vol.shape[dim + 1] // 2)[..., None].clamp(0, 1)
            coloured = vol_slice * (1 - mask_slice * alpha) + colour * (mask_slice * alpha)
            views.append(overlay_crosshair(coloured))
        return torch.cat(views, dim=1)

    def normals_vis(n, alpha=0.6):
        n = torch.tanh(n)
        n = (n + 1) / 2
        slices = []
        for dim in range(3):
            mid_idx = n.shape[dim + 2] // 2
            n_slice = n.select(dim=dim + 2, index=mid_idx)
            n_slice = rearrange(n_slice, 'b c h w -> b h w c')
            vol_slice = inputs[:, 0].select(dim=dim + 1, index=inputs.shape[dim + 2] // 2)[..., None].expand(-1, -1, -1, 3) * 0.5 + 0.5
            blended = vol_slice * (1 - alpha) + n_slice * alpha
            slices.append(overlay_crosshair(blended))
        return torch.cat(slices, dim=1)

    views = [
        torch.cat([inputs_slice(dim) for dim in range(3)], dim=1),
        projections(F.sigmoid(target_pred)),
        projections(targets),
    ]

    if seg is not None:
        views.append(seg_overlay((seg != 0).float(), torch.tensor([0.0, 1.0, 0.0], device=inputs.device)))
        if seg_pred is not None:
            seg_pred_vis = seg_pred
            if isinstance(seg_pred_vis, (list, tuple)):
                seg_pred_vis = seg_pred_vis[0]
            # Show sigmoid probabilities as grayscale slices
            seg_probs = torch.sigmoid(seg_pred_vis)
            if seg_probs.ndim == 5:
                seg_probs = seg_probs[:, 0]
            slices = []
            for dim in range(3):
                prob_slice = seg_probs.select(dim=dim + 1, index=seg_probs.shape[dim + 1] // 2)
                prob_slice = prob_slice[..., None].expand(-1, -1, -1, 3)
                slices.append(overlay_crosshair(prob_slice))
            views.append(torch.cat(slices, dim=1))

    if normals is not None:
        views.append(normals_vis(normals))
        if normals_pred is not None:
            n_pred = normals_pred
            if isinstance(n_pred, (list, tuple)):
                n_pred = n_pred[0]
            views.append(normals_vis(n_pred))

    canvas = torch.stack(views, dim=-1)
    sample_canvases = rearrange(canvas.clip(0, 1), 'b y x rgb v -> b y (v x) rgb').cpu()
    b, h, w, c = sample_canvases.shape
    cols = min(log_image_grid_cols, b)
    rows = math.ceil(b / cols)
    grid = torch.zeros((rows * h, cols * w, c), dtype=sample_canvases.dtype)
    for idx in range(b):
        row, col = divmod(idx, cols)
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = sample_canvases[idx]

    # Save if path provided
    if save_path is not None:
        log_image_ext = config['log_image_ext']
        log_image_quality = config['log_image_quality']
        save_kwargs = {'format': log_image_ext}
        if log_image_ext in ('jpg', 'jpeg'):
            save_kwargs['pil_kwargs'] = {'quality': log_image_quality}
        plt.imsave(save_path, grid, **save_kwargs)

    return grid
