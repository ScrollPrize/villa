"""
Displacement field loss functions for neural tracing.

Surface-sampled loss for training models to predict dense displacement fields
from extrapolated surfaces.
"""
import torch
import torch.nn.functional as F


def surface_sampled_loss(pred_field, extrap_coords, gt_displacement, valid_mask,
                         loss_type='vector_l2', beta=5.0):
    """
    Sample predicted displacement field at extrapolated surface coords.

    Args:
        pred_field: (B, 3, D, H, W) predicted dense displacement field
        extrap_coords: (B, N, 3) surface point coords in [0, shape) range (z, y, x)
        gt_displacement: (B, N, 3) ground truth displacement (dz, dy, dx)
        valid_mask: (B, N) binary mask for valid (non-padded) points
        loss_type: Loss formulation:
            - 'vector_l2': Squared Euclidean distance (default)
            - 'vector_huber': Huber loss on Euclidean distance
            - 'component_huber': Independent Huber loss per component (legacy)
        beta: Huber transition point for huber losses (default 5.0 voxels)

    Returns:
        Loss between sampled predictions and ground truth
    """
    B, C, D, H, W = pred_field.shape

    # Normalize coords to [-1, 1] for grid_sample
    coords_normalized = extrap_coords.clone()
    coords_normalized[..., 0] = 2 * coords_normalized[..., 0] / (D - 1) - 1  # z
    coords_normalized[..., 1] = 2 * coords_normalized[..., 1] / (H - 1) - 1  # y
    coords_normalized[..., 2] = 2 * coords_normalized[..., 2] / (W - 1) - 1  # x

    # grid_sample expects (B, N, 1, 1, 3) for 3D, with order (x, y, z)
    grid = coords_normalized[..., [2, 1, 0]].view(B, -1, 1, 1, 3)

    # Sample predicted field at surface locations
    sampled = F.grid_sample(pred_field, grid, mode='bilinear', align_corners=True)
    sampled = sampled.view(B, 3, -1).permute(0, 2, 1)  # (B, N, 3)

    # Compute loss based on loss_type
    error = sampled - gt_displacement  # (B, N, 3)

    if loss_type == 'vector_l2':
        # Squared Euclidean distance
        diff = (error ** 2).sum(dim=-1)  # (B, N)
    elif loss_type == 'vector_huber':
        # Huber loss on Euclidean distance
        dist = (error ** 2).sum(dim=-1).sqrt()  # (B, N)
        diff = F.smooth_l1_loss(dist, torch.zeros_like(dist), beta=beta, reduction='none')
    elif loss_type == 'component_huber':
        # Legacy: per-component Huber, then sum
        diff = F.smooth_l1_loss(sampled, gt_displacement, beta=beta, reduction='none')  # (B, N, 3)
        diff = diff.sum(dim=-1)  # (B, N)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    masked_diff = diff * valid_mask
    loss = masked_diff.sum() / valid_mask.sum().clamp(min=1)

    return loss


def smoothness_loss(pred_field):
    """
    L2 norm on spatial gradients of the deformation field.

    Encourages smooth displacement fields by penalizing large gradients.

    Args:
        pred_field: (B, 3, D, H, W)

    Returns:
        Mean squared gradient magnitude
    """
    # Gradients along each spatial dimension
    dz = pred_field[:, :, 1:, :, :] - pred_field[:, :, :-1, :, :]
    dy = pred_field[:, :, :, 1:, :] - pred_field[:, :, :, :-1, :]
    dx = pred_field[:, :, :, :, 1:] - pred_field[:, :, :, :, :-1]

    loss = (dz ** 2).mean() + (dy ** 2).mean() + (dx ** 2).mean()
    return loss
