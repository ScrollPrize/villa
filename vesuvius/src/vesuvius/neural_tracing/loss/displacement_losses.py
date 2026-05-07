"""
Displacement field loss functions for neural tracing.

Surface-sampled loss for training models to predict dense displacement fields
from extrapolated surfaces.
"""

import torch
import torch.nn.functional as F


def _safe_vector_norm(x, dim, eps=1e-12):
    """Stable vector norm with finite gradient at zero."""
    return torch.sqrt((x ** 2).sum(dim=dim) + eps)


def _resolve_dense_sample_weights(sample_weights, ref_tensor):
    """Resolve optional dense sample weights to a [B, D, H, W] tensor."""
    if sample_weights is None:
        return torch.ones_like(ref_tensor)
    if sample_weights.ndim == 5:
        effective_mask = sample_weights.squeeze(1)
    elif sample_weights.ndim == 4:
        effective_mask = sample_weights
    else:
        raise ValueError(
            "sample_weights must have shape (B, 1, D, H, W) or (B, D, H, W), "
            f"got {tuple(sample_weights.shape)}"
        )
    return effective_mask.to(dtype=ref_tensor.dtype, device=ref_tensor.device)


def _dense_displacement_branch_magnitudes(error):
    """Return per-branch vector magnitudes as [B, num_branches, D, H, W]."""
    channels = int(error.shape[1])
    if channels % 3 != 0:
        return _safe_vector_norm(error, dim=1).unsqueeze(1)
    num_branches = channels // 3
    branch_error = error.reshape(error.shape[0], num_branches, 3, *error.shape[2:])
    return _safe_vector_norm(branch_error, dim=2)


def _sample_pred_field(pred_field, extrap_coords):
    """Sample predicted field at query coordinates.

    Args:
        pred_field: (B, 3, D, H, W)
        extrap_coords: (B, N, 3) in (z, y, x) voxel coords

    Returns:
        sampled: (B, N, 3)
    """
    B, _, D, H, W = pred_field.shape

    # Normalize coords to [-1, 1] for grid_sample
    coords_normalized = extrap_coords.clone()
    d_denom = max(D - 1, 1)
    h_denom = max(H - 1, 1)
    w_denom = max(W - 1, 1)
    coords_normalized[..., 0] = 2 * coords_normalized[..., 0] / d_denom - 1  # z
    coords_normalized[..., 1] = 2 * coords_normalized[..., 1] / h_denom - 1  # y
    coords_normalized[..., 2] = 2 * coords_normalized[..., 2] / w_denom - 1  # x

    # grid_sample expects (B, N, 1, 1, 3) for 3D, with order (x, y, z)
    grid = coords_normalized[..., [2, 1, 0]].view(B, -1, 1, 1, 3)

    # Sample predicted field at surface locations
    sampled = F.grid_sample(pred_field, grid, mode='bilinear', align_corners=True)
    sampled = sampled.view(B, 3, -1).permute(0, 2, 1)  # (B, N, 3)
    return sampled


def surface_sampled_loss(pred_field, extrap_coords, gt_displacement, valid_mask,
                         loss_type='vector_l2', beta=5.0, sample_weights=None):
    """
    Sample predicted displacement field at extrapolated surface coords.

    Args:
        pred_field: (B, 3, D, H, W) predicted dense displacement field
        extrap_coords: (B, N, 3) surface point coords in [0, shape) range (z, y, x)
        gt_displacement: (B, N, 3) ground truth displacement (dz, dy, dx)
        valid_mask: (B, N) binary mask for valid (non-padded) points
        sample_weights: optional (B, N) per-point loss weights
        loss_type: Loss formulation:
            - 'vector_l2': Squared Euclidean distance (default)
            - 'vector_huber': Huber loss on Euclidean distance
            - 'component_huber': Independent Huber loss per component (legacy)
        beta: Huber transition point for huber losses (default 5.0 voxels)

    Returns:
        Loss between sampled predictions and ground truth
    """
    sampled = _sample_pred_field(pred_field, extrap_coords)

    # Compute loss based on loss_type
    error = sampled - gt_displacement  # (B, N, 3)

    if loss_type == 'vector_l2':
        # Squared Euclidean distance
        diff = (error ** 2).sum(dim=-1)  # (B, N)
    elif loss_type == 'vector_huber':
        # Huber loss on Euclidean distance
        dist = _safe_vector_norm(error, dim=-1)  # (B, N)
        diff = F.smooth_l1_loss(dist, torch.zeros_like(dist), beta=beta, reduction='none')
    elif loss_type == 'component_huber':
        # Legacy: per-component Huber, then sum
        diff = F.smooth_l1_loss(sampled, gt_displacement, beta=beta, reduction='none')  # (B, N, 3)
        diff = diff.sum(dim=-1)  # (B, N)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if sample_weights is None:
        effective_mask = valid_mask
    else:
        effective_mask = valid_mask * sample_weights

    masked_diff = diff * effective_mask
    loss = masked_diff.sum() / effective_mask.sum().clamp(min=1)

    return loss


def surface_sampled_normal_loss(
    pred_field,
    extrap_coords,
    gt_displacement,
    point_normals,
    valid_mask,
    loss_type='normal_huber',
    beta=5.0,
    sample_weights=None,
):
    """Supervise only displacement component along per-point normals.

    Args:
        pred_field: (B, 3, D, H, W) predicted dense displacement field
        extrap_coords: (B, N, 3) sample coords in (z, y, x)
        gt_displacement: (B, N, 3) target displacement vectors
        point_normals: (B, N, 3) per-point unit normals (z, y, x order)
        valid_mask: (B, N) binary mask for valid (non-padded) points
        loss_type: one of {'normal_l2', 'normal_huber', 'normal_l1'}
        beta: huber transition point
        sample_weights: optional (B, N) per-point weights
    """
    sampled = _sample_pred_field(pred_field, extrap_coords)  # (B, N, 3)

    normals = point_normals
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    pred_n = (sampled * normals).sum(dim=-1)  # (B, N)
    gt_n = (gt_displacement * normals).sum(dim=-1)  # (B, N)
    err_n = pred_n - gt_n

    if loss_type == 'normal_l2':
        diff = err_n ** 2
    elif loss_type == 'normal_l1':
        diff = err_n.abs()
    elif loss_type == 'normal_huber':
        diff = F.smooth_l1_loss(err_n, torch.zeros_like(err_n), beta=beta, reduction='none')
    else:
        raise ValueError(f"Unknown normal loss_type: {loss_type}")

    if sample_weights is None:
        effective_mask = valid_mask
    else:
        effective_mask = valid_mask * sample_weights

    masked_diff = diff * effective_mask
    loss = masked_diff.sum() / effective_mask.sum().clamp(min=1)
    return loss


def dense_displacement_loss(pred_field, gt_displacement, sample_weights=None,
                            loss_type='vector_l2', beta=5.0):
    """Voxelwise displacement supervision for dense targets.

    Args:
        pred_field: (B, C, D, H, W) predicted dense displacement field
        gt_displacement: (B, C, D, H, W) dense GT displacement vectors
        sample_weights: optional (B, 1, D, H, W) or (B, D, H, W) per-voxel weights
        loss_type: one of {
            'vector_l2',
            'vector_huber',
            'vector_huber_per_branch',
            'component_huber',
        }
        beta: Huber transition point
    """
    if pred_field.shape != gt_displacement.shape:
        raise ValueError(
            f"pred_field and gt_displacement must match shape, got "
            f"{tuple(pred_field.shape)} vs {tuple(gt_displacement.shape)}"
        )

    error = pred_field - gt_displacement

    if loss_type == 'vector_l2':
        diff = (error ** 2).sum(dim=1)  # [B, D, H, W]
    elif loss_type == 'vector_huber':
        dist = _safe_vector_norm(error, dim=1)  # [B, D, H, W]
        diff = F.smooth_l1_loss(dist, torch.zeros_like(dist), beta=beta, reduction='none')
    elif loss_type == 'vector_huber_per_branch':
        channels = int(error.shape[1])
        if channels % 3 != 0:
            raise ValueError(
                "vector_huber_per_branch requires channel count divisible by 3 "
                f"(dz/dy/dx groups), got C={channels}"
            )
        num_branches = channels // 3
        branch_error = error.reshape(error.shape[0], num_branches, 3, *error.shape[2:])
        branch_dist = _safe_vector_norm(branch_error, dim=2)  # [B, num_branches, D, H, W]
        branch_diff = F.smooth_l1_loss(
            branch_dist,
            torch.zeros_like(branch_dist),
            beta=beta,
            reduction='none',
        )
        diff = branch_diff.mean(dim=1)  # [B, D, H, W]
    elif loss_type == 'component_huber':
        diff = F.smooth_l1_loss(pred_field, gt_displacement, beta=beta, reduction='none').sum(dim=1)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if sample_weights is None:
        return diff.mean()

    effective_mask = _resolve_dense_sample_weights(sample_weights, diff)
    masked_diff = diff * effective_mask
    return masked_diff.sum() / effective_mask.sum().clamp(min=1)


def triplet_min_displacement_loss(
    pred_field,
    cond_mask,
    min_magnitude=1.0,
    loss_type='squared_hinge',
):
    """Penalize triplet displacement magnitudes below a minimum on conditioning voxels.

    Args:
        pred_field: (B, 6+, D, H, W) predicted triplet displacement field
        cond_mask: (B, 1, D, H, W) or (B, D, H, W) conditioning region mask
        min_magnitude: minimum desired displacement magnitude in voxels
        loss_type: one of {'squared_hinge', 'linear_hinge'}
    """
    if pred_field.ndim != 5:
        raise ValueError(f"pred_field must have shape (B, C, D, H, W), got {tuple(pred_field.shape)}")
    if pred_field.shape[1] < 6:
        raise ValueError(
            "triplet_min_displacement_loss requires at least 6 channels "
            f"(back dz/dy/dx + front dz/dy/dx), got {pred_field.shape[1]}"
        )
    if cond_mask.ndim == 5:
        if cond_mask.shape[1] != 1:
            raise ValueError(
                "cond_mask with 5 dims must have shape (B, 1, D, H, W), "
                f"got {tuple(cond_mask.shape)}"
            )
        mask = cond_mask.squeeze(1)
    elif cond_mask.ndim == 4:
        mask = cond_mask
    else:
        raise ValueError(
            "cond_mask must have shape (B, 1, D, H, W) or (B, D, H, W), "
            f"got {tuple(cond_mask.shape)}"
        )

    if min_magnitude < 0:
        raise ValueError(f"min_magnitude must be >= 0, got {min_magnitude}")

    back_mag = _safe_vector_norm(pred_field[:, 0:3], dim=1)   # [B, D, H, W]
    front_mag = _safe_vector_norm(pred_field[:, 3:6], dim=1)  # [B, D, H, W]

    back_deficit = F.relu(float(min_magnitude) - back_mag)
    front_deficit = F.relu(float(min_magnitude) - front_mag)

    if loss_type == 'squared_hinge':
        back_pen = back_deficit ** 2
        front_pen = front_deficit ** 2
    elif loss_type == 'linear_hinge':
        back_pen = back_deficit
        front_pen = front_deficit
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    mask = mask.to(dtype=pred_field.dtype, device=pred_field.device)
    masked_pen = (back_pen + front_pen) * mask
    denom = (2.0 * mask.sum()).clamp(min=1.0)
    return masked_pen.sum() / denom


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


def weighted_vector_smoothness_loss(pred_field, sample_weights=None, normalize_vectors=True):
    """Weighted smoothness loss for dense vector fields.

    Args:
        pred_field: (B, C, D, H, W) dense vector field. C is normally 3.
        sample_weights: optional (B, 1, D, H, W) or (B, D, H, W) weights.
        normalize_vectors: if true, penalize angular variation instead of speed
            variation. This is usually what we want for velocity direction heads.
    """
    if pred_field.ndim != 5:
        raise ValueError(f"pred_field must have shape (B, C, D, H, W), got {tuple(pred_field.shape)}")

    field = pred_field.float()
    if normalize_vectors:
        field = F.normalize(field, dim=1, eps=1e-6)

    if sample_weights is None:
        dz = field[:, :, 1:, :, :] - field[:, :, :-1, :, :]
        dy = field[:, :, :, 1:, :] - field[:, :, :, :-1, :]
        dx = field[:, :, :, :, 1:] - field[:, :, :, :, :-1]
        return (dz ** 2).sum(dim=1).mean() + (dy ** 2).sum(dim=1).mean() + (dx ** 2).sum(dim=1).mean()

    weights = _resolve_dense_sample_weights(sample_weights, field[:, 0]).float()
    dz = field[:, :, 1:, :, :] - field[:, :, :-1, :, :]
    dy = field[:, :, :, 1:, :] - field[:, :, :, :-1, :]
    dx = field[:, :, :, :, 1:] - field[:, :, :, :, :-1]

    wz = torch.minimum(weights[:, 1:, :, :], weights[:, :-1, :, :])
    wy = torch.minimum(weights[:, :, 1:, :], weights[:, :, :-1, :])
    wx = torch.minimum(weights[:, :, :, 1:], weights[:, :, :, :-1])

    loss_z = ((dz ** 2).sum(dim=1) * wz).sum()
    loss_y = ((dy ** 2).sum(dim=1) * wy).sum()
    loss_x = ((dx ** 2).sum(dim=1) * wx).sum()
    denom = (wz.sum() + wy.sum() + wx.sum()).clamp(min=1.0)
    return (loss_z + loss_y + loss_x) / denom


def _coords_to_grid(coords_zyx, spatial_shape):
    d, h, w = (int(v) for v in spatial_shape)
    coords = coords_zyx.clone()
    coords[..., 0] = 2.0 * coords[..., 0] / max(d - 1, 1) - 1.0
    coords[..., 1] = 2.0 * coords[..., 1] / max(h - 1, 1) - 1.0
    coords[..., 2] = 2.0 * coords[..., 2] / max(w - 1, 1) - 1.0
    return coords[..., [2, 1, 0]]


def _sample_field_at_zyx(field, coords_zyx):
    """Sample a dense field at per-batch z/y/x coordinates.

    Args:
        field: (B, C, D, H, W)
        coords_zyx: (B, N, 3)
    Returns:
        (B, N, C)
    """
    if field.ndim != 5:
        raise ValueError(f"field must have shape (B, C, D, H, W), got {tuple(field.shape)}")
    if coords_zyx.ndim != 3 or coords_zyx.shape[-1] != 3:
        raise ValueError(f"coords_zyx must have shape (B, N, 3), got {tuple(coords_zyx.shape)}")
    if coords_zyx.shape[0] != field.shape[0]:
        raise ValueError(
            f"batch size mismatch between field and coords: {field.shape[0]} vs {coords_zyx.shape[0]}"
        )

    grid = _coords_to_grid(coords_zyx, field.shape[2:]).view(field.shape[0], -1, 1, 1, 3)
    sampled = F.grid_sample(field.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return sampled.view(field.shape[0], field.shape[1], -1).permute(0, 2, 1)


def velocity_streamline_integration_loss(
    velocity_dir_pred,
    velocity_dir_target,
    velocity_loss_weight,
    *,
    steps=2,
    step_size=1.0,
    max_points=2048,
    min_weight=0.5,
    detach_steps=False,
    random_sample=True,
):
    """Short-horizon streamline consistency loss for velocity direction fields.

    This samples supervised voxels, advances them through the predicted velocity
    field for a few Euler steps, and compares the predicted direction along that
    path against the target direction sampled at the same continuous locations.
    It is intended as a cheap local rollout regularizer, not a full tracing
    objective.
    """
    if velocity_dir_pred.ndim != 5 or velocity_dir_pred.shape[1] != 3:
        raise ValueError(
            "velocity_dir_pred must have shape (B, 3, D, H, W), "
            f"got {tuple(velocity_dir_pred.shape)}"
        )
    if velocity_dir_target.shape != velocity_dir_pred.shape:
        raise ValueError(
            "velocity_dir_target must match velocity_dir_pred shape, got "
            f"{tuple(velocity_dir_target.shape)} vs {tuple(velocity_dir_pred.shape)}"
        )
    if velocity_loss_weight.ndim != 5 or velocity_loss_weight.shape[1] != 1:
        raise ValueError(
            "velocity_loss_weight must have shape (B, 1, D, H, W), "
            f"got {tuple(velocity_loss_weight.shape)}"
        )

    steps = int(steps)
    max_points = int(max_points)
    if steps <= 0 or max_points <= 0 or float(step_size) <= 0.0:
        return velocity_dir_pred.new_zeros(())

    bsz, _, depth, height, width = velocity_dir_pred.shape
    losses = []
    weights_out = []
    weight = velocity_loss_weight.float()

    for b in range(bsz):
        valid = weight[b, 0] > float(min_weight)
        coords = torch.nonzero(valid, as_tuple=False)
        if coords.numel() == 0:
            continue
        if coords.shape[0] > max_points:
            if random_sample:
                selected = torch.randperm(coords.shape[0], device=coords.device)[:max_points]
            else:
                selected = torch.linspace(
                    0,
                    coords.shape[0] - 1,
                    steps=max_points,
                    device=coords.device,
                    dtype=torch.float32,
                ).round().long()
            coords = coords[selected]
        coords = coords.to(dtype=torch.float32).unsqueeze(0)

        pred_b = velocity_dir_pred[b:b + 1]
        target_b = velocity_dir_target[b:b + 1]
        weight_b = weight[b:b + 1]

        for _ in range(steps):
            pred_vec = F.normalize(_sample_field_at_zyx(pred_b, coords), dim=-1, eps=1e-6)
            step_vec = pred_vec.detach() if detach_steps else pred_vec
            coords = coords + float(step_size) * step_vec

            in_bounds = (
                (coords[..., 0] >= 0.0)
                & (coords[..., 0] <= depth - 1)
                & (coords[..., 1] >= 0.0)
                & (coords[..., 1] <= height - 1)
                & (coords[..., 2] >= 0.0)
                & (coords[..., 2] <= width - 1)
            )
            target_w = _sample_field_at_zyx(weight_b, coords).squeeze(-1).clamp(min=0.0, max=1.0)
            target_w = target_w * in_bounds.to(dtype=target_w.dtype)
            if not bool((target_w > 1e-6).any().item()):
                continue

            pred_next = F.normalize(_sample_field_at_zyx(pred_b, coords), dim=-1, eps=1e-6)
            target_next = F.normalize(_sample_field_at_zyx(target_b, coords), dim=-1, eps=1e-6)
            dir_diff = 1.0 - (pred_next * target_next).sum(dim=-1).clamp(min=-1.0, max=1.0)
            losses.append((dir_diff * target_w).sum())
            weights_out.append(target_w.sum())

    if not losses:
        return velocity_dir_pred.new_zeros(())
    return torch.stack(losses).sum() / torch.stack(weights_out).sum().clamp(min=1.0)
