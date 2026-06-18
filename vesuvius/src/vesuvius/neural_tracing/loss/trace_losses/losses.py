import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.loss.displacement_losses import (
    dense_displacement_loss,
    velocity_streamline_integration_loss,
    weighted_vector_smoothness_loss,
)
from vesuvius.neural_tracing.trainers.loss_config import TraceLossConfig
from vesuvius.neural_tracing.trainers.loss_config import CopyNeighborLossConfig


def _zero_loss_from_output(output):
    for tensor in output.values():
        return tensor.new_zeros(())
    raise ValueError("Model returned no output tensors")


def compute_velocity_dir_loss(velocity_dir_pred, velocity_dir_target, velocity_loss_weight):
    if velocity_dir_target is None or velocity_loss_weight is None:
        raise ValueError("Velocity targets are enabled but missing from the batch")
    pred = F.normalize(velocity_dir_pred.float(), dim=1, eps=1e-6)
    target = F.normalize(velocity_dir_target.float(), dim=1, eps=1e-6)
    dir_diff = 1.0 - (pred * target).sum(dim=1, keepdim=True).clamp(min=-1.0, max=1.0)
    weight = velocity_loss_weight.float()
    return (dir_diff * weight).sum() / weight.sum().clamp(min=1.0)


def compute_surface_attract_loss(surface_attract_pred, surface_attract_target, surface_attract_weight, beta):
    if surface_attract_pred is None or surface_attract_target is None:
        raise ValueError("Surface attraction loss is enabled but surface attraction tensors are missing")
    return dense_displacement_loss(
        surface_attract_pred,
        surface_attract_target,
        sample_weights=surface_attract_weight,
        loss_type='vector_huber',
        beta=beta,
    )


def compute_trace_validity_loss(
    trace_validity_pred,
    trace_validity_target,
    trace_validity_weight,
    pos_weight_value,
):
    if trace_validity_pred is None or trace_validity_target is None or trace_validity_weight is None:
        raise ValueError("Trace validity loss is enabled but validity tensors are missing")
    target = trace_validity_target.float().clamp(min=0.0, max=1.0)
    weight = trace_validity_weight.float()
    pos_weight = torch.tensor(
        max(pos_weight_value, 1e-6),
        device=trace_validity_pred.device,
        dtype=torch.float32,
    )
    diff = F.binary_cross_entropy_with_logits(
        trace_validity_pred.float(),
        target,
        pos_weight=pos_weight,
        reduction='none',
    )
    return (diff * weight).sum() / weight.sum().clamp(min=1.0)


def compute_progress_phi_loss(progress_phi_pred, progress_phi_target, progress_phi_weight):
    if progress_phi_pred is None or progress_phi_target is None or progress_phi_weight is None:
        raise ValueError("Progress phi loss is enabled but progress tensors are missing")
    diff = F.smooth_l1_loss(
        progress_phi_pred.float(),
        progress_phi_target.float(),
        beta=0.05,
        reduction="none",
    )
    weight = progress_phi_weight.float()
    return (diff * weight).sum() / weight.sum().clamp(min=1.0)


def compute_progress_gradient_loss(progress_phi_pred, velocity_dir_target, weight):
    if progress_phi_pred is None or velocity_dir_target is None or weight is None:
        raise ValueError("Progress gradient loss is enabled but tensors are missing")
    phi = progress_phi_pred.float()
    dz = torch.zeros_like(phi)
    dy = torch.zeros_like(phi)
    dx = torch.zeros_like(phi)
    dz[:, :, 1:-1] = 0.5 * (phi[:, :, 2:] - phi[:, :, :-2])
    dz[:, :, 0] = phi[:, :, 1] - phi[:, :, 0]
    dz[:, :, -1] = phi[:, :, -1] - phi[:, :, -2]
    dy[:, :, :, 1:-1] = 0.5 * (phi[:, :, :, 2:] - phi[:, :, :, :-2])
    dy[:, :, :, 0] = phi[:, :, :, 1] - phi[:, :, :, 0]
    dy[:, :, :, -1] = phi[:, :, :, -1] - phi[:, :, :, -2]
    dx[:, :, :, :, 1:-1] = 0.5 * (phi[:, :, :, :, 2:] - phi[:, :, :, :, :-2])
    dx[:, :, :, :, 0] = phi[:, :, :, :, 1] - phi[:, :, :, :, 0]
    dx[:, :, :, :, -1] = phi[:, :, :, :, -1] - phi[:, :, :, :, -2]
    grad = torch.cat([dz, dy, dx], dim=1)
    pred = F.normalize(grad, dim=1, eps=1e-6)
    target = F.normalize(velocity_dir_target.float(), dim=1, eps=1e-6)
    diff = 1.0 - (pred * target).sum(dim=1, keepdim=True).clamp(min=-1.0, max=1.0)
    weight = weight.float()
    return (diff * weight).sum() / weight.sum().clamp(min=1.0)


def compute_stop_loss(stop_pred, stop_target, stop_weight, pos_weight_value):
    if stop_pred is None or stop_target is None or stop_weight is None:
        raise ValueError("Stop loss is enabled but stop tensors are missing")
    pos_weight = torch.tensor(
        max(float(pos_weight_value), 1e-6),
        device=stop_pred.device,
        dtype=torch.float32,
    )
    diff = F.binary_cross_entropy_with_logits(
        stop_pred.float(),
        stop_target.float().clamp(min=0.0, max=1.0),
        pos_weight=pos_weight,
        reduction="none",
    )
    weight = stop_weight.float()
    return (diff * weight).sum() / weight.sum().clamp(min=1.0)


def _sample_zyx(field: torch.Tensor, points_zyx: torch.Tensor) -> torch.Tensor:
    b, _, d, h, w = field.shape
    points = points_zyx.float()
    x = points[..., 2] / max(w - 1, 1) * 2.0 - 1.0
    y = points[..., 1] / max(h - 1, 1) * 2.0 - 1.0
    z = points[..., 0] / max(d - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x, y, z], dim=-1).view(b, -1, 1, 1, 3)
    sampled = F.grid_sample(
        field.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[:, :, :, 0, 0]


def compute_copy_neighbor_endpoint_loss(output, targets, loss_config: CopyNeighborLossConfig):
    velocity = output.get("velocity_dir")
    if velocity is None:
        raise ValueError("Endpoint loss is enabled but velocity_dir output is missing")

    points = targets.endpoint_seed_points.float()
    active = targets.endpoint_seed_mask.float() > 0.5
    b, n, _ = points.shape
    _, _, d, h, w = velocity.shape
    if targets.endpoint_step_count is not None:
        step_counts = targets.endpoint_step_count.to(device=points.device, dtype=torch.long)
        max_steps = int(step_counts.max().detach().item()) if step_counts.numel() else 0
    else:
        step_counts = torch.full((b,), int(loss_config.endpoint_steps), device=points.device, dtype=torch.long)
        max_steps = int(loss_config.endpoint_steps)
    max_steps = max(max_steps, 0)

    valid = active.clone()
    for step in range(max_steps):
        step_active = valid & (step < step_counts[:, None])
        if not bool(step_active.any().detach().item()):
            break
        sampled_velocity = _sample_zyx(velocity, points).transpose(1, 2)
        unit_velocity = F.normalize(sampled_velocity, dim=-1, eps=1e-6)
        next_points = points + float(loss_config.endpoint_step_size) * unit_velocity
        points = torch.where(step_active[..., None], next_points, points)
        in_bounds = (
            torch.isfinite(points).all(dim=-1)
            & (points[..., 0] >= 0.0)
            & (points[..., 0] <= d - 1)
            & (points[..., 1] >= 0.0)
            & (points[..., 1] <= h - 1)
            & (points[..., 2] >= 0.0)
            & (points[..., 2] <= w - 1)
        )
        valid = valid & in_bounds
        if loss_config.endpoint_detach_steps and step + 1 < max_steps:
            points = points.detach()

    endpoint_dist = _sample_zyx(targets.target_edt.float(), points).squeeze(1)
    endpoint_dist = endpoint_dist.clamp(min=0.0, max=float(loss_config.endpoint_max_distance))
    mask = valid.float()
    loss = F.smooth_l1_loss(
        endpoint_dist,
        torch.zeros_like(endpoint_dist),
        beta=float(loss_config.endpoint_huber_beta),
        reduction="none",
    )
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def compute_trace_losses(output, targets, loss_config: TraceLossConfig, *, random_trace_sample: bool):
    total_loss = _zero_loss_from_output(output)
    metrics = {}

    if loss_config.lambda_velocity_dir > 0.0:
        loss = compute_velocity_dir_loss(
            output['velocity_dir'],
            targets.velocity_dir,
            targets.velocity_loss_weight,
        )
        weighted_loss = loss_config.lambda_velocity_dir * loss
        total_loss = total_loss + weighted_loss
        metrics['velocity_dir_loss'] = weighted_loss

    if loss_config.lambda_velocity_smooth > 0.0:
        loss = weighted_vector_smoothness_loss(
            output['velocity_dir'],
            sample_weights=targets.velocity_loss_weight,
            normalize_vectors=loss_config.velocity_smooth_normalize,
        )
        weighted_loss = loss_config.lambda_velocity_smooth * loss
        total_loss = total_loss + weighted_loss
        metrics['velocity_smooth_loss'] = weighted_loss

    if loss_config.lambda_trace_integration > 0.0:
        loss = velocity_streamline_integration_loss(
            output['velocity_dir'],
            targets.velocity_dir,
            targets.velocity_loss_weight,
            steps=loss_config.trace_integration_steps,
            step_size=loss_config.trace_integration_step_size,
            max_points=loss_config.trace_integration_max_points,
            min_weight=loss_config.trace_integration_min_weight,
            detach_steps=loss_config.trace_integration_detach_steps,
            random_sample=random_trace_sample,
        )
        weighted_loss = loss_config.lambda_trace_integration * loss
        total_loss = total_loss + weighted_loss
        metrics['trace_integration_loss'] = weighted_loss

    if loss_config.lambda_surface_attract > 0.0:
        loss = compute_surface_attract_loss(
            output.get('surface_attract'),
            targets.surface_attract,
            targets.surface_attract_weight,
            beta=loss_config.surface_attract_huber_beta,
        )
        weighted_loss = loss_config.lambda_surface_attract * loss
        total_loss = total_loss + weighted_loss
        metrics['surface_attract_loss'] = weighted_loss

    if loss_config.lambda_trace_validity > 0.0:
        loss = compute_trace_validity_loss(
            output.get('trace_validity'),
            targets.trace_validity,
            targets.trace_validity_weight,
            pos_weight_value=loss_config.trace_validity_pos_weight,
        )
        weighted_loss = loss_config.lambda_trace_validity * loss
        total_loss = total_loss + weighted_loss
        metrics['trace_validity_loss'] = weighted_loss

    return total_loss, metrics


def compute_copy_neighbor_losses(output, targets, loss_config: CopyNeighborLossConfig):
    total_loss = _zero_loss_from_output(output)
    metrics = {}

    if loss_config.lambda_velocity_dir > 0.0:
        loss = compute_velocity_dir_loss(
            output["velocity_dir"],
            targets.velocity_dir,
            targets.velocity_loss_weight,
        )
        weighted_loss = loss_config.lambda_velocity_dir * loss
        total_loss = total_loss + weighted_loss
        metrics["copy_neighbor_velocity_dir_loss"] = weighted_loss

    if loss_config.lambda_velocity_smooth > 0.0:
        loss = weighted_vector_smoothness_loss(
            output["velocity_dir"],
            sample_weights=targets.velocity_loss_weight,
            normalize_vectors=loss_config.velocity_smooth_normalize,
        )
        weighted_loss = loss_config.lambda_velocity_smooth * loss
        total_loss = total_loss + weighted_loss
        metrics["copy_neighbor_velocity_smooth_loss"] = weighted_loss

    if loss_config.lambda_progress_phi > 0.0:
        loss = compute_progress_phi_loss(
            output.get("progress_phi"),
            targets.progress_phi,
            targets.progress_phi_weight,
        )
        weighted_loss = loss_config.lambda_progress_phi * loss
        total_loss = total_loss + weighted_loss
        metrics["copy_neighbor_progress_phi_loss"] = weighted_loss

    if loss_config.lambda_progress_gradient > 0.0:
        loss = compute_progress_gradient_loss(
            output.get("progress_phi"),
            targets.velocity_dir,
            targets.velocity_loss_weight,
        )
        weighted_loss = loss_config.lambda_progress_gradient * loss
        total_loss = total_loss + weighted_loss
        metrics["copy_neighbor_progress_gradient_loss"] = weighted_loss

    if loss_config.lambda_surface_attract > 0.0:
        loss = compute_surface_attract_loss(
            output.get("surface_attract"),
            targets.surface_attract,
            targets.surface_attract_weight,
            beta=loss_config.surface_attract_huber_beta,
        )
        weighted_loss = loss_config.lambda_surface_attract * loss
        total_loss = total_loss + weighted_loss
        metrics["copy_neighbor_surface_attract_loss"] = weighted_loss

    if loss_config.lambda_stop > 0.0:
        loss = compute_stop_loss(
            output.get("stop"),
            targets.stop,
            targets.stop_weight,
            pos_weight_value=loss_config.stop_pos_weight,
        )
        weighted_loss = loss_config.lambda_stop * loss
        total_loss = total_loss + weighted_loss
        metrics["copy_neighbor_stop_loss"] = weighted_loss

    if loss_config.lambda_endpoint > 0.0:
        loss = compute_copy_neighbor_endpoint_loss(output, targets, loss_config)
        weighted_loss = loss_config.lambda_endpoint * loss
        total_loss = total_loss + weighted_loss
        metrics["copy_neighbor_endpoint_loss"] = weighted_loss

    return total_loss, metrics
