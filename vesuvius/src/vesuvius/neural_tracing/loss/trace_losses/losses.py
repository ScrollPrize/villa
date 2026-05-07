import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.loss.displacement_losses import (
    dense_displacement_loss,
    velocity_streamline_integration_loss,
    weighted_vector_smoothness_loss,
)
from vesuvius.neural_tracing.trainers.loss_config import TraceLossConfig


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
