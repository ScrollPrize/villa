"""Dense cached neural observations for Spiral phase and density losses.

This module deliberately contains no H2 inference code.  An E-step produces a
detached cache along frozen model-phase samples; one or more cheap M-steps map
the same phase samples through the live Spiral transform and call these losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DensePhaseObservation:
    center_zyx: torch.Tensor
    direction_zyx: torch.Tensor
    physical_std_wv: torch.Tensor
    normalized_entropy: torch.Tensor
    peak_probability: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor


def _cached_arclength(points_zyx: torch.Tensor) -> torch.Tensor:
    step = torch.linalg.vector_norm(points_zyx[:, 1:] - points_zyx[:, :-1], dim=-1)
    return F.pad(step.cumsum(dim=1), (1, 0))


def dense_phase_observations(
    cache_points_zyx: torch.Tensor,
    cache_sample_phase: torch.Tensor,
    phase_probability: torch.Tensor,
    modeled_phase: torch.Tensor,
    *,
    sample_valid: torch.Tensor | None = None,
    window_windings: float = 0.75,
    probability_power: float = 16.0,
    maximum_std_wv: float = 12.0,
    minimum_peak_probability: float = 0.0,
) -> DensePhaseObservation:
    """Summarize a local dense posterior without thresholding it into events.

    All returned tensors are detached observations.  ``modeled_phase`` is used
    only to choose a fixed local window and must be the E-step phase coordinate,
    not a live trainable coordinate.
    """
    if cache_points_zyx.ndim != 3 or cache_points_zyx.shape[-1] != 3:
        raise ValueError("cache points must have shape [rays,samples,3]")
    if cache_sample_phase.shape != cache_points_zyx.shape[:2]:
        raise ValueError("cache sample phase must have shape [rays,samples]")
    if phase_probability.shape != cache_sample_phase.shape:
        raise ValueError("phase probability must have shape [rays,samples]")
    if modeled_phase.ndim != 2 or modeled_phase.shape[0] != cache_points_zyx.shape[0]:
        raise ValueError("modeled phase must have shape [rays,windings]")
    if float(window_windings) <= 0 or float(probability_power) <= 0:
        raise ValueError("phase window and probability power must be positive")
    if float(maximum_std_wv) <= 0:
        raise ValueError("maximum phase std must be positive")

    with torch.no_grad():
        points = cache_points_zyx.detach().float()
        sample_phase = cache_sample_phase.detach().float()
        probability = phase_probability.detach().float().clamp(1.0e-8, 1.0)
        target_phase = modeled_phase.detach().float()
        valid_sample = (
            torch.ones_like(sample_phase, dtype=torch.bool)
            if sample_valid is None
            else sample_valid.detach().bool()
        )
        in_window = (
            (sample_phase[:, None, :] - target_phase[:, :, None]).abs()
            <= float(window_windings)
        ) & valid_sample[:, None, :]
        log_weight = float(probability_power) * probability.log()
        log_weight = log_weight[:, None, :].expand_as(in_window).masked_fill(
            ~in_window, float("-inf")
        )
        has_support = in_window.any(dim=-1)
        safe_log_weight = torch.where(
            has_support[:, :, None], log_weight, torch.zeros_like(log_weight)
        )
        posterior = torch.softmax(safe_log_weight, dim=-1)
        posterior = posterior * in_window.to(posterior.dtype)
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

        center = torch.einsum("bks,bsc->bkc", posterior, points)
        arc = _cached_arclength(points)
        mean_arc = torch.einsum("bks,bs->bk", posterior, arc)
        variance = torch.einsum(
            "bks,bks->bk", posterior, (arc[:, None, :] - mean_arc[:, :, None]).square()
        )
        physical_std = variance.clamp_min(0.0).sqrt()

        tangent = torch.empty_like(points)
        tangent[:, 1:-1] = points[:, 2:] - points[:, :-2]
        tangent[:, 0] = points[:, 1] - points[:, 0]
        tangent[:, -1] = points[:, -1] - points[:, -2]
        tangent = F.normalize(tangent, dim=-1)
        direction = F.normalize(torch.einsum("bks,bsc->bkc", posterior, tangent), dim=-1)

        entropy = -(posterior * posterior.clamp_min(1.0e-12).log()).sum(dim=-1)
        support_count = in_window.sum(dim=-1).clamp_min(1).to(entropy.dtype)
        normalized_entropy = entropy / support_count.log().clamp_min(1.0)
        peak = torch.where(
            in_window,
            probability[:, None, :],
            torch.zeros_like(probability[:, None, :]),
        ).amax(dim=-1)
        std_confidence = (1.0 - physical_std / float(maximum_std_wv)).clamp(0.0, 1.0)
        entropy_confidence = (1.0 - normalized_entropy).clamp(0.0, 1.0)
        confidence = std_confidence * entropy_confidence
        valid = (
            has_support
            & torch.isfinite(center).all(dim=-1)
            & torch.isfinite(direction).all(dim=-1)
            & (physical_std <= float(maximum_std_wv))
            & (peak >= float(minimum_peak_probability))
        )
        confidence = confidence * valid.to(confidence.dtype)
    return DensePhaseObservation(
        center_zyx=center,
        direction_zyx=direction,
        physical_std_wv=physical_std,
        normalized_entropy=normalized_entropy,
        peak_probability=peak,
        confidence=confidence,
        valid=valid,
    )


def dense_phase_registration_loss(
    live_target_zyx: torch.Tensor,
    observation: DensePhaseObservation,
    reference_gap_wv: torch.Tensor,
    *,
    huber_delta: float = 0.5,
    minimum_reference_gap_wv: float = 4.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pull live winding targets toward detached dense-posterior centers."""
    if live_target_zyx.shape != observation.center_zyx.shape:
        raise ValueError("live targets and phase centers must have equal [B,K,3] shape")
    if reference_gap_wv.shape != live_target_zyx.shape[:2]:
        raise ValueError("reference gap must have shape [B,K]")
    gap = reference_gap_wv.detach().float().clamp_min(float(minimum_reference_gap_wv))
    projected = (
        (live_target_zyx - observation.center_zyx)
        * observation.direction_zyx
    ).sum(dim=-1)
    residual = projected / gap
    per_target = F.huber_loss(
        residual,
        torch.zeros_like(residual),
        delta=float(huber_delta),
        reduction="none",
    )
    weight = observation.confidence
    denominator = weight.sum().clamp_min(1.0)
    loss = (weight * per_target).sum() / denominator
    metrics = {
        "valid_fraction": observation.valid.float().mean().detach(),
        "confidence_mean": weight.sum().detach() / observation.valid.sum().clamp_min(1),
        "residual_mean_abs": (
            weight * residual.detach().abs()
        ).sum() / denominator.detach(),
        "posterior_std_mean_wv": (
            weight * observation.physical_std_wv
        ).sum() / denominator.detach(),
    }
    return loss, metrics


def dense_metric_density_loss(
    live_points_zyx: torch.Tensor,
    density_windings_per_wv: torch.Tensor,
    target_windings: torch.Tensor,
    *,
    sample_valid: torch.Tensor | None = None,
    minimum_coverage: float = 0.95,
    huber_delta_windings: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Integrate a detached neural winding-rate field over live polylines."""
    if live_points_zyx.ndim != 3 or live_points_zyx.shape[-1] != 3:
        raise ValueError("live points must have shape [rays,samples,3]")
    if density_windings_per_wv.shape != live_points_zyx.shape[:2]:
        raise ValueError("density must have shape [rays,samples]")
    if target_windings.shape != live_points_zyx.shape[:1]:
        raise ValueError("target windings must have shape [rays]")
    if not 0.0 <= float(minimum_coverage) <= 1.0:
        raise ValueError("minimum coverage must be in [0,1]")
    density = density_windings_per_wv.detach().float().clamp_min(0.0)
    valid = (
        torch.ones_like(density, dtype=torch.bool)
        if sample_valid is None
        else sample_valid.detach().bool()
    )
    segment_valid = valid[:, :-1] & valid[:, 1:]
    rate = 0.5 * (density[:, :-1] + density[:, 1:])
    step = torch.linalg.vector_norm(
        live_points_zyx[:, 1:] - live_points_zyx[:, :-1], dim=-1
    )
    integrated = (rate * step * segment_valid.to(step.dtype)).sum(dim=-1)
    coverage = segment_valid.float().mean(dim=-1)
    ray_valid = coverage >= float(minimum_coverage)
    residual = integrated - target_windings.float()
    per_ray = F.huber_loss(
        residual,
        torch.zeros_like(residual),
        delta=float(huber_delta_windings),
        reduction="none",
    )
    weight = ray_valid.to(per_ray.dtype)
    loss = (weight * per_ray).sum() / weight.sum().clamp_min(1.0)
    metrics = {
        "valid_fraction": ray_valid.float().mean().detach(),
        "coverage_mean": coverage.mean().detach(),
        "integrated_mean": (
            integrated.detach() * weight
        ).sum() / weight.sum().clamp_min(1.0),
        "residual_mean_abs": (
            residual.detach().abs() * weight
        ).sum() / weight.sum().clamp_min(1.0),
    }
    return loss, metrics
