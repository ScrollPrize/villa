"""Stable lower-bounded parameterisation for inter-winding gaps."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


GAP_PARAMETERIZATION_VERSION = 2
LEGACY_EXPONENT_SCALE = 200.0
DR_PARAMETER_SCALE = 12.0


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of softplus for strictly positive values."""
    result = torch.empty_like(value)
    large = value > 20.0
    result[large] = value[large]
    result[~large] = torch.log(torch.expm1(value[~large]))
    return result


def calibrated_gap_softplus_scale(
    nominal_dr: float,
    min_gap: float,
    bias: float,
) -> float:
    """Match the legacy exponential's derivative at the identity.

    The transform's effective latent value already includes
    ``model_gap_expander_lr_scale``.  This scale therefore preserves the old
    ``d(gap)/d(effective_latent) = 200 * dr`` at latent zero without retaining
    the exponential's overflow or zero-gap tails.
    """
    if not (0.0 < min_gap < nominal_dr):
        raise ValueError(
            "model_gap_expander_min_gap must be positive and smaller than "
            "model_initial_dr_per_winding")
    softplus_bias = math.log1p(math.exp(-abs(bias))) + max(bias, 0.0)
    sigmoid_bias = 1.0 / (1.0 + math.exp(-bias))
    return (
        LEGACY_EXPONENT_SCALE
        * nominal_dr / (nominal_dr - min_gap)
        * softplus_bias / sigmoid_bias
    )


def lower_bounded_gap(
    effective_latent: torch.Tensor,
    dr_per_winding: torch.Tensor,
    min_gap: float,
    bias: float,
    softplus_scale: float,
) -> torch.Tensor:
    """Map an unconstrained latent value to ``[min_gap, +inf)``.

    A zero latent maps exactly to ``dr_per_winding``.  ``F.softplus`` is
    stable for large positive inputs, so the positive tail grows linearly
    instead of exponentially.
    """
    bias_tensor = effective_latent.new_tensor(bias)
    denominator = F.softplus(bias_tensor)
    ratio = F.softplus(
        bias_tensor + softplus_scale * effective_latent) / denominator
    return min_gap + (dr_per_winding - min_gap) * ratio


def lower_bounded_dr(raw_logit: torch.Tensor, min_gap: float) -> torch.Tensor:
    """Keep the global nominal winding spacing above the local gap floor."""
    return min_gap + F.softplus(raw_logit * DR_PARAMETER_SCALE)


def initial_dr_logit(initial_dr: float, min_gap: float) -> torch.Tensor:
    """Return a raw scalar whose lower-bounded spacing is ``initial_dr``."""
    residual = torch.tensor(initial_dr - min_gap, dtype=torch.float32)
    if float(residual) <= 0.0:
        raise ValueError(
            "model_initial_dr_per_winding must exceed "
            "model_gap_expander_min_gap")
    return inverse_softplus(residual) / DR_PARAMETER_SCALE
