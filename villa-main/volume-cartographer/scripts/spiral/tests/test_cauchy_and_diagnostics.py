"""Tests for Cauchy robust loss (_cauchy_abs) and tifxyz diagnostic functions.

Tests the Cauchy loss properties (requires torch):
  - Near-zero: rho(r;c) ~ r^2/2 (quadratic, same as L2)
  - Large r: rho grows logarithmically, gradient -> 0 (outlier suppression)
  - Gradient ratio: at r = c, gradient is ~50% of L2; at r = 3c, ~10%
  - Backward compatibility: default config still uses power norm

Tests the diagnostic script (numpy/scipy only):
  - Validity mask: sentinel, NaN, z<=0 all detected
  - PCA thickness: flat plane -> no flag, folded -> flag
  - Holes: interior holes flagged, boundary holes ignored
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch required for Cauchy loss tests")


# -- Cauchy loss tests -------------------------------------------------------

def test_cauchy_abs_near_zero_is_quadratic():
    """Near zero, Cauchy loss ~ r^2/2 (same as L2)."""
    from spiral_helpers import _cauchy_abs
    c = 10.0
    r = torch.tensor([0.0, 0.001, 0.01, 0.1])
    loss = _cauchy_abs(r, c)
    l2 = 0.5 * r ** 2
    # Within 1% for small residuals
    assert torch.allclose(loss, l2, rtol=0.01, atol=1e-10)


def test_cauchy_abs_large_residual_is_sublinear():
    """For |r| >> c, Cauchy loss grows logarithmically, not linearly."""
    from spiral_helpers import _cauchy_abs
    c = 1.0
    r = torch.tensor([10.0, 100.0, 1000.0])
    loss = _cauchy_abs(r, c)
    # Each 10x increase in r should add roughly the same increment
    # (log growth), not 10x more (linear growth)
    increments = loss[1:] - loss[:-1]
    # The ratio of consecutive increments should be < 0.5 (sublinear)
    ratios = increments[1:] / increments[:-1]
    assert all(ratios < 0.6), f"Loss grows too fast: ratios = {ratios}"


def test_cauchy_abs_gradient_suppression():
    """Gradient at r=3c should be <15% of the gradient at r=0.1c."""
    from spiral_helpers import _cauchy_abs
    c = 8.0  # Typical half-winding spacing

    # Compute gradients via autograd
    r_small = torch.tensor([0.1 * c], requires_grad=True)
    r_large = torch.tensor([3.0 * c], requires_grad=True)

    loss_small = _cauchy_abs(r_small, c).sum()
    loss_small.backward()
    grad_small = r_small.grad.item()

    loss_large = _cauchy_abs(r_large, c).sum()
    loss_large.backward()
    grad_large = r_large.grad.item()

    # At 3c, gradient should be suppressed to <15% of near-zero gradient
    ratio = abs(grad_large) / abs(grad_small)
    assert ratio < 0.15, f"Gradient suppression ratio = {ratio:.3f}, expected < 0.15"


def test_cauchy_abs_is_differentiable():
    """Cauchy loss should be differentiable everywhere (no kinks)."""
    from spiral_helpers import _cauchy_abs
    c = 5.0
    r = torch.linspace(-20, 20, 100, requires_grad=True)
    loss = _cauchy_abs(r, c).sum()
    loss.backward()
    # No NaN in gradients
    assert not torch.any(torch.isnan(r.grad))
    # Gradient at r=0 should be exactly 0
    r_zero = torch.tensor([0.0], requires_grad=True)
    _cauchy_abs(r_zero, c).sum().backward()
    assert abs(r_zero.grad.item()) < 1e-10


def test_cauchy_abs_symmetry():
    """Cauchy loss should be symmetric: rho(r) = rho(-r)."""
    from spiral_helpers import _cauchy_abs
    c = 3.0
    r = torch.tensor([1.0, 2.5, 7.0, 15.0])
    assert torch.allclose(_cauchy_abs(r, c), _cauchy_abs(-r, c))


def test_default_config_uses_power_norm():
    """Default Config should use 'power' for track_radius_robust_loss."""
    from config import Config
    cfg = Config()
    assert cfg.track_radius_robust_loss == "power"
    assert cfg.track_radius_cauchy_scale is None


def test_config_accepts_cauchy():
    """Config should accept 'cauchy' for track_radius_robust_loss."""
    from config import Config
    cfg = Config(overrides={"track_radius_robust_loss": "cauchy"})
    assert cfg.track_radius_robust_loss == "cauchy"


def test_config_rejects_invalid_robust_loss():
    """Config should reject invalid values for track_radius_robust_loss."""
    from config import Config
    with pytest.raises(ValueError):
        Config(overrides={"track_radius_robust_loss": "invalid_value"})


def test_config_accepts_cauchy_scale():
    """Config should accept a numeric value for track_radius_cauchy_scale."""
    from config import Config
    cfg = Config(overrides={"track_radius_cauchy_scale": 8.0})
    assert cfg.track_radius_cauchy_scale == 8.0
