"""The fused gap-expander kernels against the eager path on the same device.

gap_triton runs on CUDA and gap_metal on MPS. Each available backend runs
GapExpandingTransform's forward map (_call) and inverse (_inverse), forward and
backward, and must match the eager path within float32 tolerance: mapped
points, point gradients, gap-logit gradients and the dr_per_winding gradient.
The points cover the border clamps (z outside the logit lattice), the w == 0
clip and radii past the outermost winding (the inverse's never-found case).
"""
import math

import pytest
import torch

import gap_metal
import gap_triton
from transforms import GapExpanderParams, GapExpandingTransform

CAPACITY = 40
DR = 16.0
MIN_Z, MAX_Z = 0.0, 240.0


def _backends():
    backends = []
    if gap_triton._HAS_TRITON and torch.cuda.is_available():
        backends.append(('cuda', 'FIT_SPIRAL_TRITON', gap_triton))
    if torch.backends.mps.is_available():
        backends.append(('mps', 'FIT_SPIRAL_METAL', gap_metal))
    return backends


BACKENDS = _backends()
pytestmark = pytest.mark.skipif(
    not BACKENDS, reason='needs CUDA with Triton, or MPS')


def _inputs(n, logit_std):
    gen = torch.Generator().manual_seed(1234)
    params = GapExpanderParams(
        resolution=24, min_z=MIN_Z, max_z=MAX_Z, num_windings=CAPACITY,
        dr_per_winding=DR)
    logits = torch.randn(params.logits.shape, generator=gen) * logit_std
    theta = torch.rand(n, generator=gen) * (2 * math.pi)
    # windings -1 .. CAPACITY + 1: below the first and past the last
    winding = torch.rand(n, generator=gen) * (CAPACITY + 2) - 1.0
    radius = ((winding + theta / (2 * math.pi)) * DR).clamp(min=0.05)
    z = torch.rand(n, generator=gen) * (MAX_Z - MIN_Z + 80.0) + MIN_Z - 40.0
    points = torch.stack(
        [z, radius * torch.sin(theta), radius * torch.cos(theta)], dim=-1)
    weights = torch.randn(2, n, 3, generator=gen)
    return logits, points, weights


def _run(device, truncate_frac, logit_std=0.01, n=4096):
    logits, points, weights = _inputs(n, logit_std)
    params = GapExpanderParams(
        resolution=24, min_z=MIN_Z, max_z=MAX_Z, num_windings=CAPACITY,
        dr_per_winding=DR).to(device)
    with torch.no_grad():
        params.logits.copy_(logits.to(device))
    dr = torch.tensor(DR, device=device, requires_grad=True)
    transform = GapExpandingTransform(
        params, dr, MIN_Z, MAX_Z, gap_expander_lr_scale=0.3, min_gap=1.0,
        softplus_bias=4.0, truncate_frac=truncate_frac)
    x = points.to(device).requires_grad_(True)
    mapped = transform._call(x)
    restored = transform._inverse(x)
    weights = weights.to(device)
    ((mapped * weights[0]).sum() + (restored * weights[1]).sum()).backward()
    backend = transform._fused_gap(x, dr)
    out = {
        'mapped': mapped, 'restored': restored, 'grad_points': x.grad,
        'grad_logits': params.logits.grad, 'grad_dr': dr.grad.reshape(1),
    }
    return backend, {k: v.detach().cpu() for k, v in out.items()}


# 0.01: the gaps stay within about 0.5-2x the nominal spacing, so some points
# lie past the outermost winding. 0.7 drives the softplus into its tails.
@pytest.mark.parametrize('logit_std', [0.01, 0.7])
@pytest.mark.parametrize('truncate_frac', [None, 0.6])
@pytest.mark.parametrize('device,switch,module', BACKENDS,
                         ids=[b[0] for b in BACKENDS])
def test_fused_gap_matches_eager(monkeypatch, device, switch, module,
                                 truncate_frac, logit_std):
    monkeypatch.setenv(switch, '0')
    eager_backend, eager = _run(device, truncate_frac, logit_std)
    monkeypatch.setenv(switch, '1')
    fused_backend, fused = _run(device, truncate_frac, logit_std)
    assert eager_backend is None
    assert fused_backend is module

    # Tolerances are fractions of each quantity's largest magnitude, about 5x
    # the fused-vs-eager spread measured on MPS with torch 2.11 and 2.12. Both
    # paths are float32: points past the outer winding extrapolate (frac > 1),
    # and dr_per_winding's gradient sums signed per-point terms, so the spread
    # sits well above one ulp, and equally above a float64 reference.
    for key, tol in (('mapped', 2e-4), ('restored', 2e-4),
                     ('grad_points', 1e-4), ('grad_logits', 5e-4),
                     ('grad_dr', 2e-3)):
        scale = float(eager[key].abs().max())
        torch.testing.assert_close(fused[key], eager[key], rtol=0,
                                   atol=tol * scale)


@pytest.mark.parametrize('device,switch,module', BACKENDS,
                         ids=[b[0] for b in BACKENDS])
def test_fused_gap_accepts_no_points(device, switch, module):
    # Called directly: the transform's own lerp fails on empty MPS tensors in
    # torch 2.11, whatever computes the radii.
    params = GapExpanderParams(
        resolution=24, min_z=MIN_Z, max_z=MAX_Z, num_windings=CAPACITY,
        dr_per_winding=DR).to(device)
    logits = params.logits.reshape(params.logits.shape[-2:])
    idx = params.winding_first_logit_idx.float().contiguous()
    dr = torch.tensor(DR, device=device, requires_grad=True)
    empty = torch.zeros(0, device=device, requires_grad=True)
    consts = (float(idx[-1]), MIN_Z, MAX_Z, 1.0, 4.0, 4.0181499, 100.0)
    r_in, r_out = module.gap_bracketing_radii(
        empty, empty, logits, idx, consts[0], dr,
        torch.zeros(0, dtype=torch.int64, device=device), None,
        *consts[1:])
    s_in, s_out, iw = module.gap_search_radii(
        empty, empty, logits, idx, consts[0], dr, empty.detach(), None,
        *consts[1:])
    (r_in.sum() + r_out.sum() + s_in.sum() + s_out.sum()).backward()
    assert r_in.shape == r_out.shape == s_in.shape == iw.shape == (0,)
    assert float(params.logits.grad.abs().sum()) == 0.0
    assert float(dr.grad) == 0.0


def test_inputs_cover_the_inverse_never_found_case():
    # The inverse's never-found branch (a radius past the outermost winding)
    # needs such points; guard that coverage against input changes.
    from transforms import get_theta_and_radii
    logits, points, _ = _inputs(4096, 0.01)
    params = GapExpanderParams(
        resolution=24, min_z=MIN_Z, max_z=MAX_Z, num_windings=CAPACITY,
        dr_per_winding=DR)
    with torch.no_grad():
        params.logits.copy_(logits)
    dr = torch.tensor(DR)
    transform = GapExpandingTransform(
        params, dr, MIN_Z, MAX_Z, gap_expander_lr_scale=0.3, min_gap=1.0,
        softplus_bias=4.0)
    with torch.no_grad():
        theta, radius, _ = get_theta_and_radii(points[..., 1:], dr)
        radii = transform.get_transformed_winding_radii(theta, points[..., 0])
    past = int((radius > radii[..., -1]).sum())
    below = int((radius < radii[..., 0]).sum())
    assert past >= 50 and below >= 50, (past, below)
