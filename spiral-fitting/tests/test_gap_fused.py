"""The fused gap-expander kernels against the eager path on the same device.

gap_triton runs on CUDA and gap_metal on MPS. Each available backend runs
GapExpandingTransform's forward map (_call) and inverse (_inverse), forward and
backward, and must match the eager path within float32 tolerance: mapped
points, point gradients, gap-logit gradients and the dr_per_winding gradient.
The points cover the border clamps (z outside the logit lattice), the w == 0
clip and radii past the outermost winding (the inverse's never-found case).

Two float32 paths that sum in different orders cannot agree everywhere, and
the comparison is built around where they cannot:
- an inverse point whose radius lies within rounding of a winding radius can
  be bracketed by neighbouring windings on the two paths, which moves its
  whole gradient onto other logits. Such points (float64 radius within
  0.05 voxel of a winding radius) are left out;
- the bilinear logit sample's slope jumps across lattice cell edges, and a
  sample within rounding of an edge may take either slope: a few points may
  differ in the per-point quantities (at most 3 in 4096 measured);
- the logit and dr_per_winding gradients each sum ~10^5 signed float32
  terms, so they are compared against the terms' absolute sum, not the
  (cancelling) result: the absolute sum is the gradient of a radial loss
  with one sign per path, in float64 (_absolute_sums).
Measured on MPS (torch 2.11) over 200 seeds of each case
(GAP_FUSED_SEEDS=200 runs them): the tolerances below are about 5x the
largest spread, and the eager path itself is as far from a float64 run.
"""
import math
import os

import pytest
import torch

import gap_metal
import gap_triton
from transforms import (GapExpanderParams, GapExpandingTransform,
                        get_theta_and_radii)

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


SEEDS = range(1234, 1234 + int(os.environ.get('GAP_FUSED_SEEDS', '1')))


def _inputs(n, logit_std, seed=1234):
    gen = torch.Generator().manual_seed(seed)
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


def _transform(device, truncate_frac, logits, dtype=torch.float32):
    params = GapExpanderParams(
        resolution=24, min_z=MIN_Z, max_z=MAX_Z, num_windings=CAPACITY,
        dr_per_winding=DR).to(device=device, dtype=dtype)
    with torch.no_grad():
        params.logits.copy_(logits.to(device=device, dtype=dtype))
    dr = torch.tensor(DR, device=device, dtype=dtype, requires_grad=True)
    transform = GapExpandingTransform(
        params, dr, MIN_Z, MAX_Z, gap_expander_lr_scale=0.3, min_gap=1.0,
        softplus_bias=4.0, truncate_frac=truncate_frac)
    return params, dr, transform


def _unambiguous(truncate_frac, logits, points, margin=0.05):
    # The points whose inverse bracket no float32 summation of the radii can
    # change: float64 radius at least `margin` voxels from every winding radius.
    _, _, transform = _transform('cpu', truncate_frac, logits, torch.float64)
    with torch.no_grad():
        p = points.double()
        theta, radius, _ = get_theta_and_radii(p[..., 1:], transform.dr_per_winding)
        radii = transform.get_transformed_winding_radii(theta, p[..., 0])
    return (radii - radius[..., None]).abs().min(dim=-1).values >= margin


def _absolute_sums(truncate_frac, logits, points, weights):
    # The gap transform moves points radially, so each point's term in a logit
    # or dr_per_winding gradient is at most |w_p| times its radius' derivative.
    # With the outward unit vector as the forward map's loss weight and its
    # negation as the inverse's, those derivatives all take one sign (forward
    # radii grow with every gap, inverse ones shrink), so this loss's float64
    # gradients are their absolute sums, times the largest |w_p|.
    params, dr, transform = _transform('cpu', truncate_frac, logits, torch.float64)
    x = points.double()
    outward = torch.cat([torch.zeros_like(x[..., :1]),
                         torch.nn.functional.normalize(x[..., 1:], dim=-1)], dim=-1)
    ((transform._call(x) * outward).sum()
     - (transform._inverse(x) * outward).sum()).backward()
    largest = float(weights.norm(dim=-1).max())
    return (float(params.logits.grad.abs().max()) * largest,
            float(dr.grad.abs()) * largest)


def _run(device, truncate_frac, logit_std=0.01, n=4096, seed=1234):
    logits, points, weights = _inputs(n, logit_std, seed)
    keep = _unambiguous(truncate_frac, logits, points)
    points, weights = points[keep], weights[:, keep]
    params, dr, transform = _transform(device, truncate_frac, logits)
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
    out = {k: v.detach().cpu() for k, v in out.items()}
    out['absolute_sums'] = _absolute_sums(truncate_frac, logits, points, weights)
    return backend, out


# 0.01: the gaps stay within about 0.5-2x the nominal spacing, so some points
# lie past the outermost winding. 0.7 drives the softplus into its tails.
@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('logit_std', [0.01, 0.7])
@pytest.mark.parametrize('truncate_frac', [None, 0.6])
@pytest.mark.parametrize('device,switch,module', BACKENDS,
                         ids=[b[0] for b in BACKENDS])
def test_fused_gap_matches_eager(monkeypatch, device, switch, module,
                                 truncate_frac, logit_std, seed):
    monkeypatch.setenv(switch, '0')
    eager_backend, eager = _run(device, truncate_frac, logit_std, seed=seed)
    monkeypatch.setenv(switch, '1')
    fused_backend, fused = _run(device, truncate_frac, logit_std, seed=seed)
    assert eager_backend is None
    assert fused_backend is module

    # Per point, as fractions of each quantity's largest magnitude. At most 4
    # points may take the other slope at a lattice cell edge: over 200 seeds
    # of each case at most 1, 3 and 2 points exceeded these tolerances.
    for key, tol in (('mapped', 2e-4), ('restored', 2e-4),
                     ('grad_points', 1e-4)):
        error = (fused[key] - eager[key]).abs().max(dim=-1).values
        outliers = int((error > tol * float(eager[key].abs().max())).sum())
        assert outliers <= 4, f'{key}: {outliers} points over {tol}'
    # The sums, against their terms' absolute sums (largest spread over 200
    # seeds: 3.8e-4 for the logits, 2.6e-4 for dr_per_winding).
    logits_scale, dr_scale = eager['absolute_sums']
    torch.testing.assert_close(fused['grad_logits'], eager['grad_logits'],
                               rtol=0, atol=2e-3 * logits_scale)
    torch.testing.assert_close(
        fused['grad_dr'], eager['grad_dr'], rtol=0,
        atol=2e-3 * max(dr_scale, float(eager['grad_dr'].abs().max())))


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


# z exactly at max_z puts the logit lattice's row coordinate on its last row.
# The eager path's border sample (on MPS, gathers that repeat the last row)
# gives z no gradient through the logits there, and the kernels must not
# either; z at min_z is the first row, where both take the one-sided slope.
@pytest.mark.parametrize('z', [MAX_Z, MIN_Z], ids=['upper', 'lower'])
@pytest.mark.parametrize('truncate_frac', [None, 0.6])
@pytest.mark.parametrize('device,switch,module', BACKENDS,
                         ids=[b[0] for b in BACKENDS])
def test_fused_gap_gradient_at_the_z_boundaries_matches_eager(
        monkeypatch, device, switch, module, truncate_frac, z):
    logits, points, weights = _inputs(1024, 0.7)
    points[:, 0] = z
    keep = _unambiguous(truncate_frac, logits, points)
    points, weights = points[keep], weights[:, keep]
    runs = {}
    for flag in ('0', '1'):
        monkeypatch.setenv(switch, flag)
        runs[flag] = _run_points(device, truncate_frac, logits, points, weights)
    eager, fused = runs['0'], runs['1']
    # The z component, for every point (the slope choices at cell edges are
    # along the winding axis, which only theta reaches).
    scale = float(eager['grad_points'].abs().max())
    torch.testing.assert_close(fused['grad_points'][:, 0],
                               eager['grad_points'][:, 0],
                               rtol=0, atol=1e-4 * scale)


def _run_points(device, truncate_frac, logits, points, weights):
    params, dr, transform = _transform(device, truncate_frac, logits)
    x = points.to(device).requires_grad_(True)
    weights = weights.to(device)
    ((transform._call(x) * weights[0]).sum()
     + (transform._inverse(x) * weights[1]).sum()).backward()
    return {'grad_points': x.grad.detach().cpu(),
            'grad_logits': params.logits.grad.detach().cpu()}
