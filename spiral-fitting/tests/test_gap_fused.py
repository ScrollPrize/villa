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
  terms, so they are compared against the terms' exact absolute sums
  (_absolute_sums, float64), not the (cancelling) results, element by
  element, and the logit gradient also as a whole (its projection on the
  eager one).
The tolerances are set from the largest spreads over 200 seeds of each case
on MPS (GAP_FUSED_SEEDS=200 runs them; see LOGIT_TOL), and the eager path
itself is as far from a float64 run as the kernel is. They were measured on
MPS only: on CUDA the same bounds hold the Triton kernels to the eager path
untried.
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
# Tolerances of the accumulated gradients (see test_fused_gap_matches_eager),
# against the largest spreads over 200 seeds of each case on MPS (torch 2.11;
# the spread of CUDA's Triton kernels against its eager path was not measured
# for these bounds):
# - per logit element, |fused - eager| against the element's terms' absolute
#   sum plus 1e-5 of the largest: up to 0.52. Two float32 orders do not
#   agree on every element (a point past the outermost winding, for one,
#   extrapolates its bracket and moved one element by 2.5% of its terms), so
#   this bound catches misrouted contributions, not small biases;
# - the projection of the fused logit gradient on the eager one: within
#   5.4e-5 of 1, so 1e-3 catches a bias of 0.1% in the scatter;
# - dr_per_winding, against its terms' absolute sum: up to 1.3e-5, so 4e-5.
#   Where that sum cancels, a 1% error in the kernel's dr terms can hide
#   below it (7 of 40 cases over 10 seeds), so dr_per_winding's gradient
#   through a radial loss (_radial_loss) is compared too, against its own
#   terms' absolute sum: up to 1.5e-5 over 800 cases, against at least
#   8.8e-3 with that 1% error, so 2e-4.
LOGIT_TOL, LOGIT_FLOOR = 1.0, 1e-5
PROJECTION_TOL = 1e-3
DR_TOL = 4e-5
RADIAL_DR_TOL = 2e-4


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
    """Each accumulated gradient's terms' absolute sum, exactly, in float64.

    Logits: an element's gradient sums g * w over every (point, winding)
    sample whose bilinear corners include it, with g the loss gradient at
    the sampled logit and w >= 0 the corner weight. Recording both maps'
    samples, then backpropagating |g| through the same samples, gives the
    sum of |g| * w per element. dr_per_winding: each point's loss term's
    derivative by central differences, summed in absolute value.
    """
    params, _, transform = _transform('cpu', truncate_frac, logits, torch.float64)
    x, w = points.double(), weights.double()
    samples, sample = [], transform.get_logits_by_winding

    def recording(theta, z):
        out = sample(theta, z)
        out.retain_grad()
        samples.append((theta.detach(), z.detach(), out))
        return out
    transform.get_logits_by_winding = recording
    ((transform._call(x) * w[0]).sum() + (transform._inverse(x) * w[1]).sum()).backward()
    transform.get_logits_by_winding = sample
    params.logits.grad = None
    transform._pinned_scaled_logits = None   # its graph went with that backward
    sum((sample(theta, z) * out.grad.abs()).sum()
        for theta, z, out in samples).backward()
    logits_sums = params.logits.grad.abs().reshape(-1)
    terms = []
    with torch.no_grad():
        for step in (1e-6, -1e-6):
            _, _, shifted = _transform('cpu', truncate_frac, logits, torch.float64)
            shifted.dr_per_winding = torch.tensor(DR + step, dtype=torch.float64)
            terms.append((shifted._call(x) * w[0]).sum(-1)
                         + (shifted._inverse(x) * w[1]).sum(-1))
    dr_sum = float(((terms[0] - terms[1]) / 2e-6).abs().sum())
    return logits_sums, dr_sum


def _run(device, truncate_frac, logit_std=0.01, n=4096, seed=1234, sums=False):
    logits, points, weights = _inputs(n, logit_std, seed)
    keep = _unambiguous(truncate_frac, logits, points)
    points, weights = points[keep], weights[:, keep]
    params, dr, transform = _transform(device, truncate_frac, logits)
    x = points.to(device).requires_grad_(True)
    mapped = transform._call(x)
    restored = transform._inverse(x)
    on_device = weights.to(device)
    ((mapped * on_device[0]).sum() + (restored * on_device[1]).sum()).backward()
    backend = transform._fused_gap(x, dr)
    out = {
        'mapped': mapped, 'restored': restored, 'grad_points': x.grad,
        'grad_logits': params.logits.grad, 'grad_dr': dr.grad.reshape(1),
    }
    out = {k: v.detach().cpu() for k, v in out.items()}
    out['radial_dr'] = _radial_dr(device, truncate_frac, logits, points)
    if sums:
        out['absolute_sums'] = _absolute_sums(truncate_frac, logits, points, weights)
        out['radial_dr_sum'] = _radial_dr_sum(truncate_frac, logits, points)
    return backend, out


def _radial_loss(transform, x):
    # The outward unit vector as the forward map's loss weight, its negation
    # as the inverse's: the kernel's dr terms (gap derivative times the
    # radius gradient) then share a sign, so a bias in them adds up instead
    # of cancelling away as it can under random weights.
    outward = torch.cat([torch.zeros_like(x[..., :1]),
                         torch.nn.functional.normalize(x[..., 1:], dim=-1)], dim=-1)
    return ((transform._call(x) * outward).sum(-1)
            - (transform._inverse(x) * outward).sum(-1))


def _radial_dr(device, truncate_frac, logits, points):
    _, dr, transform = _transform(device, truncate_frac, logits)
    _radial_loss(transform, points.to(device)).sum().backward()
    return float(dr.grad)


def _radial_dr_sum(truncate_frac, logits, points):
    # The radial loss's dr_per_winding terms' absolute sum: each point's
    # term by central differences, in float64.
    terms = []
    with torch.no_grad():
        for step in (1e-6, -1e-6):
            _, _, shifted = _transform('cpu', truncate_frac, logits, torch.float64)
            shifted.dr_per_winding = torch.tensor(DR + step, dtype=torch.float64)
            terms.append(_radial_loss(shifted, points.double()))
    return float(((terms[0] - terms[1]) / 2e-6).abs().sum())


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
    eager_backend, eager = _run(device, truncate_frac, logit_std, seed=seed, sums=True)
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
    # The sums, element by element against their terms' absolute sums (plus
    # a floor of LOGIT_FLOOR of the largest: an element whose whole sum is a
    # rounding-sized corner weight of a sample on a cell edge), and the logit
    # gradient as a whole by its projection on the eager one, which a bias
    # in the scatter moves.
    logits_sums, dr_sum = eager['absolute_sums']
    fused_logits = fused['grad_logits'].reshape(-1).double()
    eager_logits = eager['grad_logits'].reshape(-1).double()
    bound = LOGIT_TOL * (logits_sums + LOGIT_FLOOR * logits_sums.max())
    over = int(((fused_logits - eager_logits).abs() > bound).sum())
    assert over == 0, f'{over} logit-gradient elements over their bound'
    projection = float(fused_logits @ eager_logits / (eager_logits @ eager_logits))
    assert abs(projection - 1.0) <= PROJECTION_TOL, projection
    dr_error = float((fused['grad_dr'] - eager['grad_dr']).abs())
    assert dr_error <= DR_TOL * dr_sum, (dr_error, dr_sum)
    radial = abs(fused['radial_dr'] - eager['radial_dr'])
    assert radial <= RADIAL_DR_TOL * eager['radial_dr_sum'], (radial, eager['radial_dr_sum'])


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


# The boundary reference is MPS's eager path: its border sample is built from
# gathers that repeat the last row or column (flat there) and take the
# one-sided slope on the first. On CUDA the eager path is grid_sample, which
# gives no gradient on either boundary, and the Triton kernels keep the slope
# on both: their boundary convention is not this one, so these run on MPS.
needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="the boundary reference is MPS's eager path (gathers); CUDA's "
           'grid_sample and the Triton kernels take other conventions there')


# z exactly at max_z puts the logit lattice's row coordinate on its last row:
# MPS's eager path gives z no gradient through the logits there, and the Metal
# kernel must not either; z at min_z is the first row, where both take the
# one-sided slope.
@needs_mps
@pytest.mark.parametrize('z', [MAX_Z, MIN_Z], ids=['upper', 'lower'])
@pytest.mark.parametrize('truncate_frac', [None, 0.6])
def test_fused_gap_gradient_at_the_z_boundaries_matches_eager(
        monkeypatch, truncate_frac, z):
    device, switch = 'mps', 'FIT_SPIRAL_METAL'
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


@needs_mps
def test_metal_gap_gradient_on_the_last_logit_column_matches_eager(monkeypatch):
    # The last column, exactly. With dr_per_winding 2 and 5 windings the
    # lattice has 6 columns, windings starting at [0, 1, 2, 4, 6]; at
    # theta = 4 pi winding 2's column coordinate is 2 + 2 * (4 - 2) = 6, the
    # last one, in float32 on both paths (the eager lerp gives 4 + 2). The
    # bracketing winding 2 is the last one the backward walks.
    monkeypatch.setenv('FIT_SPIRAL_METAL', '1')
    params = GapExpanderParams(resolution=24, min_z=MIN_Z, max_z=MAX_Z,
                               num_windings=5, dr_per_winding=2.0).to('mps')
    assert params.winding_first_logit_idx.tolist() == [0, 1, 2, 4, 6]
    with torch.no_grad():
        params.logits.copy_(torch.randn(
            params.logits.shape, generator=torch.Generator().manual_seed(5)) * 0.7)
    dr = torch.tensor(2.0, device='mps')
    z = torch.tensor([50.0, 100.0, 150.0], device='mps')   # off the row edges
    iw = torch.full((3,), 2, dtype=torch.int64, device='mps')
    grads = {}
    for path in ('fused', 'eager'):
        # One transform per path: it caches the pinned logits' graph.
        transform = GapExpandingTransform(
            params, dr, MIN_Z, MAX_Z, gap_expander_lr_scale=0.3, min_gap=0.5,
            softplus_bias=4.0)
        consts = transform._triton_consts()
        theta = torch.full((3,), 4 * math.pi, device='mps', requires_grad=True)
        if path == 'fused':
            r_in, r_out = gap_metal.gap_bracketing_radii(
                theta, z, transform._get_pinned_scaled_logits(), consts['idx'],
                consts['idx_total'], dr, iw, None, consts['min_z'],
                consts['max_z'], consts['min_gap'], consts['softplus_bias'],
                consts['softplus_denominator'], consts['softplus_scale'])
        else:
            radii = transform.get_transformed_winding_radii(theta, z)
            r_in = radii.gather(-1, iw[:, None]).squeeze(-1)
            r_out = radii.gather(-1, (iw + 1)[:, None]).squeeze(-1)
        (r_in * 0.3 + r_out * 1.7).sum().backward()
        grads[path] = theta.grad.cpu()
    torch.testing.assert_close(grads['fused'], grads['eager'], rtol=1e-5, atol=1e-5)
