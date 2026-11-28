"""The fused direct-LR RK4 kernels against an eager reference integrator.

flow_triton.TritonRK4DirectIntegrate (CUDA) and flow_metal.MetalRK4DirectIntegrate
(MPS) integrate points through every slab of a low- and a high-resolution
lattice, sampling the low one directly, and flow_fields' eager integrator
(_EagerDirect, every device) does where neither kernel runs. The reference
(direct_lr_reference.py) runs the same integrator in eager PyTorch with
autograd: trilinear samples at the lattice
coordinates `normalised * a + b` from flow_triton._field_geoms (border clamp),
RK4 with the kernels' stage arithmetic. Each available backend must match it:
- the integrated points, forward and with `reverse`;
- the field gradients accumulated into acc_lo / acc_hi;
- the point gradients. The trilinear interpolant's derivative jumps across
  lattice planes, so a stage point whose float32 rounding lands on the other
  side of a plane can legitimately change its point's gradient. A handful of
  such points is allowed; everything else must match tightly.

A geometry test checks the direct sample against the upsampled field
(F.interpolate, then grid_sample) on high-resolution lattice nodes, where the
two interpolants agree by construction.
"""
import pytest
import torch
import torch.nn.functional as F

import flow_fields
import flow_metal
import flow_triton
from direct_lr_reference import reference_integrate, trilinear


def _backends():
    # The eager integrator (direct sampling where no fused kernel runs) is
    # held to the same contract, on the CPU.
    backends = [('cpu', None, flow_fields._EagerDirect)]
    if flow_triton._HAS_TRITON and torch.cuda.is_available():
        backends.append(('cuda', 'FIT_SPIRAL_TRITON', flow_triton))
    if torch.backends.mps.is_available():
        backends.append(('mps', 'FIT_SPIRAL_METAL', flow_metal))
    return backends


BACKENDS = _backends()
LOW = (2, 3, 5, 6, 7)       # slabs, 3, Zl, Yl, Xl
HIGH = (2, 3, 17, 19, 23)   # slabs, 3, Z, Y, X (ratios are not integers)


def _case(n=4096, seed=7):
    gen = torch.Generator().manual_seed(seed)
    low = torch.randn(LOW, generator=gen) * 0.05
    high = torch.randn(HIGH, generator=gen) * 0.02
    y0 = torch.rand(n, 3, generator=gen) * 1.1 - 0.05   # some outside [0, 1]
    grad = torch.randn(n, 3, generator=gen)
    return low, high, y0, grad


def _run_fused(module, device, reverse, n_steps, scales):
    low, high, y0, grad = _case()
    low, high = low.to(device), high.to(device)
    acc_lo, acc_hi = torch.zeros_like(low), torch.zeros_like(high)
    y = y0.to(device).requires_grad_(True)
    out = module.rk4_direct_integrate(
        y, low, high, scales[0], scales[1], acc_lo, acc_hi, 1.0 / n_steps,
        n_steps, reverse)
    out.backward(grad.to(device))
    no_grad = module.rk4_direct_integrate(
        y0.to(device), low, high, scales[0], scales[1], None, None,
        1.0 / n_steps, n_steps, reverse)
    return {'out': out.detach().cpu(), 'no_grad': no_grad.cpu(),
            'grad_points': y.grad.cpu(), 'acc_lo': acc_lo.cpu(),
            'acc_hi': acc_hi.cpu()}


def _run_reference(reverse, n_steps, scales):
    low, high, y0, grad = _case()
    low = low.requires_grad_(True)
    high = high.requires_grad_(True)
    y = y0.clone().requires_grad_(True)
    out = reference_integrate(y, low, high, scales[0], scales[1],
                               1.0 / n_steps, n_steps, reverse)
    out.backward(grad)
    # The kernels accumulate unscaled field gradients.
    return {'out': out.detach(), 'grad_points': y.grad,
            'acc_lo': low.grad / scales[0], 'acc_hi': high.grad / scales[1]}


@pytest.mark.parametrize('scales', [(1.0, 1.0), (0.7, 1.3)])
@pytest.mark.parametrize('n_steps', [1, 3])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('device,switch,module', BACKENDS,
                         ids=[b[0] for b in BACKENDS])
def test_fused_direct_rk4_matches_reference(device, switch, module, reverse,
                                            n_steps, scales):
    ref = _run_reference(reverse, n_steps, scales)
    fused = _run_fused(module, device, reverse, n_steps, scales)

    torch.testing.assert_close(fused['out'], ref['out'], rtol=0, atol=2e-6)
    torch.testing.assert_close(fused['no_grad'], fused['out'], rtol=0, atol=0)
    for key in ('acc_lo', 'acc_hi'):
        scale = float(ref[key].abs().max())
        torch.testing.assert_close(
            fused[key], ref[key], rtol=0, atol=2e-5 * scale)

    diff = (fused['grad_points'] - ref['grad_points']).abs().max(dim=-1).values
    tol = 1e-4 * float(ref['grad_points'].abs().max())
    outliers = int((diff > tol).sum())
    assert outliers <= 8, f'{outliers} points with point-gradient error above {tol:.3g}'


@pytest.mark.parametrize('device,switch,module', BACKENDS,
                         ids=[b[0] for b in BACKENDS])
def test_fused_direct_rk4_accepts_no_points(device, switch, module):
    low, high, _, _ = _case()
    low, high = low.to(device), high.to(device)
    y = torch.zeros(0, 3, device=device, requires_grad=True)
    out = module.rk4_direct_integrate(
        y, low, high, 1.0, 1.0, torch.zeros_like(low), torch.zeros_like(high),
        0.5, 2, False)
    out.backward(torch.zeros_like(out))
    assert out.shape == (0, 3)
    assert y.grad.shape == (0, 3)


def test_direct_sample_matches_upsampled_field_on_high_res_nodes():
    # Direct sampling composes F.interpolate(align_corners=False)'s
    # source-index map with grid_sample(align_corners=True), so on HR lattice
    # nodes it equals sampling the upsampled LR field.
    low, high, _, _ = _case()
    low, high = low.double(), high.double()
    geom = flow_triton._field_geoms(low, high)
    Z, Y, X = HIGH[2:]
    axes = [torch.arange(n, dtype=torch.float64) / (n - 1) for n in (Z, Y, X)]
    grid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, 3)
    upsampled = F.interpolate(low, size=(Z, Y, X), mode='trilinear')
    for slab in range(LOW[0]):
        direct = trilinear(low[slab], grid, geom[:6])
        reference = flow_fields.sample_field(grid, upsampled[slab])
        torch.testing.assert_close(direct, reference, rtol=0, atol=1e-12)
