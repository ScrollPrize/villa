"""Flow-gradient smoothing and lazy Adam moments.

Both are optional run-boundary optimizer settings, off by default. Covers:
the separable Cartesian blur against a dense 3-D reference, border
renormalisation, slab and component independence, the cylindrical z and
circular-ring passes, the model-level width conversion; LazyMomentAdamW
against torch.optim.SparseAdam on sparse gradients and against plain AdamW
for non-lazy groups, including toggling the flag between steps and
checkpoint round-trips; and the config classification of the new keys.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flow_grad_smoothing
from config import Config
from flow_fields import CartesianFlowField, CylindricalFlowField
from lazy_moment_adamw import LazyMomentAdamW
from transforms import SpiralAndTransform


def _dense_reference(grad, sigma):
    # Full 3-D Gaussian with per-position border renormalisation.
    kernel = flow_grad_smoothing.gaussian_kernel(sigma, dtype=torch.float64)
    radius = kernel.numel() // 2
    kernel3 = kernel[:, None, None] * kernel[None, :, None] * kernel[None, None, :]
    volumes = grad.to(torch.float64).reshape(-1, 1, *grad.shape[-3:])
    ones = torch.ones_like(volumes[:1])
    norm = F.conv3d(ones, kernel3[None, None], padding=radius)
    out = F.conv3d(volumes, kernel3[None, None], padding=radius) / norm
    return out.view(grad.shape).to(grad.dtype)


def test_kernel_is_normalised_and_tiny_widths_are_identity():
    kernel = flow_grad_smoothing.gaussian_kernel(1.5)
    assert kernel.numel() == 2 * math.ceil(4.5) + 1
    assert float(kernel.sum()) == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(kernel, kernel.flip(0))
    assert flow_grad_smoothing.gaussian_kernel(0.0) is None
    assert flow_grad_smoothing.gaussian_kernel(0.01) is None
    grad = torch.randn(2, 3, 5, 6, 7)
    before = grad.clone()
    flow_grad_smoothing.smooth_cartesian_(grad, 0.01)
    assert torch.equal(grad, before)


def test_cartesian_matches_dense_reference_and_keeps_constants():
    torch.manual_seed(0)
    grad = torch.randn(2, 3, 7, 9, 11)
    expected = _dense_reference(grad, 1.2)
    flow_grad_smoothing.smooth_cartesian_(grad, 1.2)
    torch.testing.assert_close(grad, expected, rtol=1e-5, atol=1e-6)
    constant = torch.full((1, 3, 6, 6, 6), 2.5)
    flow_grad_smoothing.smooth_cartesian_(constant, 2.0)
    torch.testing.assert_close(constant, torch.full_like(constant, 2.5))


def test_cartesian_slabs_and_components_do_not_mix():
    # The delta's support (radius 3) stays clear of the borders, where the
    # renormalisation preserves constants rather than mass.
    grad = torch.zeros(2, 3, 15, 15, 15)
    grad[1, 2, 7, 7, 7] = 1.0
    flow_grad_smoothing.smooth_cartesian_(grad, 1.0)
    assert float(grad[0].abs().sum()) == 0.0
    assert float(grad[1, :2].abs().sum()) == 0.0
    assert float(grad[1, 2].sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(grad[1, 2, 7, 7, 7]) < 1.0
    assert float(grad[1, 2, 7, 7, 8]) == pytest.approx(float(grad[1, 2, 7, 7, 6]))


def _cylinder_tables(nr):
    num_phi = [1] + [max(1, int(round(2 * math.pi * r))) for r in range(1, nr)]
    offsets = [0]
    for n in num_phi:
        offsets.append(offsets[-1] + n)
    return num_phi, offsets


def test_cylindrical_smooths_z_and_wraps_rings_without_mixing_them():
    num_phi, offsets = _cylinder_tables(5)
    nz = 15  # the delta's z support (radius 3) stays clear of the borders
    grad = torch.zeros(2, 3, nz, offsets[-1])
    # A delta on ring 3 at phi index 0, middle z, slab 1, component 1.
    ring, start = 3, offsets[3]
    grad[1, 1, 7, start] = 1.0
    grad[..., :1] = 5.0  # ring 0 (the pinned axis cell) must not change
    flow_grad_smoothing.smooth_cylindrical_(grad, num_phi, offsets, 1.0)
    torch.testing.assert_close(grad[..., 0], torch.full_like(grad[..., 0], 5.0))
    assert float(grad[0, :, :, 1:].abs().sum()) == 0.0
    assert float(grad[1, [0, 2], :, 1:].abs().sum()) == 0.0
    spread = grad[1, 1, :, start:start + num_phi[ring]]
    assert float(spread.sum()) == pytest.approx(1.0, abs=1e-5)
    # Circular: the last cell of the ring is the delta's neighbour.
    assert float(spread[7, -1]) == pytest.approx(float(spread[7, 1]), rel=1e-5)
    assert float(spread[7, -1]) > 0.0
    # Other rings untouched.
    others = torch.ones(offsets[-1], dtype=torch.bool)
    others[start:start + num_phi[ring]] = False
    others[0] = False
    assert float(grad[1, 1][:, others].abs().sum()) == 0.0
    # z smoothing renormalises at the border: a constant ring stays constant.
    constant = torch.full((1, 3, nz, offsets[-1]), 1.5)
    flow_grad_smoothing.smooth_cylindrical_(constant, num_phi, offsets, 1.5)
    torch.testing.assert_close(constant, torch.full_like(constant, 1.5))


@pytest.mark.parametrize('kind', ['cartesian', 'cylindrical'])
def test_field_smooth_grad_scales_width_for_the_low_res_lattice(kind, monkeypatch):
    if kind == 'cartesian':
        field = CartesianFlowField(torch.tensor([24, 24, 24]), spatial_scale_factor=4)
    else:
        field = CylindricalFlowField((24, 24, 24), spatial_scale_factor=4)
    for flow in field.flows:
        flow.grad = torch.randn_like(flow)
    seen = []
    target = 'smooth_cartesian_' if kind == 'cartesian' else 'smooth_cylindrical_'
    original = getattr(flow_grad_smoothing, target)

    def recording(grad, *args):
        seen.append((grad.shape, args[-1]))
        return original(grad, *args)

    monkeypatch.setattr(flow_grad_smoothing, target, recording)
    field.smooth_grad_(2.0)
    assert [sigma for _, sigma in seen] == [0.5, 2.0]
    assert seen[0][0] == field.flows[0].shape and seen[1][0] == field.flows[1].shape
    # Untouched lattices (no gradient) are skipped.
    field.flows[0].grad = None
    seen.clear()
    field.smooth_grad_(2.0)
    assert len(seen) == 1


def test_model_converts_voxels_to_cells():
    config = Config().as_dict()
    config.update({
        'model_flow_voxel_resolution': 16,
        'model_gap_expander_capacity_windings': 8,
        'model_gap_expander_num_windings': 8,
        'model_linear_z_resolution': 8,
    })
    model = SpiralAndTransform(
        flow_integration_steps=2, flow_integration_solver='rk4',
        flow_min_corner_zyx=torch.tensor([0, -48, -48]),
        flow_max_corner_zyx=torch.tensor([48, 48, 48]),
        umbilicus_zyx=torch.zeros(48, 3), config=config)
    seen = []
    # Every flow stage module receives the converted width.
    for flow_field in model.flow_fields:
        flow_field.smooth_grad_ = lambda sigma: seen.append(sigma)
    model.smooth_flow_grad_(40.0)
    assert len(model.flow_fields) == 2
    assert seen == [2.5, 2.5]


# ------------------------------------------------------------ lazy moments

def _sparse_pattern(shape, density, generator):
    mask = torch.rand(shape, generator=generator) < density
    return torch.randn(shape, generator=generator) * mask


def test_lazy_groups_match_sparse_adam_and_others_match_adamw():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    lazy_param = torch.nn.Parameter(torch.randn(2, 3, 5, 5))
    dense_param = torch.nn.Parameter(torch.randn(7))
    ref_lazy = torch.nn.Parameter(lazy_param.detach().clone())
    ref_dense = torch.nn.Parameter(dense_param.detach().clone())
    lr, betas, eps = 1e-2, (0.9, 0.999), 1e-8
    optimiser = LazyMomentAdamW([
        {'params': [dense_param], 'weight_decay': 0.01},
        {'params': [lazy_param], 'weight_decay': 0.0, 'lazy_moments': True},
    ], lr=lr, betas=betas, eps=eps)
    sparse_adam = torch.optim.SparseAdam([ref_lazy], lr=lr, betas=betas, eps=eps)
    adamw = torch.optim.AdamW([ref_dense], lr=lr, betas=betas, eps=eps, weight_decay=0.01)
    for _ in range(12):
        lazy_grad = _sparse_pattern(lazy_param.shape, 0.2, generator)
        dense_grad = torch.randn(7, generator=generator)
        lazy_param.grad = lazy_grad.clone()
        dense_param.grad = dense_grad.clone()
        ref_lazy.grad = lazy_grad.to_sparse()
        ref_dense.grad = dense_grad.clone()
        optimiser.step()
        sparse_adam.step()
        adamw.step()
        # SparseAdam adds epsilon before the bias-corrected root, AdamW after:
        # an epsilon-scale difference, well inside this tolerance.
        torch.testing.assert_close(lazy_param, ref_lazy, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(dense_param, ref_dense)
    state = optimiser.state[lazy_param]
    ref_state = sparse_adam.state[ref_lazy]
    torch.testing.assert_close(state['exp_avg'], ref_state['exp_avg'], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(state['exp_avg_sq'], ref_state['exp_avg_sq'], rtol=1e-5, atol=1e-7)
    assert float(state['step']) == 12


def test_lazy_flag_can_toggle_between_steps_and_round_trips_state():
    torch.manual_seed(2)
    generator = torch.Generator().manual_seed(3)
    param = torch.nn.Parameter(torch.randn(4, 6))
    optimiser = LazyMomentAdamW([{'params': [param], 'lazy_moments': False}], lr=1e-2)
    param.grad = _sparse_pattern(param.shape, 0.5, generator)
    optimiser.step()  # fused/plain AdamW path initialises the state
    state = optimiser.state[param]
    assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
    before_avg = state['exp_avg'].clone()
    optimiser.param_groups[0]['lazy_moments'] = True
    grad = _sparse_pattern(param.shape, 0.3, generator)
    param.grad = grad.clone()
    optimiser.step()
    untouched = grad == 0
    torch.testing.assert_close(state['exp_avg'][untouched], before_avg[untouched])
    assert not torch.equal(state['exp_avg'][~untouched], before_avg[~untouched])
    assert float(state['step']) == 2
    # Back to eager AdamW with the state the lazy step left behind.
    optimiser.param_groups[0]['lazy_moments'] = False
    param.grad = torch.randn(param.shape, generator=generator)
    optimiser.step()
    assert float(state['step']) == 3
    # The state dict is a plain AdamW state dict.
    plain = torch.optim.AdamW([torch.nn.Parameter(param.detach().clone())], lr=1e-2)
    plain.load_state_dict(optimiser.state_dict())
    assert float(next(iter(plain.state.values()))['step']) == 3


def test_lazy_step_applies_decoupled_weight_decay_everywhere():
    param = torch.nn.Parameter(torch.ones(10))
    optimiser = LazyMomentAdamW(
        [{'params': [param], 'lazy_moments': True, 'weight_decay': 0.5}], lr=0.1)
    param.grad = torch.zeros(10)
    optimiser.step()
    torch.testing.assert_close(param.detach(), torch.full((10,), 0.95))


def test_lazy_step_with_no_state_and_first_touch_matches_sparse_adam():
    param = torch.nn.Parameter(torch.zeros(5))
    ref = torch.nn.Parameter(torch.zeros(5))
    optimiser = LazyMomentAdamW([{'params': [param], 'lazy_moments': True}], lr=0.1)
    sparse_adam = torch.optim.SparseAdam([ref], lr=0.1)
    grad = torch.tensor([0.0, 2.0, 0.0, -1.0, 0.0])
    param.grad = grad.clone()
    ref.grad = grad.to_sparse()
    optimiser.step()
    sparse_adam.step()
    torch.testing.assert_close(param, ref, rtol=1e-5, atol=1e-6)
    assert float(param.detach()[0]) == 0.0


# -------------------------------------------------------------------- config

def test_new_optimizer_keys_are_run_boundary_and_off_by_default():
    fields = Config.catalog()['schema']['fields']
    defaults = Config().as_dict()
    for key in ('optimizer_flow_grad_smoothing', 'optimizer_flow_lazy_moments'):
        assert fields[key]['type'] == 'boolean'
        assert fields[key]['runtime_impact'] == 'run_boundary'
        assert defaults[key] is False
        assert 'description' in fields[key]
    sigma = 'optimizer_flow_grad_smoothing_sigma_voxels'
    assert fields[sigma]['type'] == 'number'
    assert fields[sigma]['runtime_impact'] == 'run_boundary'
    assert defaults[sigma] == 32.0


cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not __import__('flow_triton')._HAS_TRITON,
    reason='requires CUDA and Triton')


@cuda
@pytest.mark.parametrize('sigma', [0.7, 1.6, 3.0])
def test_fused_cartesian_blur_matches_conv_reference(sigma, monkeypatch):
    torch.manual_seed(5)
    grad = torch.randn(2, 3, 13, 21, 37, device='cuda')
    fused = flow_grad_smoothing.smooth_cartesian_(grad.clone(), sigma)
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '0')
    reference = flow_grad_smoothing.smooth_cartesian_(grad.clone(), sigma)
    torch.testing.assert_close(fused, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fused.cpu(), _dense_reference(grad.cpu(), sigma), rtol=1e-5, atol=1e-6)


@cuda
@pytest.mark.parametrize('sigma', [0.7, 1.6, 3.0])
def test_fused_cylindrical_blur_matches_conv_reference(sigma, monkeypatch):
    torch.manual_seed(6)
    num_phi, offsets = _cylinder_tables(9)
    grad = torch.randn(2, 3, 11, offsets[-1], device='cuda')
    num_phi_t = torch.tensor(num_phi, device='cuda')
    offsets_t = torch.tensor(offsets, device='cuda')
    fused = flow_grad_smoothing.smooth_cylindrical_(grad.clone(), num_phi_t, offsets_t, sigma)
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '0')
    reference = flow_grad_smoothing.smooth_cylindrical_(grad.clone(), num_phi, offsets, sigma)
    torch.testing.assert_close(fused, reference, rtol=1e-5, atol=1e-6)


@cuda
def test_fused_blur_handles_awkward_sizes_and_wide_kernels(monkeypatch):
    # Axis lengths below the tile, rings shorter than the kernel, and a kernel
    # wider than the fused unroll (which takes the conv path).
    grad = torch.randn(1, 3, 3, 5, 300, device='cuda')
    fused = flow_grad_smoothing.smooth_cartesian_(grad.clone(), 1.0)
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '0')
    reference = flow_grad_smoothing.smooth_cartesian_(grad.clone(), 1.0)
    torch.testing.assert_close(fused, reference, rtol=1e-5, atol=1e-6)
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '1')
    wide = torch.randn(1, 3, 4, 4, 200, device='cuda')
    sigma = (flow_grad_smoothing.MAX_FUSED_RADIUS + 1) / flow_grad_smoothing.KERNEL_TRUNCATE
    fused_wide = flow_grad_smoothing.smooth_cartesian_(wide.clone(), sigma)
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '0')
    torch.testing.assert_close(
        fused_wide, flow_grad_smoothing.smooth_cartesian_(wide.clone(), sigma))
