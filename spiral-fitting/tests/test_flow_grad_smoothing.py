"""Flow-gradient smoothing and lazy Adam moments.

Both are optional run-boundary optimizer settings, off by default. Covers:
the separable Cartesian blur against a dense 3-D reference, border
renormalisation, slab and component independence, the cylindrical z,
circular-ring and across-ring passes (fused against eager), the model-level
width conversion and the startup width report; LazyMomentAdamW against
torch.optim.SparseAdam on sparse gradients and against plain AdamW for
non-lazy groups, including toggling the flag between steps and checkpoint
round-trips, and its shared (per-plane) second moment; and the config
classification of the new keys.
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
        # Cartesian: (sigma,); cylindrical: (num_phi, offsets, sigma, across).
        widths = args[-1:] if kind == 'cartesian' else args[-2:]
        seen.append((grad.shape, widths))
        return original(grad, *args)

    monkeypatch.setattr(flow_grad_smoothing, target, recording)
    field.smooth_grad_(2.0, 1.0)
    expected = [(0.5,), (2.0,)] if kind == 'cartesian' else [(0.5, 0.25), (2.0, 1.0)]
    assert [widths for _, widths in seen] == expected
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
    # Every flow stage module receives the converted widths.
    for flow_field in model.flow_fields:
        flow_field.smooth_grad_ = lambda sigma, across: seen.append((sigma, across))
    model.smooth_flow_grad_(40.0)
    model.smooth_flow_grad_(40.0, 8.0)
    assert len(model.flow_fields) == 2
    assert seen == [(2.5, 0.0)] * 2 + [(2.5, 0.5)] * 2
    # The startup report converts the same way and names identity kernels.
    report = model.describe_flow_grad_smoothing(6.0, 0.0)
    assert report.startswith('flow gradient smoothing (cartesian): along-sheet 6 voxels')
    assert 'HR 0.38 cells (radius 2)' in report
    assert 'LR 0.06 cells (identity)' in report
    assert 'across' not in report
    assert 'ignored' in model.describe_flow_grad_smoothing(6.0, 16.0)


def test_describe_widths_reports_both_directions_for_cylindrical_lattices():
    report = flow_grad_smoothing.describe_widths(96.0, 16.0, 16.0, 6, 'cylindrical')
    assert 'along-sheet 96 voxels = HR 6.00 cells (radius 18), LR 1.00 cells (radius 3)' in report
    assert 'across rings 16 voxels = HR 1.00 cells (radius 3), LR 0.17 cells (identity)' in report


def test_cylindrical_radial_pass_spreads_across_rings_at_the_same_angle():
    num_phi, offsets = _cylinder_tables(7)
    nz = 5
    grad = torch.zeros(2, 3, nz, offsets[-1])
    ring, start = 3, offsets[3]
    grad[1, 1, 2, start] = 1.0  # angle 0 on ring 3
    grad[..., :1] = 5.0  # ring 0 (the pinned axis cell) must not change
    # Along-sheet width 0 isolates the radial pass.
    flow_grad_smoothing.smooth_cylindrical_(grad, num_phi, offsets, 0.0, 1.0)
    torch.testing.assert_close(grad[..., 0], torch.full_like(grad[..., 0], 5.0))
    # Other slabs, components and z rows untouched.
    assert float(grad[0, :, :, 1:].abs().sum()) == 0.0
    assert float(grad[1, [0, 2], :, 1:].abs().sum()) == 0.0
    assert float(grad[1, 1, [0, 1, 3, 4], 1:].abs().sum()) == 0.0
    row = grad[1, 1, 2]
    kernel = flow_grad_smoothing.gaussian_kernel(1.0).tolist()
    radius = len(kernel) // 2

    def expected(dst_ring):
        # Ring dst reads ring 3 with the weight of tap (3 - dst), renormalised
        # over the taps that land on rings 1..6.
        valid = [kernel[k + radius] for k in range(-radius, radius + 1)
                 if 1 <= dst_ring + k < len(num_phi)]
        return kernel[(3 - dst_ring) + radius] / sum(valid)

    for dst_ring in range(1, 7):
        n = num_phi[dst_ring]
        cells = row[offsets[dst_ring]:offsets[dst_ring] + n]
        # Angle 0 falls exactly on cell 0 of every ring, so cell 0 reads the
        # delta with its full tap weight.
        assert float(cells[0]) == pytest.approx(expected(dst_ring), rel=1e-5)
        for i in range(1, n):
            # A cell at angle i/n reads ring 3 at position i/n * 19; only
            # positions within one source cell of 0 (either way round) see
            # the delta, through the linear interpolation.
            position = i / n * num_phi[ring]
            if position < 1.0 or position > num_phi[ring] - 1:
                assert float(cells[i]) > 0.0
            else:
                assert float(cells[i]) == 0.0
    # Off-cell angles interpolate between a ring's two nearest cells.
    grad = torch.zeros(1, 1, 1, offsets[-1])
    grad[0, 0, 0, offsets[2] + 1] = 1.0  # ring 2 (13 cells), angle 1/13 turn
    flow_grad_smoothing.smooth_cylindrical_(grad, num_phi, offsets, 0.0, 1.0)
    ring3 = grad[0, 0, 0, offsets[3]:offsets[3] + num_phi[3]]  # 19 cells
    # Ring 3's cells 1 (1/19) and 2 (2/19) straddle 1/13 of a turn.
    assert float(ring3[1]) > 0.0 and float(ring3[2]) > 0.0
    assert float(ring3[[0, 3]].abs().sum()) == 0.0
    # A constant plane stays constant with both widths active (border
    # renormalisation and the angular interpolation are both affine-exact).
    constant = torch.full((1, 3, nz, offsets[-1]), 1.5)
    flow_grad_smoothing.smooth_cylindrical_(constant, num_phi, offsets, 1.2, 1.5)
    torch.testing.assert_close(constant, torch.full_like(constant, 1.5))


def test_cylindrical_zero_across_width_is_the_ring_and_z_blur_alone():
    torch.manual_seed(11)
    num_phi, offsets = _cylinder_tables(6)
    grad = torch.randn(2, 3, 9, offsets[-1])
    reference = grad.clone()
    flow_grad_smoothing.smooth_cylindrical_(reference, num_phi, offsets, 1.3)
    flow_grad_smoothing.smooth_cylindrical_(grad, num_phi, offsets, 1.3, 0.0)
    torch.testing.assert_close(grad, reference)
    # Both widths below the identity threshold: a no-op.
    before = grad.clone()
    flow_grad_smoothing.smooth_cylindrical_(grad, num_phi, offsets, 0.01, 0.01)
    assert torch.equal(grad, before)


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


def _reference_shared_step(param, grad, state, lr, betas, eps, masked):
    # Plain-torch reference of the shared-denominator step on a [S, C, ...]
    # parameter: per-cell EMA moments, one denominator per (S, C) plane from
    # the mean second moment over cells with a nonzero second moment.
    beta1, beta2 = betas
    state['step'] += 1
    step = state['step']
    touched = grad != 0 if masked else torch.ones_like(grad, dtype=torch.bool)
    state['exp_avg'] = torch.where(touched, torch.lerp(state['exp_avg'], grad, 1 - beta1), state['exp_avg'])
    state['exp_avg_sq'] = torch.where(touched, torch.lerp(state['exp_avg_sq'], grad * grad, 1 - beta2), state['exp_avg_sq'])
    sq = state['exp_avg_sq']
    planes = sq.reshape(sq.shape[0] * sq.shape[1], -1)
    ever = planes > 0
    mean_sq = planes.sum(1) / ever.sum(1).clamp(min=1)
    denom = (mean_sq.sqrt() / (1 - beta2 ** step) ** 0.5 + eps).view(sq.shape[0], sq.shape[1], *([1] * (sq.dim() - 2)))
    update = state['exp_avg'] / denom * (lr / (1 - beta1 ** step))
    update = torch.where(touched, update, torch.zeros_like(update))
    return param - update


@pytest.mark.parametrize('masked', [True, False])
def test_shared_second_moment_matches_reference_and_keeps_adamw_state(masked):
    torch.manual_seed(4)
    generator = torch.Generator().manual_seed(5)
    param = torch.nn.Parameter(torch.randn(2, 3, 4, 5))
    ref = param.detach().clone()
    lr, betas, eps = 1e-2, (0.9, 0.999), 1e-8
    optimiser = LazyMomentAdamW([{
        'params': [param], 'weight_decay': 0.0,
        'lazy_moments': masked, 'shared_second_moment': True,
    }], lr=lr, betas=betas, eps=eps)
    ref_state = {'step': 0, 'exp_avg': torch.zeros_like(ref), 'exp_avg_sq': torch.zeros_like(ref)}
    for _ in range(6):
        grad = _sparse_pattern(param.shape, 0.4, generator)
        # Plane (1, 2) never receives gradient: its denominator falls back to
        # epsilon and its (zero) first moment keeps it exactly still.
        grad[1, 2] = 0.0
        param.grad = grad.clone()
        optimiser.step()
        ref = _reference_shared_step(ref, grad, ref_state, lr, betas, eps, masked)
        torch.testing.assert_close(param.detach(), ref, rtol=1e-5, atol=1e-7)
    assert torch.equal(param.detach()[1, 2], ref[1, 2])
    state = optimiser.state[param]
    assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
    torch.testing.assert_close(state['exp_avg_sq'], ref_state['exp_avg_sq'])
    assert float(state['step']) == 6
    # The per-cell state is untouched by the flag, so a plain AdamW loads it
    # and the fused step continues from it.
    plain = torch.optim.AdamW([torch.nn.Parameter(param.detach().clone())], lr=lr)
    plain.load_state_dict(optimiser.state_dict())
    optimiser.param_groups[0]['shared_second_moment'] = False
    optimiser.param_groups[0]['lazy_moments'] = False
    param.grad = torch.randn(param.shape, generator=generator)
    optimiser.step()
    assert float(state['step']) == 7


def test_shared_second_moment_keeps_the_gradient_profile_and_is_per_plane():
    # A smooth gradient profile on two planes, the second ten times larger.
    torch.manual_seed(8)
    profile = torch.exp(-0.5 * ((torch.arange(9.0) - 4.0) / 1.5) ** 2)
    grad = torch.stack([profile, 10.0 * profile])[:, None, :].expand(2, 2, 9).clone()
    param = torch.nn.Parameter(torch.zeros(2, 2, 9))
    optimiser = LazyMomentAdamW([{'params': [param], 'shared_second_moment': True}], lr=1e-2)
    param.grad = grad.clone()
    optimiser.step()
    update = -param.detach()
    # Within a plane the update is proportional to the gradient (one scale
    # per plane), not flattened to its sign as per-cell Adam would do.
    ratio = update / grad
    for plane in ratio.view(4, 9):
        assert float(plane.std() / plane.mean()) < 1e-5
    # Adam's scale invariance holds per plane: the ten-times gradient gets
    # the same update.
    torch.testing.assert_close(update[0], update[1])
    # The per-cell step, for contrast, moves every cell of the profile by
    # about the learning rate.
    cellwise = torch.nn.Parameter(torch.zeros(2, 2, 9))
    LazyMomentAdamW([{'params': [cellwise], 'lazy_moments': True}], lr=1e-2)
    per_cell = LazyMomentAdamW([{'params': [cellwise], 'lazy_moments': True}], lr=1e-2)
    cellwise.grad = grad.clone()
    per_cell.step()
    torch.testing.assert_close(-cellwise.detach(), torch.full_like(cellwise, 1e-2), rtol=1e-4, atol=1e-6)


def test_shared_second_moment_equals_per_cell_for_uniform_gradients():
    # Every cell of a plane carrying the same gradient makes the mean second
    # moment equal to each cell's own, so the two denominators coincide.
    torch.manual_seed(9)
    generator = torch.Generator().manual_seed(10)
    shared = torch.nn.Parameter(torch.randn(2, 3, 4, 4))
    cellwise = torch.nn.Parameter(shared.detach().clone())
    a = LazyMomentAdamW([{'params': [shared], 'lazy_moments': True, 'shared_second_moment': True}], lr=1e-2)
    b = LazyMomentAdamW([{'params': [cellwise], 'lazy_moments': True}], lr=1e-2)
    for _ in range(5):
        per_plane = torch.randn(2, 3, 1, 1, generator=generator)
        shared.grad = per_plane.expand(2, 3, 4, 4).clone()
        cellwise.grad = shared.grad.clone()
        a.step()
        b.step()
        torch.testing.assert_close(shared, cellwise, rtol=1e-6, atol=1e-8)


def test_shared_second_moment_on_a_vector_is_one_plane():
    param = torch.nn.Parameter(torch.zeros(6))
    optimiser = LazyMomentAdamW([{'params': [param], 'shared_second_moment': True}], lr=0.1)
    param.grad = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    optimiser.step()
    # mean second moment over the three touched cells: (1 + 4 + 9) / 3 * (1 - beta2)
    mean_sq = (14.0 / 3.0) * 1e-3
    denom = (mean_sq ** 0.5) / (1e-3 ** 0.5) + 1e-8
    expected = -0.1 * torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]) / denom
    torch.testing.assert_close(param.detach(), expected, rtol=1e-5, atol=1e-7)


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
    across = 'optimizer_flow_grad_smoothing_across_sigma_voxels'
    assert fields[across]['type'] == 'number'
    assert fields[across]['runtime_impact'] == 'run_boundary'
    assert defaults[across] == 0.0
    shared = 'optimizer_flow_shared_second_moment'
    assert fields[shared]['type'] == 'boolean'
    assert fields[shared]['runtime_impact'] == 'run_boundary'
    assert defaults[shared] is False
    assert 'description' in fields[across] and 'description' in fields[shared]
    # Both postdate durable checkpoints: a checkpoint without them loads as
    # if they were off.
    from config import BACKFILLABLE_CONFIG_DEFAULTS
    assert BACKFILLABLE_CONFIG_DEFAULTS[across] == 0.0
    assert BACKFILLABLE_CONFIG_DEFAULTS[shared] is False


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
@pytest.mark.parametrize('sigma, across', [
    (0.7, 0.0), (1.6, 0.0), (3.0, 0.0), (0.0, 0.8), (0.0, 1.5), (1.6, 0.6), (2.5, 1.2)])
def test_fused_cylindrical_blur_matches_conv_reference(sigma, across, monkeypatch):
    torch.manual_seed(6)
    num_phi, offsets = _cylinder_tables(9)
    grad = torch.randn(2, 3, 11, offsets[-1], device='cuda')
    num_phi_t = torch.tensor(num_phi, device='cuda')
    offsets_t = torch.tensor(offsets, device='cuda')
    fused = flow_grad_smoothing.smooth_cylindrical_(
        grad.clone(), num_phi_t, offsets_t, sigma, across)
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '0')
    reference = flow_grad_smoothing.smooth_cylindrical_(
        grad.clone(), num_phi, offsets, sigma, across)
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
