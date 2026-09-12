"""Flow stages as the slabs of one flow field.

The N flow stages are the slabs of the flow lattices' leading axis,
integrated in sequence by one IntegratedFlowDiffeomorphism (slab 0 first in
the spiral->slice direction; the inverse runs the slabs backwards in reverse
order). Covers: the slab walk equals composing N single-stage fields (eager
path, both lattice types, both directions, field and point gradients); the
fused CUDA kernels match for N > 1 in both directions; and the migration of
checkpoints written in the per-stage module layout (model state, optimiser
moments) plus the retired time-axis config key.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flow_triton
from checkpoint_migrations import (merge_flow_stage_lattices,
                                   merge_flow_stage_state)
from config import RETIRED_CONFIG_KEYS, Config, durable_config
from flow_fields import CartesianFlowField, CylindricalFlowField
from transforms import SpiralAndTransform
import update_checkpoint


N_STEPS = 3
H = 1.0 / N_STEPS


def _make_field(kind, num_stages, device='cpu', seed=7, **kwargs):
    torch.manual_seed(seed)
    if kind == 'cartesian':
        field = CartesianFlowField(
            torch.tensor([12, 16, 16]), spatial_scale_factor=4,
            num_stages=num_stages, **kwargs)
    else:
        field = CylindricalFlowField(
            (12, 16, 16), spatial_scale_factor=4, num_stages=num_stages)
    field = field.to(device)
    with torch.no_grad():
        field.flows[0].uniform_(-0.03, 0.03)
        field.flows[1].uniform_(-0.012, 0.012)
    return field


def _single_stage_copies(field, kind, **kwargs):
    """One single-stage field per slab, holding that slab's lattices."""
    copies = []
    for slab in range(field.num_stages):
        single = _make_field(kind, 1, device=field.flows[0].device, **kwargs)
        with torch.no_grad():
            for level in range(2):
                single.flows[level].copy_(field.flows[level][slab:slab + 1])
        copies.append(single)
    return copies


def _points(n, device='cpu', seed=41):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(n, 3, generator=generator, device=device) * 1.2 - 0.1


@pytest.mark.parametrize('kind', ['cartesian', 'cylindrical'])
@pytest.mark.parametrize('reverse', [False, True])
def test_slab_walk_equals_composition_of_single_stage_fields(kind, reverse):
    merged = _make_field(kind, 3)
    singles = _single_stage_copies(merged, kind)
    h = -H if reverse else H
    order = list(range(3))[::-1] if reverse else list(range(3))

    points = _points(53).requires_grad_(True)
    reference_points = points.detach().clone().requires_grad_(True)

    out = merged.get_integrator()(points, h, N_STEPS, reverse=reverse)
    y = reference_points
    for slab in order:
        y = singles[slab].get_integrator()(y, h, N_STEPS)
    upstream = torch.randn(points.shape, generator=torch.Generator().manual_seed(3))
    out.backward(upstream)
    y.backward(upstream)
    merged.apply_accumulated_field_grad()
    for single in singles:
        single.apply_accumulated_field_grad()

    torch.testing.assert_close(out, y)
    torch.testing.assert_close(points.grad, reference_points.grad)
    for level in range(2):
        assert merged.flows[level].grad.shape == merged.flows[level].shape
        for slab, single in enumerate(singles):
            torch.testing.assert_close(
                merged.flows[level].grad[slab], single.flows[level].grad[0])


def test_no_grad_walk_matches_grad_walk():
    merged = _make_field('cartesian', 2)
    points = _points(29)
    with torch.no_grad():
        plain = merged.get_integrator()(points, -H, N_STEPS, reverse=True)
    traced = merged.get_integrator()(
        points.clone().requires_grad_(True), -H, N_STEPS, reverse=True)
    torch.testing.assert_close(plain, traced.detach())


def test_reverse_walk_inverts_forward_walk_approximately():
    # Not exact (RK4 forward/backward inconsistency), but the slab order must
    # be reversed for the round trip to close at all at this flow amplitude.
    merged = _make_field('cartesian', 3)
    points = _points(200)
    with torch.no_grad():
        integrate = merged.get_integrator()
        forward = integrate(points, H, N_STEPS)
        back = integrate(forward, -H, N_STEPS, reverse=True)
        wrong_order = integrate(forward, -H, N_STEPS, reverse=False)
    torch.testing.assert_close(back, points, atol=2e-3, rtol=0.)
    assert (wrong_order - points).abs().max() > 1e-2


def test_model_forward_and_inverse_use_one_diffeomorphism():
    config = {
        'model_initial_dr_per_winding': 16.,
        'model_flow_voxel_resolution': 8,
        'model_flow_field_type': 'cartesian',
        'model_num_flow_stages': 2,
        'model_linear_z_resolution': 48,
        'model_gap_expander_logit_resolution': 24,
        'model_gap_expander_num_windings': 6,
        'model_gap_expander_capacity_windings': 6,
        'model_gap_expander_min_gap': 1.0,
        'model_gap_expander_softplus_bias': 4.0,
        'model_gap_expander_lr_scale': 0.3,
        'output_first_winding': 1,
    }
    torch.manual_seed(0)
    umbilicus = torch.zeros([5, 3])
    umbilicus[:, 0] = torch.linspace(0., 192., 5)
    model = SpiralAndTransform(
        flow_integration_steps=3, flow_integration_solver='rk4',
        flow_min_corner_zyx=torch.tensor([0, -96, -96]),
        flow_max_corner_zyx=torch.tensor([192, 96, 96]),
        umbilicus_zyx=umbilicus, config=config)
    assert [flow.shape[0] for flow in model.flow_field.flows] == [2, 2]
    with torch.no_grad():
        for flow in model.flow_field.flows:
            flow.normal_(std=1e-3)
    transform = model.get_slice_to_spiral_transform()
    # Compose([gap, diffeo, linear, umbilicus]).inv: exactly one diffeomorphism.
    parts = getattr(transform, 'parts', None)
    if parts is None:
        parts = transform._inv.parts
    from transforms import IntegratedFlowDiffeomorphism
    assert sum(isinstance(getattr(p, '_inv', p), IntegratedFlowDiffeomorphism)
               or isinstance(p, IntegratedFlowDiffeomorphism) for p in parts) == 1
    points = torch.stack([
        torch.empty([128]).uniform_(10., 180.),
        torch.empty([128]).uniform_(-80., 80.),
        torch.empty([128]).uniform_(-80., 80.),
    ], dim=-1)
    with torch.no_grad():
        round_trip = transform.inv(transform(points))
    torch.testing.assert_close(round_trip, points, atol=0.5, rtol=0.)


cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not flow_triton._HAS_TRITON,
    reason='requires CUDA and Triton')


def _eager_reference(field, points, h, n_steps, reverse, monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '0')
    try:
        return field.get_integrator()(points, h, n_steps, reverse=reverse)
    finally:
        monkeypatch.setenv('FIT_SPIRAL_TRITON', '1')


@cuda
@pytest.mark.parametrize('kind', ['cartesian', 'cylindrical'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('coalesce', ['0', '1'])
def test_fused_multi_slab_matches_eager(monkeypatch, kind, reverse, coalesce):
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '1')
    monkeypatch.setenv('FIT_SPIRAL_RK4_COALESCE', coalesce)
    monkeypatch.delenv('FIT_SPIRAL_DIRECT_LR', raising=False)
    fused = _make_field(kind, 3, device='cuda', seed=23, **(
        {'direct_lr': False} if kind == 'cartesian' else {}))
    eager = _make_field(kind, 3, device='cuda', seed=23, **(
        {'direct_lr': False} if kind == 'cartesian' else {}))
    eager.load_state_dict(fused.state_dict())
    h = -H if reverse else H
    points = _points(97, device='cuda').requires_grad_(True)
    reference_points = points.detach().clone().requires_grad_(True)
    upstream = torch.randn(
        points.shape, generator=torch.Generator(device='cuda').manual_seed(5),
        device='cuda')

    output = fused.get_integrator()(points, h, N_STEPS, reverse=reverse)
    reference = _eager_reference(
        eager, reference_points, h, N_STEPS, reverse, monkeypatch)
    output.backward(upstream)
    reference.backward(upstream)
    fused.apply_accumulated_field_grad()
    eager.apply_accumulated_field_grad()

    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        points.grad, reference_points.grad, rtol=2e-4, atol=2e-5)
    for level in range(2):
        torch.testing.assert_close(
            fused.flows[level].grad, eager.flows[level].grad,
            rtol=2e-4, atol=2e-5)
        # Every slab received its own gradient; none is a copy of another.
        grads = fused.flows[level].grad
        assert grads.shape[0] == 3
        assert not torch.equal(grads[0], grads[1])


@cuda
@pytest.mark.parametrize('reverse', [False, True])
def test_fused_direct_multi_slab_equals_sequential_single_slabs(monkeypatch, reverse):
    # Direct-LR sampling differs from the upsampled interpolant, so its
    # reference is the same kernel run one slab at a time.
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '1')
    monkeypatch.setenv('FIT_SPIRAL_DIRECT_LR', '1')
    merged = _make_field('cartesian', 3, device='cuda', seed=29, direct_lr=True)
    singles = _single_stage_copies(merged, 'cartesian', direct_lr=True)
    h = -H if reverse else H
    order = list(range(3))[::-1] if reverse else list(range(3))
    points = _points(131, device='cuda').requires_grad_(True)
    reference_points = points.detach().clone().requires_grad_(True)
    upstream = torch.randn(
        points.shape, generator=torch.Generator(device='cuda').manual_seed(9),
        device='cuda')

    out = merged.get_integrator()(points, h, N_STEPS, reverse=reverse)
    y = reference_points
    for slab in order:
        y = singles[slab].get_integrator()(y, h, N_STEPS)
    out.backward(upstream)
    y.backward(upstream)
    merged.apply_accumulated_field_grad()
    for single in singles:
        single.apply_accumulated_field_grad()

    torch.testing.assert_close(out, y, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(points.grad, reference_points.grad, rtol=1e-5, atol=1e-6)
    for level in range(2):
        for slab, single in enumerate(singles):
            torch.testing.assert_close(
                merged.flows[level].grad[slab], single.flows[level].grad[0],
                rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------- migration

def _old_layout_state(model):
    """The per-stage module layout a pre-slab checkpoint stored."""
    state = {key: value.clone() for key, value in model.state_dict().items()}
    num_stages = state['flow_field.flows.0'].shape[0]
    old = {}
    for key, value in state.items():
        if key.startswith('flow_field.'):
            suffix = key[len('flow_field.'):]
            if suffix.startswith('flows.'):
                old[key] = value[:1].clone()
                for stage in range(1, num_stages):
                    old[f'extra_flow_fields.{stage - 1}.{suffix}'] = (
                        value[stage:stage + 1].clone())
            else:
                # Cylindrical ring tables were buffers of every stage module.
                old[key] = value
                for stage in range(1, num_stages):
                    old[f'extra_flow_fields.{stage - 1}.{suffix}'] = value.clone()
        else:
            old[key] = value
    return state, old


def _small_model(flow_field_type, num_stages):
    config = Config().as_dict()
    config.update({
        'model_flow_field_type': flow_field_type,
        'model_num_flow_stages': num_stages,
        'model_gap_expander_capacity_windings': 8,
        'model_gap_expander_num_windings': 8,
        'model_flow_voxel_resolution': 16,
        'model_linear_z_resolution': 8,
    })
    torch.manual_seed(11)
    model = SpiralAndTransform(
        flow_integration_steps=2, flow_integration_solver='rk4',
        flow_min_corner_zyx=torch.tensor([0, -48, -48]),
        flow_max_corner_zyx=torch.tensor([48, 48, 48]),
        umbilicus_zyx=torch.zeros(48, 3), config=config)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel() > 1:
                parameter.normal_(std=0.01)
    return model, config


def _fit_spiral_groups(model):
    low = [model.flow_field.flows[0]]
    high = [model.flow_field.flows[1]]
    gap = list(model.gap_expander_params.parameters())
    linear = [model.linear_logits]
    grouped = {id(p) for p in low + high + gap + linear}
    other = [p for p in model.parameters() if id(p) not in grouped]
    return [
        {'params': other, 'weight_decay': 0.0},
        {'params': linear, 'weight_decay': 0.0},
        {'params': gap, 'weight_decay': 0.01},
        {'params': low, 'weight_decay': 0.0},
        {'params': high, 'weight_decay': 0.0},
    ]


@pytest.mark.parametrize('flow_field_type', ['cartesian', 'cylindrical'])
def test_merge_flow_stage_state_folds_stage_modules(flow_field_type):
    model, _ = _small_model(flow_field_type, 3)
    new_state, old_state = _old_layout_state(model)
    assert any(key.startswith('extra_flow_fields.') for key in old_state)
    merged = merge_flow_stage_state(old_state)
    assert set(merged) == set(new_state)
    for key in new_state:
        assert torch.equal(merged[key], new_state[key]), key
    # Already in the slab layout: returned as is.
    assert merge_flow_stage_state(new_state) is new_state
    fresh, _ = _small_model(flow_field_type, 3)
    fresh.load_state_dict(merged)


def test_merge_flow_stage_lattices_is_a_noop_for_single_stage_checkpoints():
    model, config = _small_model('cartesian', 1)
    checkpoint = {'spiral_and_transform': model.state_dict(),
                  'cfg': {**config, 'model_num_flow_timesteps': 1}}
    assert merge_flow_stage_lattices(checkpoint) is checkpoint


def test_merge_flow_stage_lattices_refuses_time_varying_flows():
    model, config = _small_model('cartesian', 1)
    checkpoint = {'spiral_and_transform': model.state_dict(),
                  'cfg': {**config, 'model_num_flow_timesteps': 2}}
    with pytest.raises(ValueError, match='time-varying'):
        merge_flow_stage_lattices(checkpoint)


def test_merge_flow_stage_lattices_migrates_optimizer():
    num_stages = 2
    model, config = _small_model('cartesian', num_stages)
    new_state, old_state = _old_layout_state(model)

    # An optimiser over the OLD layout: one parameter per stage in each of
    # the two flow groups, with real Adam moments.
    old_model_low = [torch.nn.Parameter(old_state['flow_field.flows.0'].clone()),
                     torch.nn.Parameter(old_state['extra_flow_fields.0.flows.0'].clone())]
    old_model_high = [torch.nn.Parameter(old_state['flow_field.flows.1'].clone()),
                      torch.nn.Parameter(old_state['extra_flow_fields.0.flows.1'].clone())]
    groups = _fit_spiral_groups(model)
    groups[3] = {'params': old_model_low, 'weight_decay': 0.0}
    groups[4] = {'params': old_model_high, 'weight_decay': 0.0}
    old_optimiser = torch.optim.AdamW(groups, lr=1e-3)
    torch.manual_seed(2)
    for group in old_optimiser.param_groups:
        for param in group['params']:
            param.grad = torch.randn_like(param)
    old_optimiser.step()
    expected_linear = old_optimiser.state[model.linear_logits]['exp_avg'].clone()
    expected_low = torch.cat(
        [old_optimiser.state[p]['exp_avg'] for p in old_model_low], dim=0)
    expected_high_sq = torch.cat(
        [old_optimiser.state[p]['exp_avg_sq'] for p in old_model_high], dim=0)

    checkpoint = {
        'spiral_and_transform': old_state,
        'optimiser': old_optimiser.state_dict(),
        'cfg': {**config, 'model_num_flow_timesteps': 1},
    }
    migrated = merge_flow_stage_lattices(checkpoint)
    assert migrated is not checkpoint
    # The input is not mutated.
    assert 'extra_flow_fields.0.flows.0' in checkpoint['spiral_and_transform']
    assert len(checkpoint['optimiser']['param_groups'][3]['params']) == num_stages

    merged_groups = migrated['optimiser']['param_groups']
    assert [g['params'] for g in merged_groups] == [
        list(range(len(groups[0]['params']))),
        [len(groups[0]['params'])],
        [len(groups[0]['params']) + 1],
        [len(groups[0]['params']) + 2],
        [len(groups[0]['params']) + 3],
    ]
    fresh, _ = _small_model('cartesian', num_stages)
    fresh.load_state_dict(migrated['spiral_and_transform'])
    for key in new_state:
        assert torch.equal(fresh.state_dict()[key], new_state[key]), key
    new_optimiser = torch.optim.AdamW(_fit_spiral_groups(fresh), lr=1e-3)
    new_optimiser.load_state_dict(migrated['optimiser'])
    assert torch.equal(
        new_optimiser.state[fresh.linear_logits]['exp_avg'], expected_linear)
    assert torch.equal(
        new_optimiser.state[fresh.flow_field.flows[0]]['exp_avg'], expected_low)
    assert torch.equal(
        new_optimiser.state[fresh.flow_field.flows[1]]['exp_avg_sq'],
        expected_high_sq)
    assert new_optimiser.state[fresh.flow_field.flows[0]]['step'] == \
        old_optimiser.state[old_model_low[0]]['step']

    # Another pass finds nothing left to do.
    assert merge_flow_stage_lattices(migrated) is migrated


def test_retired_time_axis_key_is_dropped_by_config_migration():
    assert 'model_num_flow_timesteps' in RETIRED_CONFIG_KEYS
    with pytest.raises(ValueError, match='Unknown Spiral config keys'):
        Config({'model_num_flow_timesteps': 1})
    source = {**durable_config(Config().as_dict()), 'model_num_flow_timesteps': 1}
    migrated, renamed, removed, added = update_checkpoint.migrate_config(source)
    assert 'model_num_flow_timesteps' not in migrated
    assert removed == ['model_num_flow_timesteps']
    legacy = {**durable_config(Config().as_dict()), 'num_flow_timesteps': 1}
    _, _, removed, _ = update_checkpoint.migrate_config(legacy)
    assert removed == ['num_flow_timesteps']
