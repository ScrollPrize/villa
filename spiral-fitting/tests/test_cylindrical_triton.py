import math
from unittest import mock

import pytest
import torch

import flow_triton
import transforms
from flow_fields import CylindricalFlowField


def _manual_rk4(sampler, points, h, n_steps):
    y = points
    for _ in range(n_steps):
        k1 = sampler(y)
        k2 = sampler(y + (h / 2) * k1)
        k3 = sampler(y + (h / 2) * k2)
        k4 = sampler(y + h * k3)
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def _make_flow(device='cpu', seed=7):
    torch.manual_seed(seed)
    flow = CylindricalFlowField((12, 16, 16), spatial_scale_factor=4).to(device)
    with torch.no_grad():
        flow.flows[0].uniform_(-0.015, 0.015)
        flow.flows[1].uniform_(-0.006, 0.006)
    return flow


def _edge_points(flow, device):
    eps = torch.finfo(torch.float32).eps
    nr_lr = flow._lr_num_phi.numel()
    nr_hr = flow._hr_num_phi.numel()
    rows = [
        [0.5, 0.5, 0.5],                         # exact axis
        [0.5, 0.5 + eps / 2, 0.5],               # inside axis epsilon
        [0.5, 0.5 + eps, 0.5],                   # outside axis epsilon
        [0.5, 0.5, 1.0],                         # positive-x seam
        [0.5, math.nextafter(0.5, 0.0), 1.0],    # negative side of seam
        [0.5, math.nextafter(0.5, 1.0), 1.0],    # positive side of seam
        [0.5, 0.5, 0.5 + 0.5 / (nr_lr - 1)],     # LR ring boundary
        [0.5, 0.5, 0.5 + 1.0 / (nr_hr - 1)],     # HR ring boundary
        [0.5, 0.5, 1.0],                         # unit radius clamp
        [0.5, 0.5, 1.1],                         # outside radius
        [0.0, 0.7, 0.7],                         # lower z border
        [1.0, 0.7, 0.7],                         # upper z border
        [-0.1, 0.7, 0.7],                        # below z
        [1.1, 0.7, 0.7],                         # above z
        [1.0 / (flow.flows[1].shape[2] - 1), 0.7, 0.7],  # HR z plane
        [1.0 / (flow.flows[0].shape[2] - 1), 0.7, 0.7],  # LR z plane
    ]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def test_cpu_stationary_integrator_falls_back_lazily(monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '0')
    actual = _make_flow()
    expected = _make_flow()
    expected.load_state_dict(actual.state_dict())
    points = torch.rand(31, 3, requires_grad=True)
    reference_points = points.detach().clone().requires_grad_(True)

    integrator = actual.get_time_invariant_integrator()
    assert actual._pending_field_graphs is None
    output = integrator(points, -0.1, 3)
    reference = _manual_rk4(expected.get_sampler(0.0), reference_points, -0.1, 3)
    output.square().sum().backward()
    reference.square().sum().backward()
    actual.apply_accumulated_field_grad()
    expected.apply_accumulated_field_grad()

    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(points.grad, reference_points.grad)
    torch.testing.assert_close(actual.flows[0].grad, expected.flows[0].grad)
    torch.testing.assert_close(actual.flows[1].grad, expected.flows[1].grad)


def test_temporally_varying_flow_bypasses_stationary_integrator(monkeypatch):
    class FakeFlow:
        num_flow_timesteps = 2

        def get_time_invariant_integrator(self):
            raise AssertionError('stationary integrator must not be requested')

        def get_sampler(self, _t):
            return torch.zeros_like

    calls = []

    def fake_odeint(func, y, ts, method):
        calls.append((func, ts, method))
        return torch.stack([y, y])

    monkeypatch.setattr(transforms, 'odeint', fake_odeint)
    transform = transforms.IntegratedFlowDiffeomorphism(
        FakeFlow(), torch.zeros(3), torch.ones(3), num_steps=3, solver='rk4')
    result = transform._call(torch.rand(5, 3))
    assert len(calls) == 1
    assert calls[0][2] == 'rk4'
    assert result.shape == (5, 3)


cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not flow_triton._HAS_TRITON,
    reason='requires CUDA and Triton')


@cuda
@pytest.mark.parametrize('n_steps', [1, 3])
@pytest.mark.parametrize('h', [-0.08, 0.08])
def test_fused_matches_eager_forward_and_adjoint(monkeypatch, n_steps, h):
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '1')
    fused = _make_flow('cuda', seed=23)
    eager = _make_flow('cuda', seed=23)
    eager.load_state_dict(fused.state_dict())
    generator = torch.Generator(device='cuda').manual_seed(41)
    random_points = torch.rand(97, 3, generator=generator, device='cuda') * 1.4 - 0.2
    points = torch.cat([random_points, _edge_points(fused, 'cuda')]).requires_grad_(True)
    reference_points = points.detach().clone().requires_grad_(True)
    upstream = torch.randn(points.shape, generator=generator, device='cuda')

    output = fused.get_time_invariant_integrator()(points, h, n_steps)
    reference = _manual_rk4(eager.get_sampler(0.0), reference_points, h, n_steps)
    output.backward(upstream)
    reference.backward(upstream)
    fused.apply_accumulated_field_grad()
    eager.apply_accumulated_field_grad()

    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(points.grad, reference_points.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(fused.flows[0].grad, eager.flows[0].grad,
                               rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(fused.flows[1].grad, eager.flows[1].grad,
                               rtol=2e-4, atol=2e-5)
    assert torch.count_nonzero(fused.flows[0].grad[0, :, :, 0]) == 0
    assert torch.count_nonzero(fused.flows[1].grad[0, :, :, 0]) == 0
    assert (fused.flows[0].grad.untyped_storage().data_ptr()
            == fused._lr_grad_acc.untyped_storage().data_ptr())
    assert (fused.flows[1].grad.untyped_storage().data_ptr()
            == fused._hr_grad_acc.untyped_storage().data_ptr())


@cuda
def test_two_fused_backwards_share_accumulators(monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '1')
    fused = _make_flow('cuda', seed=29)
    eager = _make_flow('cuda', seed=29)
    eager.load_state_dict(fused.state_dict())
    a = torch.rand(43, 3, device='cuda', requires_grad=True)
    b = torch.rand(37, 3, device='cuda', requires_grad=True)
    ar = a.detach().clone().requires_grad_(True)
    br = b.detach().clone().requires_grad_(True)
    fi = fused.get_time_invariant_integrator()
    es = eager.get_sampler(0.0)
    fi(a, 0.1, 3).square().mean().backward()
    fi(b, 0.1, 3).abs().mean().backward()
    _manual_rk4(es, ar, 0.1, 3).square().mean().backward()
    _manual_rk4(es, br, 0.1, 3).abs().mean().backward()
    fused.apply_accumulated_field_grad()
    eager.apply_accumulated_field_grad()
    torch.testing.assert_close(fused.flows[0].grad, eager.flows[0].grad,
                               rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(fused.flows[1].grad, eager.flows[1].grad,
                               rtol=2e-4, atol=2e-5)


@cuda
def test_empty_batch_and_no_grad_avoid_stage_storage(monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '1')
    flow = _make_flow('cuda')
    integrator = flow.get_time_invariant_integrator()
    empty = torch.empty(0, 3, device='cuda', requires_grad=True)
    assert integrator(empty, 0.1, 3).shape == (0, 3)

    points = torch.rand(11, 3, device='cuda')
    original = flow_triton._run_cylindrical_fwd
    seen = []

    def recording_run(*args):
        seen.append(args[-1])
        return original(*args)

    with mock.patch.object(flow_triton, '_run_cylindrical_fwd', recording_run):
        with torch.no_grad():
            result = flow.get_time_invariant_integrator()(points, 0.1, 3)
    assert result.grad_fn is None
    assert seen == [None]
