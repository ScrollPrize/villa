"""Opt-in FIT_SPIRAL_SEP_UPSAMPLE and FIT_SPIRAL_GRAPH_RK4 paths.

Covers the separable trilinear upsample against F.interpolate (forward and
gradient, including non-integer ratios), and graph_rk4.RK4GraphIntegrate
against _RK4SparseFlowIntegrate: identical end points and point gradients,
and the same field-gradient accumulator up to atomic-add ordering.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flow_fields


@pytest.mark.parametrize('lr_shape, hr_shape', [
    ((4, 5, 6), (8, 10, 12)),
    ((3, 7, 5), (10, 16, 11)),
    ((6, 4, 9), (13, 4, 20)),
])
def test_sep_trilinear_matches_interpolate(lr_shape, hr_shape):
    torch.manual_seed(0)
    x = torch.randn(2, 3, *lr_shape, dtype=torch.float64, requires_grad=True)
    g = torch.randn(2, 3, *hr_shape, dtype=torch.float64)

    ref = F.interpolate(x, size=hr_shape, mode='trilinear', align_corners=False)
    (ref_grad,) = torch.autograd.grad(ref, x, g)
    out = flow_fields._sep_trilinear(x, hr_shape)
    (out_grad,) = torch.autograd.grad(out, x, g)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, rtol=0, atol=1e-12)
    assert torch.allclose(out_grad, ref_grad, rtol=0, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA/HIP')
@pytest.mark.parametrize('n_points', [700, 2500])
def test_graph_rk4_matches_eager(n_points):
    import graph_rk4

    torch.manual_seed(0)
    device = torch.device('cuda')
    shape = (3, 12, 14, 16)
    low = 0.05 * torch.randn(1, *shape, device=device)
    high = 0.05 * torch.randn(1, *shape, device=device)
    h, n_steps = 0.25, 3
    y0 = 0.2 + 0.6 * torch.rand(n_points, 3, device=device)
    grad_out = torch.randn(n_points, 3, device=device)

    def run(fn, *args):
        y = y0.clone().requires_grad_(True)
        out = fn(y, *args)
        out.backward(grad_out)
        torch.cuda.synchronize()
        return out.detach(), y.grad

    ref_acc = torch.zeros_like(high[0])
    ref_y, ref_g = run(flow_fields._RK4SparseFlowIntegrate.apply,
                       low[0], high[0], 1.0, ref_acc, h, n_steps)

    # The graphs bake in the accumulator pointer, so reuse one buffer. Run
    # twice: the first call captures, the second replays the cached graphs.
    acc = torch.zeros_like(high[0])
    for _ in range(2):
        acc.zero_()
        out_y, out_g = run(graph_rk4.RK4GraphIntegrate.apply,
                           low[0], high[0], acc, h, n_steps)
        out_acc = acc.clone()
        assert torch.equal(out_y, ref_y)
        assert torch.equal(out_g, ref_g)
        rel = (out_acc - ref_acc).abs().max() / ref_acc.abs().max()
        assert rel.item() <= 1e-5


@pytest.fixture
def graph_cache():
    import graph_rk4

    graph_rk4.clear()
    yield graph_rk4
    graph_rk4.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA/HIP')
@pytest.mark.parametrize('path', ['triton', 'sampler'])
def test_non_graph_paths_do_not_allocate_graph_buffer(monkeypatch, graph_cache, path):
    monkeypatch.setenv('FIT_SPIRAL_GRAPH_RK4', '1')
    monkeypatch.setenv('FIT_SPIRAL_DIRECT_LR', '0')
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '1')
    field = flow_fields.CartesianFlowField((12, 14, 16), spatial_scale_factor=2).cuda()
    if path == 'triton':
        if not flow_fields.flow_triton.rk4_triton_available(*field.flows):
            pytest.skip('needs Triton')
        field.get_integrator()
    else:
        field.get_sampler()
    assert not hasattr(field, '_graph_low_buf')
    assert not graph_cache._FWD and not graph_cache._BWD


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA/HIP')
def test_frozen_fields_do_not_cache_graphs(monkeypatch, graph_cache):
    monkeypatch.setenv('FIT_SPIRAL_GRAPH_RK4', '1')
    monkeypatch.setenv('FIT_SPIRAL_DIRECT_LR', '0')
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '0')
    field = flow_fields.CartesianFlowField((12, 14, 16), spatial_scale_factor=2).cuda()
    field.requires_grad_(False)
    for _ in range(3):
        points = torch.rand(17, 3, device='cuda', requires_grad=True)
        out = field.get_integrator()(points, 0.25, 3)
        out.sum().backward()
        # A zero velocity field is the identity, including its point gradient.
        assert torch.equal(out, points)
        assert torch.equal(points.grad, torch.ones_like(points))
        assert not hasattr(field, '_graph_low_buf')
        assert not graph_cache._FWD and not graph_cache._BWD


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA/HIP')
@pytest.mark.parametrize('reverse', [False, True])
def test_training_graph_reuses_updated_fields(monkeypatch, graph_cache, reverse):
    monkeypatch.setenv('FIT_SPIRAL_DIRECT_LR', '0')
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '0')
    torch.manual_seed(1)
    field = flow_fields.CartesianFlowField(
        (12, 14, 16), spatial_scale_factor=2, num_stages=2).cuda()
    reference = flow_fields.CartesianFlowField(
        (12, 14, 16), spatial_scale_factor=2, num_stages=2).cuda()
    points = 0.2 + 0.6 * torch.rand(17, 3, device='cuda')
    weights = torch.randn_like(points)
    h = -0.25 if reverse else 0.25
    buffer_ptr = None
    for _ in range(3):
        with torch.no_grad():
            for param, ref in zip(field.flows, reference.flows):
                param.add_(0.01 * torch.randn_like(param))
                ref.copy_(param)
        results = []
        for model, enabled in [(reference, '0'), (field, '1')]:
            monkeypatch.setenv('FIT_SPIRAL_GRAPH_RK4', enabled)
            model.zero_grad(set_to_none=True)
            y = points.clone().requires_grad_(True)
            out = model.get_integrator()(y, h, 3, reverse=reverse)
            out.backward(weights)
            model.apply_accumulated_field_grad()
            results.append((out.detach(), y.grad))
        assert torch.equal(results[0][0], results[1][0])
        assert torch.equal(results[0][1], results[1][1])
        for param, ref in zip(field.flows, reference.flows):
            torch.testing.assert_close(param.grad, ref.grad, rtol=1e-5, atol=1e-6)
        current_ptr = field._graph_low_buf.data_ptr()
        if buffer_ptr is not None:
            assert current_ptr == buffer_ptr
        buffer_ptr = current_ptr
        assert len(graph_cache._FWD) == len(graph_cache._BWD) == 2
