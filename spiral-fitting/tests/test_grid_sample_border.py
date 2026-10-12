import pytest
import torch
import torch.nn.functional as F

from geom_utils import gather_grid_sample_border, grid_sample_border


def _reference(input, grid):
    return F.grid_sample(input, grid, mode='bilinear', padding_mode='border',
                         align_corners=True)


def _case(spatial, points, seed):
    generator = torch.Generator().manual_seed(seed)
    input = torch.randn((2, 3, *spatial), generator=generator, dtype=torch.float64)
    # Coordinates past [-1, 1] exercise the border clamp.
    grid = torch.rand((2, *points, len(spatial)), generator=generator,
                      dtype=torch.float64) * 2.6 - 1.3
    return input, grid


def _values_and_grads(fn, input, grid):
    input = input.clone().requires_grad_()
    grid = grid.clone().requires_grad_()
    out = fn(input, grid)
    weights = torch.linspace(-1, 1, out.numel(), dtype=out.dtype, device=out.device)
    (out * weights.view(out.shape)).sum().backward()
    return out.detach(), input.grad, grid.grad


@pytest.mark.parametrize('spatial,points', [
    ((5, 7), (4, 6)),
    ((1, 9), (3, 5)),
    ((4, 5, 6), (3, 2, 5)),
    ((3, 1, 4), (2, 3, 1)),
])
def test_gathers_match_grid_sample(spatial, points):
    input, grid = _case(spatial, points, seed=len(spatial) * 10 + points[0])
    expected = _values_and_grads(_reference, input, grid)
    actual = _values_and_grads(gather_grid_sample_border, input, grid)
    for want, got in zip(expected, actual):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_off_mps_it_is_grid_sample(monkeypatch):
    input, grid = _case((5, 7), (4, 6), seed=0)
    monkeypatch.setattr(F, 'grid_sample', lambda *a, **k: 'native')
    assert grid_sample_border(input.requires_grad_(), grid) == 'native'


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs MPS')
@pytest.mark.parametrize('spatial,points', [((5, 7), (4, 6)), ((4, 5, 6), (3, 2, 5))])
def test_mps_backward_matches_cpu_grid_sample(spatial, points):
    input, grid = _case(spatial, points, seed=7)
    input, grid = input.float(), grid.float()
    expected = _values_and_grads(_reference, input, grid)
    actual = _values_and_grads(grid_sample_border, input.to('mps'), grid.to('mps'))
    for want, got in zip(expected, actual):
        torch.testing.assert_close(got.cpu(), want, rtol=1e-4, atol=1e-5)


def _with_nan(input, grid):
    """``grid`` with NaN in some coordinates, and what CUDA's grid sampler
    does with them: the forward pass samples index 0 along the NaN axis
    (normalised -1 with align_corners=True), and the point is left out of
    the backward pass."""
    grid = grid.clone()
    flat = grid.view(-1, grid.shape[-1])
    flat[1, 0] = float('nan')
    flat[4, -1] = float('nan')
    flat[5, :] = float('nan')
    nan_point = torch.isnan(grid).any(-1)
    return grid, grid.nan_to_num(nan=-1.0), nan_point


def _nan_case_grads(fn, input, grid, nan_point):
    input = input.clone().requires_grad_()
    grid = grid.clone().requires_grad_()
    out = fn(input, grid)
    weights = torch.linspace(-1, 1, out.numel(), dtype=out.dtype, device=out.device).view(out.shape)
    weights = weights * ~nan_point.to(out.device)[:, None]
    (out * weights).sum().backward()
    return input.grad, grid.grad


@pytest.mark.parametrize('spatial,points', [((5, 7), (4, 6)), ((4, 5, 6), (3, 2, 5))])
def test_nan_coordinates_behave_as_on_cuda(spatial, points):
    input, grid = _case(spatial, points, seed=3)
    grid, grid_at_index_0, nan_point = _with_nan(input, grid)
    with torch.no_grad():
        torch.testing.assert_close(gather_grid_sample_border(input, grid),
                                   _reference(input, grid_at_index_0), rtol=1e-10, atol=1e-10)
    # Full weights on the gathers: the NaN points must still add nothing.
    input_grad, grid_grad = _nan_case_grads(gather_grid_sample_border, input, grid,
                                            torch.zeros_like(nan_point))
    want_input_grad, want_grid_grad = _nan_case_grads(_reference, input, grid_at_index_0, nan_point)
    torch.testing.assert_close(input_grad, want_input_grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(grid_grad, want_grid_grad, rtol=1e-10, atol=1e-10)
    assert torch.all(grid_grad[nan_point] == 0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs MPS')
@pytest.mark.parametrize('spatial,points', [((5, 7), (4, 6)), ((4, 5, 6), (3, 2, 5))])
def test_mps_nan_coordinates_keep_gradients_finite(spatial, points):
    input, grid = _case(spatial, points, seed=5)
    grid, grid_at_index_0, nan_point = _with_nan(input, grid)
    input, grid, grid_at_index_0 = input.float(), grid.float(), grid_at_index_0.float()
    input_grad, grid_grad = _nan_case_grads(grid_sample_border, input.to('mps'), grid.to('mps'),
                                            torch.zeros_like(nan_point))
    want_input_grad, want_grid_grad = _nan_case_grads(_reference, input, grid_at_index_0, nan_point)
    torch.testing.assert_close(input_grad.cpu(), want_input_grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(grid_grad.cpu(), want_grid_grad, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs MPS')
@pytest.mark.parametrize('spatial,points', [((5, 7), (4, 6)), ((4, 5, 6), (3, 2, 5))])
@pytest.mark.parametrize('needs_grad', [False, True])
def test_mps_without_autograd_matches_cpu_grid_sample(spatial, points, needs_grad):
    # Preview export, satisfaction metrics and mesh save sample under no_grad,
    # so on MPS they take the native path (aten::grid_sampler_3d for 5-D).
    input, grid = _case(spatial, points, seed=11)
    input, grid = input.float(), grid.float()
    expected = _reference(input, grid)
    with torch.no_grad():
        actual = grid_sample_border(input.to('mps').requires_grad_(needs_grad), grid.to('mps'))
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
