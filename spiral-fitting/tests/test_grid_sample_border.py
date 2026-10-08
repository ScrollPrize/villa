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
