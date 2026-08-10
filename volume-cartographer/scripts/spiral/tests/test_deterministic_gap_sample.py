import contextlib

import pytest
import torch
import torch.nn.functional as F

import transforms
from transforms import (
    GapExpanderParams,
    GapExpandingTransform,
    _bilinear_sample_2d_border,
)


@contextlib.contextmanager
def deterministic_algorithms(enabled=True):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous, warn_only=previous_warn_only)


def grid_sample_reference(values, x, y):
    grid = torch.stack([x, y], dim=-1).reshape(1, -1, 1, 2)
    return F.grid_sample(
        values,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    ).reshape_as(x)


def test_bilinear_border_forward_matches_grid_sample_on_cpu():
    values = torch.arange(35, dtype=torch.float64).view(1, 1, 5, 7)
    x = torch.tensor([
        [-2.0, -1.0, -0.75, 0.0, 0.6, 1.0, 2.0],
        [float('nan'), float('-inf'), -0.2, 0.2, 0.9, float('inf'), 1.2],
    ], dtype=torch.float64)
    y = torch.tensor([
        [-2.0, -1.0, -0.4, 0.0, 0.7, 1.0, 2.0],
        [float('nan'), float('-inf'), -0.8, 0.3, 0.95, float('inf'), 1.1],
    ], dtype=torch.float64)
    actual = _bilinear_sample_2d_border(values, x, y)
    expected = grid_sample_reference(values, x, y)
    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=5e-15)


def test_gap_transform_keeps_grid_sample_path_on_cpu(monkeypatch):
    params = GapExpanderParams(
        resolution=4.0,
        min_z=0.0,
        max_z=32.0,
        num_windings=7,
        dr_per_winding=12.0,
    )
    transform = GapExpandingTransform(
        params=params,
        dr_per_winding=torch.tensor(12.0),
        min_z=0.0,
        max_z=32.0,
        gap_expander_lr_scale=1.0,
    )
    theta = torch.linspace(0.0, 2 * torch.pi, 17)
    z = torch.linspace(0.0, 32.0, 17)

    def unexpected_custom_sampler(*_args, **_kwargs):
        raise AssertionError('the deterministic fallback is CUDA-only')

    monkeypatch.setattr(
        transforms, '_bilinear_sample_2d_border', unexpected_custom_sampler)
    with deterministic_algorithms():
        sampled = transform.get_logits_by_winding(theta, z)
    assert sampled.shape == (17, 6)
    assert torch.isfinite(sampled).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_values_and_interior_gradients_match_grid_sample():
    generator = torch.Generator(device='cuda').manual_seed(9917)
    x_data = torch.rand(9, 13, device='cuda', generator=generator) * 1.8 - 0.9
    y_data = torch.rand(9, 13, device='cuda', generator=generator) * 1.8 - 0.9
    loss_weights = torch.randn(9, 13, device='cuda', generator=generator)

    reference_values = torch.randn(
        1, 1, 11, 17, device='cuda', generator=generator, requires_grad=True)
    reference_x = x_data.detach().clone().requires_grad_(True)
    reference_y = y_data.detach().clone().requires_grad_(True)
    reference = grid_sample_reference(reference_values, reference_x, reference_y)
    reference_grads = torch.autograd.grad(
        (reference * loss_weights).sum(),
        (reference_values, reference_x, reference_y),
    )

    actual_values = reference_values.detach().clone().requires_grad_(True)
    actual_x = x_data.detach().clone().requires_grad_(True)
    actual_y = y_data.detach().clone().requires_grad_(True)
    actual = _bilinear_sample_2d_border(actual_values, actual_x, actual_y)
    actual_grads = torch.autograd.grad(
        (actual * loss_weights).sum(),
        (actual_values, actual_x, actual_y),
    )

    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-7)
    for got, wanted in zip(actual_grads, reference_grads):
        torch.testing.assert_close(got, wanted, rtol=3e-5, atol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_border_coordinate_gradients_match_grid_sample():
    values_data = torch.arange(35, device='cuda', dtype=torch.float32).view(1, 1, 5, 7)
    x_data = torch.tensor(
        [-1.2, -1.0, -0.999, 0.0, 0.999, 1.0, 1.2], device='cuda')
    y_data = torch.tensor(
        [-1.2, -1.0, -0.999, 0.0, 0.999, 1.0, 1.2], device='cuda')

    reference_values = values_data.clone().requires_grad_(True)
    reference_x = x_data.clone().requires_grad_(True)
    reference_y = y_data.clone().requires_grad_(True)
    reference = grid_sample_reference(reference_values, reference_x, reference_y)
    reference_grads = torch.autograd.grad(
        reference.sum(), (reference_values, reference_x, reference_y))

    actual_values = values_data.clone().requires_grad_(True)
    actual_x = x_data.clone().requires_grad_(True)
    actual_y = y_data.clone().requires_grad_(True)
    actual = _bilinear_sample_2d_border(actual_values, actual_x, actual_y)
    actual_grads = torch.autograd.grad(
        actual.sum(), (actual_values, actual_x, actual_y))

    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
    for got, wanted in zip(actual_grads, reference_grads):
        torch.testing.assert_close(got, wanted, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_backward_is_bitwise_repeatable_under_hard_determinism():
    generator = torch.Generator().manual_seed(4271)
    values_cpu = torch.randn(1, 1, 12, 19, generator=generator)
    x_cpu = torch.rand(17, 23, generator=generator) * 2.4 - 1.2
    y_cpu = torch.rand(17, 23, generator=generator) * 2.4 - 1.2
    weights_cpu = torch.randn(17, 23, generator=generator)

    def run_once():
        values = values_cpu.cuda().requires_grad_(True)
        x = x_cpu.cuda().requires_grad_(True)
        y = y_cpu.cuda().requires_grad_(True)
        result = _bilinear_sample_2d_border(values, x, y)
        grads = torch.autograd.grad(
            (result * weights_cpu.cuda()).sum(), (values, x, y))
        return (result.detach().cpu(), *(grad.detach().cpu() for grad in grads))

    with deterministic_algorithms():
        first = run_once()
        second = run_once()
    for got, wanted in zip(second, first):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gap_transform_uses_deterministic_sampler_and_preserves_pin():
    device = torch.device('cuda')
    params = GapExpanderParams(
        resolution=4.0,
        min_z=0.0,
        max_z=32.0,
        num_windings=7,
        dr_per_winding=12.0,
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(717)
    with torch.no_grad():
        params.logits.copy_(torch.randn(
            params.logits.shape, device=device, generator=generator) * 1e-4)
    transform = GapExpandingTransform(
        params=params,
        dr_per_winding=torch.tensor(12.0, device=device),
        min_z=0.0,
        max_z=32.0,
        gap_expander_lr_scale=1.0,
    )
    theta = (torch.rand(31, device=device, generator=generator) * 2 * torch.pi).requires_grad_(True)
    z = (torch.rand(31, device=device, generator=generator) * 32).requires_grad_(True)

    with deterministic_algorithms():
        sampled = transform.get_logits_by_winding(theta, z)
        sampled.square().sum().backward()
    assert torch.isfinite(sampled).all()
    assert torch.isfinite(params.logits.grad).all()
    assert torch.count_nonzero(params.logits.grad[..., 0]) == 0
