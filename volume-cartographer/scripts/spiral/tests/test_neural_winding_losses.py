import torch

from neural_winding_losses import (
    dense_metric_density_loss,
    dense_phase_observations,
    dense_phase_registration_loss,
)


def _straight_cache(length=21):
    x = torch.arange(length, dtype=torch.float32)
    points = torch.stack([torch.zeros_like(x), torch.zeros_like(x), x], dim=-1)[None]
    phase = (x / 10.0)[None]
    return points, phase


def test_dense_phase_gradient_points_toward_known_posterior_center():
    points, sample_phase = _straight_cache()
    probability = torch.exp(-0.5 * ((torch.arange(21) - 10.0) / 0.75) ** 2)[None]
    observation = dense_phase_observations(
        points,
        sample_phase,
        probability,
        torch.tensor([[1.0]]),
        window_windings=0.75,
        probability_power=2.0,
    )
    live = torch.tensor([[[0.0, 0.0, 8.0]]], requires_grad=True)
    loss, _ = dense_phase_registration_loss(
        live, observation, torch.tensor([[10.0]])
    )
    loss.backward()
    assert live.grad[0, 0, 2] < 0.0
    assert not observation.center_zyx.requires_grad


def test_broad_phase_posterior_has_lower_confidence_than_crisp_one():
    points, sample_phase = _straight_cache()
    coordinate = torch.arange(21, dtype=torch.float32)
    crisp = torch.exp(-0.5 * ((coordinate - 10.0) / 0.5) ** 2)
    broad = torch.exp(-0.5 * ((coordinate - 10.0) / 4.0) ** 2)
    observation = dense_phase_observations(
        points.repeat(2, 1, 1),
        sample_phase.repeat(2, 1),
        torch.stack([crisp, broad]),
        torch.tensor([[1.0], [1.0]]),
        probability_power=1.0,
    )
    assert observation.confidence[0, 0] > observation.confidence[1, 0]
    assert observation.physical_std_wv[0, 0] < observation.physical_std_wv[1, 0]


def test_exact_density_integral_is_zero_loss_at_correct_length():
    x = torch.linspace(0.0, 20.0, 21)
    points = torch.stack([torch.zeros_like(x), torch.zeros_like(x), x], dim=-1)[None]
    loss, metrics = dense_metric_density_loss(
        points, torch.full((1, 21), 0.1), torch.tensor([2.0])
    )
    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(metrics["integrated_mean"], torch.tensor(2.0))


def test_density_gradient_expands_an_underlength_live_polyline():
    endpoint = torch.tensor(15.0, requires_grad=True)
    x = torch.linspace(0.0, 1.0, 21) * endpoint
    points = torch.stack([torch.zeros_like(x), torch.zeros_like(x), x], dim=-1)[None]
    loss, _ = dense_metric_density_loss(
        points, torch.full((1, 21), 0.1), torch.tensor([2.0])
    )
    loss.backward()
    assert endpoint.grad < 0.0


def test_density_rejects_insufficiently_covered_ray():
    x = torch.arange(10, dtype=torch.float32)
    points = torch.stack([x * 0, x * 0, x], dim=-1)[None].requires_grad_()
    valid = torch.zeros((1, 10), dtype=torch.bool)
    valid[:, :5] = True
    loss, metrics = dense_metric_density_loss(
        points,
        torch.full((1, 10), 0.1),
        torch.tensor([1.0]),
        sample_valid=valid,
        minimum_coverage=0.9,
    )
    assert loss == 0.0
    assert metrics["valid_fraction"] == 0.0
