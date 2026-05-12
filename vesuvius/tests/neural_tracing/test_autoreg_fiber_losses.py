from __future__ import annotations

import numpy as np
import torch

from vesuvius.neural_tracing.autoreg_fiber.losses import (
    _straightness_huber_loss,
    _tube_radius_huber_loss,
)


def _build_target_and_mask(target: torch.Tensor) -> dict:
    return {
        "target_xyz": target,
        "target_supervision_mask": torch.ones(target.shape[:2], dtype=torch.bool),
    }


def test_tube_radius_loss_zero_for_pure_tangent_error() -> None:
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    pred = target.clone()
    pred[..., 0] += torch.tensor([3.0, -2.0, 5.0, 1.0])

    loss = _tube_radius_huber_loss(pred, _build_target_and_mask(target))

    assert loss.item() < 1e-5


def test_tube_radius_loss_picks_up_pure_perpendicular_error() -> None:
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    pred = target.clone()
    pred[..., 1] += 4.0

    loss = _tube_radius_huber_loss(pred, _build_target_and_mask(target))

    expected = float(torch.nn.functional.smooth_l1_loss(
        torch.tensor([4.0]), torch.tensor([0.0]), reduction="mean"
    ).item())
    assert abs(loss.item() - expected) < 1e-5


def test_tube_radius_loss_isolates_perpendicular_component_in_mixed_error() -> None:
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    pred = target.clone()
    pred[..., 0] += 7.0
    pred[..., 1] += 3.0

    loss = _tube_radius_huber_loss(pred, _build_target_and_mask(target))

    expected = float(torch.nn.functional.smooth_l1_loss(
        torch.tensor([3.0]), torch.tensor([0.0]), reduction="mean"
    ).item())
    assert abs(loss.item() - expected) < 1e-5


def test_straightness_loss_zero_when_pred_matches_curved_target() -> None:
    angles = torch.linspace(0.0, np.pi, 16)
    radius = 30.0
    target = torch.stack(
        [torch.cos(angles) * radius, torch.sin(angles) * radius, torch.zeros_like(angles)],
        dim=-1,
    ).unsqueeze(0)

    loss = _straightness_huber_loss(target.clone(), _build_target_and_mask(target))

    assert loss.item() < 1e-6


def test_straightness_loss_picks_up_excess_curvature_relative_to_target() -> None:
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    pred = target.clone()
    pred[:, 1, 1] = 2.0
    pred[:, 2, 1] = -2.0

    loss_pred_wobble = _straightness_huber_loss(pred, _build_target_and_mask(target))
    loss_target_straight = _straightness_huber_loss(target.clone(), _build_target_and_mask(target))

    assert loss_target_straight.item() < 1e-6
    assert loss_pred_wobble.item() > 0.5


def test_straightness_loss_unbiased_to_smooth_curve_offsets() -> None:
    angles = torch.linspace(0.0, np.pi, 16)
    radius = 30.0
    target = torch.stack(
        [torch.cos(angles) * radius, torch.sin(angles) * radius, torch.zeros_like(angles)],
        dim=-1,
    ).unsqueeze(0)
    pred_translated = target.clone()
    pred_translated[..., 0] += 5.0

    loss = _straightness_huber_loss(pred_translated, _build_target_and_mask(target))
    assert loss.item() < 1e-6


def test_tube_radius_loss_handles_curved_target() -> None:
    angles = torch.linspace(0.0, np.pi, 16)
    radius = 30.0
    target = torch.stack(
        [torch.cos(angles) * radius, torch.sin(angles) * radius, torch.zeros_like(angles)],
        dim=-1,
    ).unsqueeze(0)

    delta = 0.10
    pred_along = torch.stack(
        [
            torch.cos(angles + delta) * radius,
            torch.sin(angles + delta) * radius,
            torch.zeros_like(angles),
        ],
        dim=-1,
    ).unsqueeze(0)
    pred_perp = target.clone()
    pred_perp[..., 2] += 2.0

    loss_along = _tube_radius_huber_loss(pred_along, _build_target_and_mask(target))
    loss_perp = _tube_radius_huber_loss(pred_perp, _build_target_and_mask(target))

    assert loss_along.item() < 0.05
    assert loss_perp.item() > 1.0
    assert loss_perp.item() > loss_along.item() * 30
