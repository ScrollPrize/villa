import math

import torch

import fit_spiral
from fit_spiral import (
    get_exponential_lr_at_step,
    realign_optimizer_lr_schedule,
)


def test_final_factor_is_fraction_of_initial_lr_at_training_horizon():
    initial_lr = 3e-5

    assert math.isclose(
        get_exponential_lr_at_step(initial_lr, 0.9, 30_000, 30_000),
        initial_lr * 0.9,
        rel_tol=1e-12,
    )


def test_realign_uses_enlarged_horizon_and_tracks_next_optimizer_step():
    initial_lr = 3e-5
    parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    optimiser = torch.optim.SGD([parameter], lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimiser, gamma=0.9 ** (1.0 / 30_000))

    scheduler, horizon = realign_optimizer_lr_schedule(
        optimiser,
        scheduler,
        initial_lr=initial_lr,
        final_factor=0.9,
        completed_steps=30_000,
        training_horizon=60_000,
        exponential=True,
    )

    expected_at_realign = initial_lr * math.sqrt(0.9)
    assert horizon == 60_000
    assert math.isclose(
        optimiser.param_groups[0]["lr"], expected_at_realign, rel_tol=1e-12)
    assert math.isclose(
        scheduler.gamma, 0.9 ** (1.0 / 60_000), rel_tol=1e-12)
    assert scheduler.last_epoch == 30_000

    optimiser.step()
    scheduler.step()

    assert scheduler.last_epoch == 30_001
    assert math.isclose(
        optimiser.param_groups[0]["lr"],
        get_exponential_lr_at_step(
            initial_lr, 0.9, 30_001, 60_000),
        rel_tol=1e-12,
    )


def test_high_res_flow_scale_uses_absolute_iteration_after_resume(monkeypatch):
    monkeypatch.setattr(fit_spiral, "cfg", {
        "model_flow_field_high_res_lr_scale_initial": 0.2,
        "model_flow_field_high_res_lr_scale_final": 0.8,
        "model_flow_field_high_res_lr_ramp_start_step": 10_000,
        "model_flow_field_high_res_lr_ramp_steps": 20_000,
    })

    assert math.isclose(
        fit_spiral.get_flow_field_high_res_lr_scale(20_000), 0.5)
    assert math.isclose(
        fit_spiral.get_flow_field_high_res_lr_scale(30_000), 0.8)
    # Extending optimizer_num_training_steps does not restart or stretch this
    # explicitly step-based ramp; resumed execution supplies the durable step.
    assert math.isclose(
        fit_spiral.get_flow_field_high_res_lr_scale(30_001), 0.8)
