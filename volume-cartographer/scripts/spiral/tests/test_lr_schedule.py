import math

import torch

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
