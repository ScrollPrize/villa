"""Regression test for the cfg-level LR-change-on-resume recipe.

Background: `torch.optim.lr_scheduler._LRScheduler.state_dict()` saves
every instance attribute (max_steps, exponent, warmup_steps, initial_lr,
ctr, last_epoch, _step_count, ...). A naive
`scheduler.load_state_dict(saved_state)` on resume therefore overwrites
the new constructor kwargs with whatever was saved by the previous run.

`vesuvius.neural_tracing.autoreg_mesh.train` works around this by
restoring ONLY the step counters (ctr/last_epoch/_step_count) from the
saved state, letting max_steps/exponent/warmup_steps come from the fresh
constructor. This test guards that recipe so a future refactor doesn't
silently re-introduce the "config changes ignored on resume" bug.
"""
from __future__ import annotations

import torch

from vesuvius.models.training.lr_schedulers import get_scheduler


def _make_optimizer():
    return torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=3e-4)


def test_naive_load_state_dict_overwrites_constructor_kwargs():
    """Documents the broken behavior we are working around: the saved
    state's max_steps and exponent clobber the new constructor's."""
    opt_a = _make_optimizer()
    sched_a = get_scheduler("warmup_poly", opt_a, 3e-4, 150_000, warmup_steps=5000, exponent=1.0)
    for _ in range(100_000):
        sched_a.step()
    saved = sched_a.state_dict()

    opt_b = _make_optimizer()
    sched_b = get_scheduler("warmup_poly", opt_b, 3e-4, 200_000, warmup_steps=5000, exponent=0.5)
    assert sched_b.max_steps == 200_000
    assert sched_b.exponent == 0.5

    sched_b.load_state_dict(saved)
    # The new constructor kwargs are CLOBBERED by the saved state -- this
    # is the broken behaviour the resume recipe in train.py works around.
    assert sched_b.max_steps == 150_000
    assert sched_b.exponent == 1.0


def test_counter_only_restore_recipe_preserves_new_constructor_kwargs():
    """The fixed behavior: restore only counters; new constructor wins."""
    opt_a = _make_optimizer()
    sched_a = get_scheduler("warmup_poly", opt_a, 3e-4, 150_000, warmup_steps=5000, exponent=1.0)
    for _ in range(100_000):
        sched_a.step()
    saved = sched_a.state_dict()
    saved_ctr = saved["ctr"]

    opt_b = _make_optimizer()
    sched_b = get_scheduler("warmup_poly", opt_b, 3e-4, 200_000, warmup_steps=5000, exponent=0.5)

    # Apply the resume recipe used in autoreg_mesh.train.
    for counter_key in ("ctr", "last_epoch", "_step_count"):
        if counter_key in saved:
            setattr(sched_b, counter_key, saved[counter_key])

    # New constructor kwargs survive; counters carry over.
    assert sched_b.max_steps == 200_000
    assert sched_b.exponent == 0.5
    assert sched_b.warmup_steps == 5000
    assert sched_b.ctr == saved_ctr

    # LR after the next step matches the v5 closed form, not v4's.
    sched_b.step()
    after_step = sched_b.ctr - 1
    expected = 3e-4 * (1 - (after_step - 5000) / (200_000 - 5000)) ** 0.5
    actual = opt_b.param_groups[0]["lr"]
    assert abs(actual - expected) < 1e-9, f"LR mismatch: actual={actual} expected={expected}"
