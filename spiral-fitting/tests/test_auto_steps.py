import math

import pytest

from auto_steps import auto_steps_relative_improvement, auto_steps_should_stop
from config import Config
from ddp_helpers import DistributedContext
import fit_spiral
from fit_spiral import FitContext


def test_no_decision_until_the_window_has_history_before_it():
    assert auto_steps_relative_improvement([10.0, 9.0], window=2) == math.inf
    assert not auto_steps_should_stop([10.0, 9.0], window=2, rel_threshold=0.5)


def test_relative_improvement_compares_best_recent_with_best_before():
    history = [10.0, 8.0, 9.0, 7.6]
    assert math.isclose(auto_steps_relative_improvement(history, window=2), (8.0 - 7.6) / 8.0)
    assert auto_steps_should_stop(history, window=2, rel_threshold=0.06)
    assert not auto_steps_should_stop(history, window=2, rel_threshold=0.04)


def test_rising_loss_stops_at_any_non_negative_threshold():
    assert auto_steps_should_stop([5.0, 4.0, 4.5, 4.2], window=2, rel_threshold=0.0)


def test_block_minima_give_the_same_decision_as_every_step():
    # The fit records the lowest loss per 200-step block; with a window that is a whole
    # number of blocks, the decision at each block end equals the per-step rule.
    block, window = 200, 600
    losses = [100.0 / (1 + i / 300) + 3.0 * math.sin(i * 0.37) for i in range(6000)]
    per_step, blocks = [], []
    for i, value in enumerate(losses):
        per_step.append(value)
        if (i + 1) % block == 0:
            blocks.append(min(losses[i + 1 - block:i + 1]))
            for threshold in (1e-4, 1e-3, 1e-2, 5e-2):
                assert (auto_steps_should_stop(blocks, window=window // block, rel_threshold=threshold)
                        == auto_steps_should_stop(per_step, window=window, rel_threshold=threshold))


def _context(overrides, world_size=1):
    context = FitContext.__new__(FitContext)
    context.config = Config(overrides).as_dict()
    context.dist = DistributedContext(rank=0, world_size=world_size)
    return context


def test_settings_are_off_by_default():
    assert _context({})._auto_steps_settings() is None


def test_settings_resolve_window_in_blocks_and_lasagna_minimum():
    settings = _context({"optimizer_auto_steps": True})._auto_steps_settings()
    assert settings == (1000 // fit_spiral._AUTO_STEPS_BLOCK, 2000, 1.0e-4)
    settings = _context({"optimizer_auto_steps": True, "optimizer_auto_steps_window": 2000,
                         "optimizer_auto_steps_min": 3000,
                         "optimizer_auto_steps_rel_threshold": 1e-3})._auto_steps_settings()
    assert settings == (10, 3000, 1e-3)


def test_window_must_be_a_whole_number_of_blocks():
    with pytest.raises(ValueError, match="multiple of 200"):
        _context({"optimizer_auto_steps": True, "optimizer_auto_steps_window": 500})._auto_steps_settings()


def test_multi_process_fits_run_to_the_horizon(capsys):
    assert _context({"optimizer_auto_steps": True}, world_size=2)._auto_steps_settings() is None
    assert "single-process fits only" in capsys.readouterr().out
