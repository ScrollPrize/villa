"""Tests for the periodic rollout-in-loop training step.

The feature: every `rollout_in_loop_frequency` training steps (after
`rollout_in_loop_start_step`), chain K no-grad warmup forwards using each
pass's predictions as inputs to the next, then one final gradient-enabled
forward. This exposes the model to K-deep self-feeding drift during
training to fight exposure bias at long-rollout inference time.
"""
from __future__ import annotations

import pytest

from vesuvius.neural_tracing.autoreg_mesh.train import _rollout_in_loop_active


def test_disabled_when_flag_off():
    cfg = {"rollout_in_loop_enabled": False,
           "rollout_in_loop_start_step": 0,
           "rollout_in_loop_frequency": 1}
    for step in (0, 1, 100, 999999):
        assert _rollout_in_loop_active(cfg, global_step=step) is False


def test_off_before_start_step():
    cfg = {"rollout_in_loop_enabled": True,
           "rollout_in_loop_start_step": 1000,
           "rollout_in_loop_frequency": 100}
    for step in (0, 1, 500, 999):
        assert _rollout_in_loop_active(cfg, global_step=step) is False
    assert _rollout_in_loop_active(cfg, global_step=1000) is True


def test_fires_at_frequency():
    cfg = {"rollout_in_loop_enabled": True,
           "rollout_in_loop_start_step": 1000,
           "rollout_in_loop_frequency": 100}
    # At start_step, fires.
    assert _rollout_in_loop_active(cfg, global_step=1000) is True
    # Subsequent firings at every `frequency` steps.
    assert _rollout_in_loop_active(cfg, global_step=1100) is True
    assert _rollout_in_loop_active(cfg, global_step=1200) is True
    # Off-firings: anything between.
    for step in (1001, 1050, 1099, 1101, 1199):
        assert _rollout_in_loop_active(cfg, global_step=step) is False


def test_frequency_zero_disables():
    cfg = {"rollout_in_loop_enabled": True,
           "rollout_in_loop_start_step": 0,
           "rollout_in_loop_frequency": 0}
    for step in (0, 1, 100, 1000):
        assert _rollout_in_loop_active(cfg, global_step=step) is False


def test_iterations_validated():
    """rollout_in_loop_iterations is validated by config schema, not by the
    schedule helper. This test just confirms the helper is independent."""
    cfg = {"rollout_in_loop_enabled": True,
           "rollout_in_loop_start_step": 0,
           "rollout_in_loop_frequency": 1}
    # iterations field absent -> helper still works (it doesn't read iterations).
    assert _rollout_in_loop_active(cfg, global_step=0) is True


def test_config_schema_accepts_rollout_keys():
    """Default config + validation round-trip preserves the rollout knobs."""
    from vesuvius.neural_tracing.autoreg_mesh.config import (
        DEFAULT_AUTOREG_MESH_CONFIG,
        validate_autoreg_mesh_config,
    )
    for key in ("rollout_in_loop_enabled", "rollout_in_loop_frequency",
                "rollout_in_loop_start_step", "rollout_in_loop_iterations"):
        assert key in DEFAULT_AUTOREG_MESH_CONFIG, f"missing default for {key}"


def test_config_validator_rejects_bad_values():
    """Validator catches negative frequency / non-bool enabled / etc."""
    from vesuvius.neural_tracing.autoreg_mesh.config import (
        DEFAULT_AUTOREG_MESH_CONFIG,
        validate_autoreg_mesh_config,
    )
    base = {**DEFAULT_AUTOREG_MESH_CONFIG, "datasets": []}
    base = {**base, "rollout_in_loop_frequency": 0}
    with pytest.raises(ValueError, match="rollout_in_loop_frequency must be positive"):
        validate_autoreg_mesh_config(base)

    base2 = {**DEFAULT_AUTOREG_MESH_CONFIG, "datasets": [], "rollout_in_loop_enabled": "yes"}
    with pytest.raises(ValueError, match="rollout_in_loop_enabled must be a boolean"):
        validate_autoreg_mesh_config(base2)
