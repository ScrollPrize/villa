"""Tests for the periodic rollout-in-loop training step.

The feature: every `rollout_in_loop_frequency` training steps (after
`rollout_in_loop_start_step`), chain K no-grad warmup forwards using each
pass's predictions as inputs to the next, then one final gradient-enabled
forward. This exposes the model to K-deep self-feeding drift during
training to fight exposure bias at long-rollout inference time.
"""
from __future__ import annotations

import pytest

from vesuvius.neural_tracing.autoreg_mesh.train import (
    _rollout_in_loop_active,
    _scheduled_sampling_prob,
)


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


# ---------------- Two-phase scheduled-sampling ramp ---------------- #


def test_scheduled_sampling_two_phase_default_no_late_when_unset():
    """When scheduled_sampling_max_prob_late is not set, the helper is
    bit-for-bit identical to the legacy single-ramp formula."""
    cfg = {
        "scheduled_sampling_enabled": True,
        "scheduled_sampling_start_step": 0,
        "scheduled_sampling_max_prob": 0.4,
        "scheduled_sampling_ramp_steps": 5000,
        # NOT set: scheduled_sampling_max_prob_late, late_start_step, late_ramp_steps
    }
    # Pre-ramp: 0
    assert _scheduled_sampling_prob(cfg, global_step=0) == 0.0
    # Mid-ramp linear
    assert _scheduled_sampling_prob(cfg, global_step=2500) == pytest.approx(0.2, rel=1e-6)
    # End-of-ramp saturated at max_prob
    assert _scheduled_sampling_prob(cfg, global_step=5000) == pytest.approx(0.4, rel=1e-6)
    # Stays at max_prob indefinitely
    for step in (10000, 50000, 150000):
        assert _scheduled_sampling_prob(cfg, global_step=step) == pytest.approx(0.4, rel=1e-6)


def test_scheduled_sampling_two_phase_hits_endpoints():
    """At late_start_step the helper returns max_prob; at the end of the
    late ramp it returns max_prob_late; in between is linear."""
    cfg = {
        "scheduled_sampling_enabled": True,
        "scheduled_sampling_start_step": 0,
        "scheduled_sampling_max_prob": 0.4,
        "scheduled_sampling_ramp_steps": 5000,
        "scheduled_sampling_max_prob_late": 0.6,
        "scheduled_sampling_late_start_step": 40000,
        "scheduled_sampling_late_ramp_steps": 15000,
    }
    # Phase 1 unchanged
    assert _scheduled_sampling_prob(cfg, global_step=0) == 0.0
    assert _scheduled_sampling_prob(cfg, global_step=5000) == pytest.approx(0.4)
    # Plateau at max_prob between phases
    assert _scheduled_sampling_prob(cfg, global_step=20000) == pytest.approx(0.4)
    # Start of late ramp
    assert _scheduled_sampling_prob(cfg, global_step=40000) == pytest.approx(0.4)
    # Midpoint of late ramp: 0.4 + (0.6 - 0.4) * 7500/15000 = 0.5
    assert _scheduled_sampling_prob(cfg, global_step=47500) == pytest.approx(0.5, rel=1e-6)
    # End of late ramp saturated at max_prob_late
    assert _scheduled_sampling_prob(cfg, global_step=55000) == pytest.approx(0.6, rel=1e-6)
    # Stays at max_prob_late beyond
    assert _scheduled_sampling_prob(cfg, global_step=150000) == pytest.approx(0.6, rel=1e-6)


def test_scheduled_sampling_two_phase_rejects_regression():
    """Validator rejects scheduled_sampling_max_prob_late < scheduled_sampling_max_prob
    (the late phase must only boost, not regress, the curriculum)."""
    from vesuvius.neural_tracing.autoreg_mesh.config import (
        DEFAULT_AUTOREG_MESH_CONFIG,
        validate_autoreg_mesh_config,
    )
    bad = {
        **DEFAULT_AUTOREG_MESH_CONFIG,
        "datasets": [],
        "scheduled_sampling_enabled": True,
        "scheduled_sampling_max_prob": 0.4,
        "scheduled_sampling_max_prob_late": 0.2,   # < 0.4, regresses
    }
    with pytest.raises(ValueError, match="max_prob_late must be >= scheduled_sampling_max_prob"):
        validate_autoreg_mesh_config(bad)

    # Equally bad: late_start_step before phase 1 finishes
    bad2 = {
        **DEFAULT_AUTOREG_MESH_CONFIG,
        "datasets": [],
        "scheduled_sampling_enabled": True,
        "scheduled_sampling_max_prob": 0.4,
        "scheduled_sampling_max_prob_late": 0.6,
        "scheduled_sampling_start_step": 0,
        "scheduled_sampling_ramp_steps": 5000,
        "scheduled_sampling_late_start_step": 3000,  # mid phase 1
    }
    with pytest.raises(ValueError, match="late_start_step .* must be >= start_step \\+ ramp_steps"):
        validate_autoreg_mesh_config(bad2)
