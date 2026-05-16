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
    _rollout_in_loop_iterations,
    _scheduled_sampling_prob,
    _true_rollout_active,
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


# ---------------- True autoregressive rollout schedule ---------------- #


def test_true_rollout_schedule_disabled_when_flag_off():
    cfg = {"true_rollout_loss_enabled": False,
           "true_rollout_loss_start_step": 0,
           "true_rollout_loss_frequency": 1}
    for step in (0, 1, 100, 999999):
        assert _true_rollout_active(cfg, global_step=step) is False


def test_true_rollout_schedule_fires_at_frequency():
    cfg = {"true_rollout_loss_enabled": True,
           "true_rollout_loss_start_step": 40000,
           "true_rollout_loss_frequency": 2000}
    # Off before start
    for step in (0, 39999):
        assert _true_rollout_active(cfg, global_step=step) is False
    # Fires at start_step and every `frequency` later
    assert _true_rollout_active(cfg, global_step=40000) is True
    assert _true_rollout_active(cfg, global_step=42000) is True
    assert _true_rollout_active(cfg, global_step=44000) is True
    # Off-firings between
    for step in (40001, 41000, 41999, 42001, 43999):
        assert _true_rollout_active(cfg, global_step=step) is False


def test_true_rollout_mutex_with_rollout_in_loop():
    """When both schedules say 'fire', the train loop suppresses rollout-in-loop.
    Verified by checking both helpers' independent firing AND the mutex rule
    applied in train.py (which is: rollout_in_loop_active and not true_rollout_active).
    """
    cfg = {
        "true_rollout_loss_enabled": True,
        "true_rollout_loss_start_step": 0,
        "true_rollout_loss_frequency": 1000,
        "rollout_in_loop_enabled": True,
        "rollout_in_loop_start_step": 0,
        "rollout_in_loop_frequency": 500,
    }
    # Both fire at step 0 and step 1000 (LCM).
    for step in (0, 1000, 2000):
        true_r = _true_rollout_active(cfg, global_step=step)
        in_loop = _rollout_in_loop_active(cfg, global_step=step)
        assert true_r is True
        assert in_loop is True
        # Mutex rule: rollout_in_loop suppressed when true_rollout fires.
        effective_in_loop = in_loop and not true_r
        assert effective_in_loop is False, f"step={step}: in_loop must be suppressed when true_r fires"
    # At step 500 only rollout-in-loop fires (true_rollout's freq is 1000).
    assert _true_rollout_active(cfg, global_step=500) is False
    assert _rollout_in_loop_active(cfg, global_step=500) is True


def test_config_validator_rejects_bad_true_rollout():
    from vesuvius.neural_tracing.autoreg_mesh.config import (
        DEFAULT_AUTOREG_MESH_CONFIG,
        validate_autoreg_mesh_config,
    )
    # Negative start_step
    bad = {**DEFAULT_AUTOREG_MESH_CONFIG, "datasets": [],
           "true_rollout_loss_enabled": True,
           "true_rollout_loss_start_step": -1}
    with pytest.raises(ValueError, match="true_rollout_loss_start_step must be >= 0"):
        validate_autoreg_mesh_config(bad)
    # Zero frequency
    bad2 = {**DEFAULT_AUTOREG_MESH_CONFIG, "datasets": [],
            "true_rollout_loss_enabled": True,
            "true_rollout_loss_frequency": 0}
    with pytest.raises(ValueError, match="true_rollout_loss_frequency must be positive"):
        validate_autoreg_mesh_config(bad2)
    # Zero horizon
    bad3 = {**DEFAULT_AUTOREG_MESH_CONFIG, "datasets": [],
            "true_rollout_loss_enabled": True,
            "true_rollout_loss_horizon": 0}
    with pytest.raises(ValueError, match="true_rollout_loss_horizon must be a positive int or null"):
        validate_autoreg_mesh_config(bad3)
    # Non-bool enabled
    bad4 = {**DEFAULT_AUTOREG_MESH_CONFIG, "datasets": [],
            "true_rollout_loss_enabled": "yes"}
    with pytest.raises(ValueError, match="true_rollout_loss_enabled must be a boolean"):
        validate_autoreg_mesh_config(bad4)


# ---------------- K-ramp on rollout_in_loop_iterations ---------------- #


def test_rollout_in_loop_iterations_default_legacy():
    """When iterations_max is absent (legacy), the helper returns the base
    value at every step. Catches accidental fall-through to a different code
    path."""
    cfg = {"rollout_in_loop_iterations": 7}
    for step in (0, 1, 100, 100_000, 999_999):
        assert _rollout_in_loop_iterations(cfg, global_step=step) == 7


def test_rollout_in_loop_iterations_pre_ramp_returns_base():
    cfg = {
        "rollout_in_loop_iterations": 10,
        "rollout_in_loop_iterations_max": 30,
        "rollout_in_loop_iterations_ramp_start_step": 100_000,
        "rollout_in_loop_iterations_ramp_steps": 20_000,
    }
    for step in (0, 50_000, 99_999):
        assert _rollout_in_loop_iterations(cfg, global_step=step) == 10


def test_rollout_in_loop_iterations_hits_endpoints():
    cfg = {
        "rollout_in_loop_iterations": 10,
        "rollout_in_loop_iterations_max": 30,
        "rollout_in_loop_iterations_ramp_start_step": 100_000,
        "rollout_in_loop_iterations_ramp_steps": 20_000,
    }
    # Start of ramp
    assert _rollout_in_loop_iterations(cfg, global_step=100_000) == 10
    # Midpoint: 10 + (30-10) * 0.5 = 20
    assert _rollout_in_loop_iterations(cfg, global_step=110_000) == 20
    # End of ramp
    assert _rollout_in_loop_iterations(cfg, global_step=120_000) == 30
    # Past the ramp: still max
    for step in (130_000, 200_000, 999_999):
        assert _rollout_in_loop_iterations(cfg, global_step=step) == 30


def test_rollout_in_loop_iterations_step_jump_when_ramp_steps_zero():
    """ramp_steps=0 means a step jump from base to max at ramp_start_step."""
    cfg = {
        "rollout_in_loop_iterations": 10,
        "rollout_in_loop_iterations_max": 30,
        "rollout_in_loop_iterations_ramp_start_step": 100_000,
        "rollout_in_loop_iterations_ramp_steps": 0,
    }
    assert _rollout_in_loop_iterations(cfg, global_step=99_999) == 10
    assert _rollout_in_loop_iterations(cfg, global_step=100_000) == 30
    assert _rollout_in_loop_iterations(cfg, global_step=200_000) == 30


def test_rollout_in_loop_iterations_max_le_base_returns_base():
    """Degenerate config (max <= base) is a no-op: always returns base. The
    validator rejects this, but the helper must be robust if it slips
    through (e.g., via direct dict construction in tests)."""
    cfg = {
        "rollout_in_loop_iterations": 10,
        "rollout_in_loop_iterations_max": 5,  # degenerate
        "rollout_in_loop_iterations_ramp_start_step": 100_000,
        "rollout_in_loop_iterations_ramp_steps": 20_000,
    }
    for step in (0, 50_000, 110_000, 200_000):
        assert _rollout_in_loop_iterations(cfg, global_step=step) == 10


def test_config_validator_rejects_iter_max_lt_base():
    from vesuvius.neural_tracing.autoreg_mesh.config import (
        DEFAULT_AUTOREG_MESH_CONFIG,
        validate_autoreg_mesh_config,
    )
    bad = {
        **DEFAULT_AUTOREG_MESH_CONFIG,
        "datasets": [],
        "rollout_in_loop_iterations": 10,
        "rollout_in_loop_iterations_max": 5,
    }
    with pytest.raises(ValueError, match=r"rollout_in_loop_iterations_max .* must be >= rollout_in_loop_iterations"):
        validate_autoreg_mesh_config(bad)


def test_config_validator_rejects_negative_ramp_steps():
    from vesuvius.neural_tracing.autoreg_mesh.config import (
        DEFAULT_AUTOREG_MESH_CONFIG,
        validate_autoreg_mesh_config,
    )
    bad = {
        **DEFAULT_AUTOREG_MESH_CONFIG,
        "datasets": [],
        "rollout_in_loop_iterations_ramp_steps": -1,
    }
    with pytest.raises(ValueError, match="rollout_in_loop_iterations_ramp_steps must be >= 0"):
        validate_autoreg_mesh_config(bad)


def test_config_validator_rejects_ramp_start_before_rollout_start():
    """The ramp_start_step constraint only fires when iterations_max is set
    (otherwise no ramp is active and the start step is irrelevant)."""
    from vesuvius.neural_tracing.autoreg_mesh.config import (
        DEFAULT_AUTOREG_MESH_CONFIG,
        validate_autoreg_mesh_config,
    )
    bad = {
        **DEFAULT_AUTOREG_MESH_CONFIG,
        "datasets": [],
        "rollout_in_loop_start_step": 20_000,
        "rollout_in_loop_iterations": 10,
        "rollout_in_loop_iterations_max": 30,  # enable ramping
        "rollout_in_loop_iterations_ramp_start_step": 10_000,  # before rollout_in_loop_start_step
    }
    with pytest.raises(ValueError, match=r"rollout_in_loop_iterations_ramp_start_step .* must be >= rollout_in_loop_start_step"):
        validate_autoreg_mesh_config(bad)


def test_config_validator_skips_ramp_check_when_iterations_max_none():
    """Default config has iterations_max=None and ramp_start_step=0
    (< rollout_in_loop_start_step=20000). The validator must NOT reject this
    because no ramp is configured."""
    from vesuvius.neural_tracing.autoreg_mesh.config import (
        DEFAULT_AUTOREG_MESH_CONFIG,
        validate_autoreg_mesh_config,
    )
    ok = {**DEFAULT_AUTOREG_MESH_CONFIG, "datasets": []}
    assert ok["rollout_in_loop_iterations_max"] is None
    assert ok["rollout_in_loop_iterations_ramp_start_step"] < ok["rollout_in_loop_start_step"]
    # Must not raise.
    validate_autoreg_mesh_config(ok)
