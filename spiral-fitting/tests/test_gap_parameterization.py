import math

import torch

from checkpoint_migrations import (
    expand_gap_checkpoint_capacity,
    migrate_legacy_gap_parameterization,
)
from gap_parameterization import (
    calibrated_gap_softplus_scale,
    lower_bounded_dr,
    lower_bounded_gap,
)
from transforms import GapExpanderParams, GapExpandingTransform


def _params(capacity=8):
    return GapExpanderParams(
        resolution=24, min_z=0.0, max_z=24.0,
        num_windings=capacity, dr_per_winding=16.0)


def test_zero_latent_is_identity_and_extreme_tails_remain_finite_and_ordered():
    params = _params(capacity=144)
    dr = torch.tensor(16.0)
    transform = GapExpandingTransform(
        params, dr, 0.0, 24.0, gap_expander_lr_scale=0.3,
        min_gap=1.0, softplus_bias=4.0)
    theta = torch.tensor([0.7])
    z = torch.tensor([12.0])

    identity = transform.get_transformed_winding_radii(theta, z)
    torch.testing.assert_close(
        identity.diff(dim=-1), torch.full_like(identity[..., 1:], 16.0),
        atol=2.0e-4, rtol=0.0)

    with torch.no_grad():
        params.logits.fill_(-100.0)
    contracted = transform.get_transformed_winding_radii(theta, z)
    contracted_gaps = contracted.diff(dim=-1)
    assert torch.isfinite(contracted).all()
    # The globally pinned first sample remains identity; every learned tail
    # gap is nevertheless representably positive at production-scale radii.
    assert float(contracted_gaps.detach().min()) >= 0.999
    assert torch.all(contracted_gaps > 0.0)

    with torch.no_grad():
        params.logits.fill_(1.0e6)
    expanded = transform.get_transformed_winding_radii(theta, z)
    assert torch.isfinite(expanded).all()
    assert torch.all(expanded.diff(dim=-1) > 0.0)


def test_floor_prevents_the_previously_collapsed_round_trip():
    params = _params(capacity=136)
    with torch.no_grad():
        params.logits.fill_(-0.226)
    transform = GapExpandingTransform(
        params, torch.tensor(16.0), 0.0, 24.0,
        gap_expander_lr_scale=0.3, min_gap=1.0, softplus_bias=4.0)
    theta = 0.8
    canonical_radius = (130.5 + theta / (2.0 * math.pi)) * 16.0
    point = torch.tensor([[12.0, math.sin(theta) * canonical_radius,
                           math.cos(theta) * canonical_radius]])

    mapped = transform(point)
    restored = transform.inv(mapped)

    assert torch.isfinite(mapped).all()
    torch.testing.assert_close(restored, point, atol=2.0e-3, rtol=0.0)


def _legacy_checkpoint():
    params = _params(capacity=5)
    logits = torch.linspace(
        -0.2, 0.04, params.logits.numel(), dtype=torch.float32,
    ).reshape_as(params.logits)
    config = {
        "model_initial_dr_per_winding": 16.0,
        "model_gap_expander_lr_scale": 0.3,
        "model_gap_expander_logit_resolution": 24,
        "model_gap_expander_num_windings": 5,
        # Keep this tiny synthetic checkpoint tiny; real legacy checkpoints
        # lack the field and migrate to the current default spare capacity.
        "model_gap_expander_capacity_windings": 5,
    }
    return {
        "spiral_and_transform": {
            "dr_per_winding_logit": torch.tensor(16.0 / 12.0),
            "gap_expander_params.logits": logits,
            "gap_expander_params.winding_first_logit_idx":
                params.winding_first_logit_idx.clone(),
        },
        "optimiser": {
            "state": {
                10: {"step": torch.tensor(9.0), "exp_avg": torch.ones(())},
                20: {"step": torch.tensor(9.0), "exp_avg": torch.ones_like(logits),
                       "exp_avg_sq": torch.ones_like(logits)},
                30: {"step": torch.tensor(9.0), "exp_avg": torch.ones(3)},
            },
            "param_groups": [
                {"params": [10]}, {"params": [30]}, {"params": [20]},
            ],
        },
        "cfg": config,
        "requested_config": dict(config),
        "resolved_config": dict(config),
    }


def test_legacy_migration_preserves_valid_gaps_and_projects_broken_ones():
    checkpoint = _legacy_checkpoint()
    old_logits = checkpoint["spiral_and_transform"][
        "gap_expander_params.logits"]
    old_gaps = 16.0 * torch.exp(old_logits * 60.0)

    migrated = migrate_legacy_gap_parameterization(checkpoint)

    state = migrated["spiral_and_transform"]
    new_dr = lower_bounded_dr(state["dr_per_winding_logit"], 1.0)
    scale = calibrated_gap_softplus_scale(16.0, 1.0, 4.0)
    new_gaps = lower_bounded_gap(
        state["gap_expander_params.logits"] * 0.3,
        new_dr, 1.0, 4.0, scale)
    expected = old_gaps.clamp_min(1.001)
    torch.testing.assert_close(new_dr, torch.tensor(16.0))
    torch.testing.assert_close(new_gaps, expected, rtol=3.0e-4, atol=2.0e-3)
    assert migrated["gap_parameterization_version"] == 2
    assert migrated["gap_parameterization_migration"][
        "projected_gap_logits"] > 0
    assert 10 not in migrated["optimiser"]["state"]
    assert 20 not in migrated["optimiser"]["state"]
    assert 30 in migrated["optimiser"]["state"]


def test_capacity_growth_appends_identity_latents_and_zero_moments():
    checkpoint = migrate_legacy_gap_parameterization(_legacy_checkpoint())
    logits = checkpoint["spiral_and_transform"][
        "gap_expander_params.logits"]
    checkpoint["optimiser"]["state"][20] = {
        "exp_avg": torch.ones_like(logits),
        "exp_avg_sq": torch.full_like(logits, 2.0),
    }

    expanded = expand_gap_checkpoint_capacity(checkpoint, 7)
    state = expanded["spiral_and_transform"]
    expanded_logits = state["gap_expander_params.logits"]
    old_width = logits.shape[-1]

    torch.testing.assert_close(expanded_logits[..., :old_width], logits)
    assert torch.count_nonzero(expanded_logits[..., old_width:]) == 0
    assert state["gap_expander_params.winding_first_logit_idx"].numel() == 7
    for name in ("exp_avg", "exp_avg_sq"):
        moment = expanded["optimiser"]["state"][20][name]
        torch.testing.assert_close(moment[..., :old_width],
                                   checkpoint["optimiser"]["state"][20][name])
        assert torch.count_nonzero(moment[..., old_width:]) == 0
    assert expanded["cfg"]["model_gap_expander_capacity_windings"] == 7
