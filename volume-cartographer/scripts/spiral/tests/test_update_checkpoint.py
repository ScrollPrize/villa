from collections import OrderedDict

import pytest
import torch

from config import Config
from update_checkpoint import migrate_config, update_checkpoint


def test_migrate_legacy_config_to_exact_current_schema():
    migrated, renamed, removed, added = migrate_config({
        "flow_bounds_radius": 1234,
        "num_patches_per_step": 77,
        "unverified_patch_exclusion_radius": 12.5,
        "learning_rate": 2.25e-6,
        "use_tracks": True,
        "track_walk_require_loop_consistency": False,
    })

    assert set(migrated) == set(Config().as_dict())
    assert migrated["model_flow_bounds_radius"] == 1234
    assert migrated["sample_count_patches_per_step"] == 77
    assert migrated["patch_unverified_patch_exclusion_radius"] == 12.5
    assert migrated["optimizer_learning_rate"] == 2.25e-6
    assert migrated["input_use_tracks"] is True
    assert "use_tracks -> input_use_tracks" in renamed
    assert "track_walk_require_loop_consistency" in removed
    assert "flow_bounds_radius -> model_flow_bounds_radius" in renamed
    assert "influence_enabled" in added


def test_update_checkpoint_preserves_tensor_objects_and_migrates_snapshots():
    model_state = OrderedDict(weight=torch.tensor([1.0, 2.0]))
    optimiser = {"state": {0: {"step": torch.tensor(3)}}}
    checkpoint = {
        "spiral_and_transform": model_state,
        "optimiser": optimiser,
        "cfg": {"flow_bounds_radius": 3200},
    }

    updated, reports = update_checkpoint(checkpoint)

    assert updated["spiral_and_transform"] is model_state
    assert updated["optimiser"] is optimiser
    assert updated["cfg"]["model_flow_bounds_radius"] == 3200
    assert updated["requested_config"] == updated["cfg"]
    assert updated["resolved_config"] == updated["cfg"]
    assert reports["cfg"]["renamed"]


def test_unknown_legacy_config_key_is_not_silently_dropped():
    with pytest.raises(ValueError, match="no known migration.*mystery"):
        migrate_config({"mystery": 1})
