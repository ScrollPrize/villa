import json

import pytest

from config import Config


def test_catalog_is_complete_and_presets_are_resolved():
    catalog = Config.catalog()
    assert set(catalog["defaults"]) == set(catalog["schema"]["fields"])
    for preset in catalog["presets"].values():
        assert set(preset) == set(catalog["defaults"])


def test_every_key_has_generated_metadata():
    catalog = Config.catalog()
    required = {
        "type", "nullable", "label", "runtime_impact", "dependencies"}
    for key, field in catalog["schema"]["fields"].items():
        assert required <= set(field)
        assert field["label"] == key.split("_", 1)[1].replace("_", " ").title()


def test_interactive_runtime_impacts_match_resident_capabilities():
    schema = Config.catalog()["schema"]
    fields = schema["fields"]
    for key, field in fields.items():
        if key.startswith("patch_"):
            expected = (
                "prepared_input_rebuild"
                if key == "patch_erode_patches" else "run_boundary")
            assert field["runtime_impact"] == expected
        if key.startswith("dense_"):
            expected = (
                "prepared_input_rebuild"
                if key == "dense_spacing_mode" else "run_boundary")
            assert field["runtime_impact"] == expected
        if key.startswith("dt_"):
            assert field["runtime_impact"] == "run_boundary"
        if key.startswith("shell_"):
            assert field["runtime_impact"] == "shell_reload"
    assert schema["paths"]["outer_shell"] == {
        "runtime_impact": "shell_reload",
        "dependencies": ["shell"],
    }

    mutable_tracks = {
        "track_min_sample_spacing", "track_max_sample_spacing",
        "track_length_bin_weights", "track_max_tortuosity",
        "track_max_track_crossing_per_step",
        "track_min_walk_steps_per_track", "track_max_walk_steps_per_track",
        "track_n_walks_per_track", "track_walk_require_loop_consistency",
        "track_radius_target", "track_radius_loss_margin",
        "track_radius_within_norm_p", "track_dt_within_track_norm_p",
        "track_dt_norm_p", "track_dt_loss_margin",
    }
    assert all(fields[key]["runtime_impact"] == "run_boundary"
               for key in mutable_tracks)
    assert all(fields[key]["runtime_impact"] == "prepared_input_rebuild"
               for key in {
                   "track_crossing_precompute_max", "track_crossing_mode",
                   "track_exclusion_radius",
               })


def test_mapping_and_json_overrides_and_validation(tmp_path):
    changed = Config({"optimizer_learning_rate": 0.25})
    assert changed.optimizer_learning_rate == 0.25
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"optimizer_learning_rate": 0.5}))
    assert Config(profile).optimizer_learning_rate == 0.5

    with pytest.raises(ValueError, match="Unknown"):
        Config({"not_a_setting": 1})
    with pytest.raises(ValueError, match="Invalid value"):
        Config({"optimizer_learning_rate": "fast"})
    with pytest.raises(ValueError, match="Out-of-range"):
        Config({"optimizer_learning_rate": -1})
    with pytest.raises(ValueError, match="Invalid value"):
        Config({"dense_spacing_mode": "unknown"})
    with pytest.raises(ValueError, match="Invalid vector length"):
        Config({"dense_spacing_pair_m_short": [1]})
    with pytest.raises(ValueError):
        Config({"track_max_tortuosity": "unlimited"})
