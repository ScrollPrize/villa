import inspect
import json

import pytest

from config import Config, FitConfig, MODEL_STAGE_KEYS, rebuild_stage


def test_fit_config_wraps_a_resolved_mapping_with_dict_style_access():
    values = Config().as_dict()
    fit_config = FitConfig(values)
    assert fit_config["optimizer_random_seed"] == values["optimizer_random_seed"]
    assert fit_config.get("missing_key", 42) == 42
    assert "optimizer_random_seed" in fit_config
    assert dict(fit_config) == values

    fit_config.update({"optimizer_random_seed": 7})
    assert fit_config["optimizer_random_seed"] == 7
    # Construction copied the mapping: the caller's dict is untouched.
    assert values["optimizer_random_seed"] == 1


def test_catalog_is_complete_and_presets_are_resolved():
    catalog = Config.catalog()
    assert set(catalog["defaults"]) == set(catalog["schema"]["fields"])
    for preset in catalog["presets"].values():
        assert set(preset) == set(catalog["defaults"])


def test_every_key_has_generated_metadata():
    catalog = Config.catalog()
    required = {"type", "nullable", "label", "runtime_impact"}
    for key, field in catalog["schema"]["fields"].items():
        assert required <= set(field)
        assert field["label"] == key.split("_", 1)[1].replace("_", " ").title()


def test_input_participation_toggles_are_rebuild_scoped_booleans():
    catalog = Config.catalog()
    expected = {
        "input_use_verified_patches", "input_use_unverified_patches",
        "input_use_tracks", "input_use_fibers", "input_use_pcl_absolute",
        "input_use_pcl_relative", "input_use_pcl_same_winding",
        "input_use_pcl_drawn_control_points", "input_use_normals",
        "input_use_surf_sdt", "input_use_gradient_magnitude",
        "input_use_winding_inference", "input_use_outer_shell",
    }
    assert {key for key in catalog["defaults"]
            if key.startswith("input_use_")} == expected
    for key in expected:
        assert catalog["defaults"][key] is True
        assert catalog["schema"]["fields"][key]["type"] == "boolean"
        assert catalog["schema"]["fields"][key]["runtime_impact"] == "new_fit"
        assert catalog["schema"]["fields"][key]["description"]


def test_interactive_runtime_impacts_match_resident_capabilities():
    schema = Config.catalog()["schema"]
    fields = schema["fields"]
    for key, field in fields.items():
        if key.startswith("patch_"):
            expected = (
                "new_fit"
                if key in {"patch_erode_patches", "patch_2d_sampling_max_area",
                           "patch_uuid_filter_regex"} else "run_boundary")
            assert field["runtime_impact"] == expected
        if key.startswith("dense_"):
            expected = (
                "new_fit"
                if key == "dense_spacing_mode" else "run_boundary")
            assert field["runtime_impact"] == expected
        if key.startswith("dt_"):
            assert field["runtime_impact"] == "run_boundary"
        if key.startswith("shell_"):
            expected = (
                "new_fit"
                if key in {"shell_num_theta_bins",
                           "shell_table_smooth_sigma_z",
                           "shell_table_smooth_sigma_theta",
                           "shell_min_confidence"}
                else "run_boundary")
            assert field["runtime_impact"] == expected
    # Input identities and shell-atlas construction are fixed for a resident
    # session; ordinary shell loss settings remain run-mutable.
    assert schema["paths"] == {}

    mutable_tracks = {
        "track_min_sample_spacing", "track_max_sample_spacing",
        "track_length_bin_weights", "track_max_tortuosity",
        "track_max_track_crossing_per_step",
        "track_min_walk_steps_per_track", "track_max_walk_steps_per_track",
        "track_min_walks_per_track", "track_max_walks_per_track",
        "track_walk_minimum_cycle_travel",
        "track_radius_target", "track_radius_loss_margin",
        "track_radius_within_norm_p", "track_dt_within_track_norm_p",
        "track_dt_norm_p", "track_dt_loss_margin",
    }
    assert all(fields[key]["runtime_impact"] == "run_boundary"
               for key in mutable_tracks)
    assert all(fields[key]["runtime_impact"] == "new_fit"
               for key in {
                   "track_crossing_precompute_max", "track_crossing_mode",
                   "track_exclusion_radius",
               })


def test_rebuild_stage_is_model_only_for_the_allowlist():
    assert rebuild_stage([]) == "model"
    assert rebuild_stage(["model_num_flow_integration_steps"]) == "model"
    assert rebuild_stage(["model_num_flow_integration_steps",
                          "model_linear_z_resolution"]) == "model"
    # One unlisted key in the set is enough to demand the whole build.
    assert rebuild_stage(["model_num_flow_integration_steps",
                          "z_begin"]) == "all"
    assert rebuild_stage(["optimizer_random_seed"]) == "all"
    assert rebuild_stage(["model_flow_bounds_z_margin"]) == "all"
    # Unaudited/unknown keys fail safe rather than raising.
    assert rebuild_stage(["not_a_setting"]) == "all"


def test_the_allowlist_is_a_subset_of_the_new_fit_settings():
    fields = Config.catalog()["schema"]["fields"]
    assert MODEL_STAGE_KEYS <= set(fields)
    assert all(fields[key]["runtime_impact"] == "new_fit"
               for key in MODEL_STAGE_KEYS)


def test_no_allowlisted_key_is_named_while_loading_host_inputs():
    # The mechanism behind the allowlist's promise: a model-stage rebuild
    # retains whatever host preparation produced, so a key host preparation
    # reads cannot be on the list. This is exactly the leak
    # model_flow_bounds_z_margin (the host-side ShellPolarMap) and
    # optimizer_random_seed (the host RNG seeding) would have introduced.
    import fit_spiral

    context = fit_spiral.FitContext
    source = "".join(inspect.getsource(member) for member in (
        context.load_host_inputs,
        context._load_patches_from_dir,
        context._prepare_patch_sampling_cache,
        context._rebuild_pcl_sampling_strata,
    ))
    # The positive control: both keys the audit disqualified are named here,
    # so a scan that stops matching fails rather than silently passing.
    assert "model_flow_bounds_z_margin" in source
    assert "optimizer_random_seed" in source
    assert not [key for key in MODEL_STAGE_KEYS if key in source]


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
