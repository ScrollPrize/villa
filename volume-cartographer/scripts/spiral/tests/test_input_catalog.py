"""The declarative fit-input catalog shared by validation and planning."""

from config import Config
from fit_session import (FIT_INPUT_CATALOG, PCL_ROLE_CONVENTIONS, PclRole,
                         SCROLL_SPEC_PATH_OVERRIDE_KEYS, SpiralInputPaths,
                         fit_input, input_source_enabled, pcl_input_enabled,
                         phase_bundle_enabled, winding_inference_enabled)


def test_catalog_covers_every_fit_input_path_field():
    keys = {spec.key for spec in FIT_INPUT_CATALOG}
    input_fields = set(SpiralInputPaths.__dataclass_fields__) - {
        # Not fit inputs: identity, deployment, and resume locations.
        "dataset_root", "scroll_zarr", "checkpoint",
        "output_directory", "cache_directory",
    }
    assert keys == input_fields
    assert set(SCROLL_SPEC_PATH_OVERRIDE_KEYS) == keys - {"pcls"}


def test_outer_shell_is_required_by_shell_losses_or_winding_model():
    spec = fit_input("outer_shell")
    assert spec.kind == "directory"
    # Required by either shell loss weight (the outer weight defaults on) or
    # by winding-model supervision even when both shell losses are disabled.
    assert spec.required({}) is True
    assert spec.required({"loss_weight_shell_outer": 0.0,
                          "loss_weight_shell_patch_radius": 0.0}) is False
    assert spec.required({"loss_weight_shell_outer": 0.0,
                          "loss_weight_shell_patch_radius": 2.0}) is True
    assert spec.required({"dense_spacing_mode": "winding_model",
                          "loss_weight_shell_outer": 0.0,
                          "loss_weight_shell_patch_radius": 0.0}) is True


def test_no_input_path_is_advertised_as_takeable_by_a_resident_session():
    assert Config.catalog()["schema"]["paths"] == {}


def test_model_configuration_is_a_new_fit_change():
    assert not any(spec.checkpoint_domain for spec in FIT_INPUT_CATALOG)
    schema = Config.catalog()["schema"]
    fields = schema["fields"]
    assert schema["run_fields"]["z_begin"]["runtime_impact"] == "new_fit"
    assert fields["model_num_flow_stages"]["runtime_impact"] == "new_fit"


def test_lasagna_store_predicates_reproduce_the_mode_contract():
    # phase (the default) requires normals and the SDT even at zero
    # sub-weights; grad_mag requires the gradient store only with a positive
    # spacing weight and never the SDT; an invalid mode enables nothing.
    assert fit_input("normal_x").required({}) is True
    assert fit_input("surf_sdt").required({}) is True
    assert fit_input("gradient_magnitude").required({}) is False

    grad = {"dense_spacing_mode": "grad_mag",
            "loss_weight_dense_normals": 0.0}
    assert fit_input("gradient_magnitude").required(grad) is True
    assert fit_input("surf_sdt").required(grad) is False
    assert fit_input("normal_x").required(grad) is False
    assert fit_input("gradient_magnitude").required(
        {**grad, "loss_weight_dense_spacing": 0.0}) is False

    winding_model = {"dense_spacing_mode": "winding_model",
                     "loss_weight_dense_normals": 0.0}
    assert fit_input("winding_inference").required(winding_model) is True
    assert fit_input("winding_inference").enabled(winding_model) is True
    assert fit_input("normal_x").required(winding_model) is False
    assert fit_input("surf_sdt").required(winding_model) is False

    invalid = {"dense_spacing_mode": "crossing_count",
               "loss_weight_dense_normals": 0.0}
    assert not any(fit_input(key).required(invalid)
                   for key in ("normal_x", "normal_y",
                               "gradient_magnitude", "surf_sdt",
                               "winding_inference"))


def test_patch_inputs_follow_the_disable_switch():
    verified = fit_input("verified_patches")
    unverified = fit_input("unverified_patches")
    # Dataset discovery cannot know the initialization config yet; validation
    # enforces this only when the default-on source is actually selected.
    assert verified.resolve_required is False
    assert verified.required({}) is True
    assert verified.required({"input_disable_patches": True}) is False
    # A disabled source is not validated at all.
    assert verified.enabled({"input_disable_patches": True}) is False
    assert unverified.enabled({"input_disable_patches": True}) is False
    assert unverified.enabled({}) is True
    assert unverified.required({}) is False

    assert verified.enabled({"input_use_verified_patches": False}) is False
    assert unverified.enabled({"input_use_unverified_patches": False}) is False


def test_source_toggles_and_dependencies_are_centralized():
    assert input_source_enabled({}, "tracks_dbm") is True
    assert input_source_enabled({"input_use_tracks": False}, "tracks_dbm") is False
    assert fit_input("tracks_dbm").enabled({"input_use_tracks": False}) is False

    assert phase_bundle_enabled({"dense_spacing_mode": "phase"}) is True
    assert phase_bundle_enabled({
        "dense_spacing_mode": "phase", "input_use_normals": False,
    }) is False
    assert phase_bundle_enabled({
        "dense_spacing_mode": "phase", "input_use_surf_sdt": False,
    }) is False

    winding = {"dense_spacing_mode": "winding_model"}
    assert winding_inference_enabled(winding) is True
    assert winding_inference_enabled({
        **winding, "input_use_winding_inference": False,
    }) is False
    assert winding_inference_enabled({
        **winding, "input_use_outer_shell": False,
    }) is False


def test_pcl_role_toggles_include_legacy_role_inference():
    for role in PclRole:
        key = f"input_use_pcl_{role.value}"
        assert pcl_input_enabled({}, role) is True
        assert pcl_input_enabled({key: False}, role) is False

    assert pcl_input_enabled(
        {"input_use_pcl_absolute": False}, None, "/data/abs_winding.json") is False
    assert pcl_input_enabled(
        {"input_use_pcl_relative": False}, None, "/data/legacy.json") is False
    # Absolute PCLs cascade off when their verified-patch prerequisite is off.
    assert pcl_input_enabled(
        {"input_use_verified_patches": False}, PclRole.ABSOLUTE) is False
    # Non-absolute inputs may still become unattached-strip supervision.
    assert pcl_input_enabled(
        {"input_use_verified_patches": False}, PclRole.RELATIVE) is True


def test_every_pcl_role_has_one_conventional_file():
    roles = {role.value: filename for role, filename in PCL_ROLE_CONVENTIONS}
    # One filename per role, serving both discovery and commit; the set is
    # exactly the role vocabulary, so no role can be uploaded without a
    # commit target and none is silently undiscoverable.
    assert roles == {
        "absolute": "abs_winding.json",
        "relative": "relative_windings.json",
        "same_winding": "same_windings.json",
        "drawn_control_points": "drawn_control_points.json",
    }
    assert set(roles) == {role.value for role in PclRole}
    assert len(set(roles.values())) == len(roles)
