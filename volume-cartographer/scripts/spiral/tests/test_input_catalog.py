"""The declarative fit-input catalog shared by validation and planning."""

from config import Config
from fit_session import (FIT_INPUT_CATALOG, FULL_REBUILD_DEPENDENCIES,
                         PCL_ROLE_CONVENTIONS, SCROLL_SPEC_PATH_OVERRIDE_KEYS,
                         SpiralInputPaths, fit_input, input_change_impact,
                         input_path_schema)


def test_catalog_covers_every_fit_input_path_field():
    keys = {spec.key for spec in FIT_INPUT_CATALOG}
    input_fields = set(SpiralInputPaths.__dataclass_fields__) - {
        # Not fit inputs: identity, deployment, and resume locations.
        "dataset_root", "scroll_zarr", "checkpoint",
        "output_directory", "cache_directory",
    }
    assert keys == input_fields
    assert set(SCROLL_SPEC_PATH_OVERRIDE_KEYS) == keys - {"pcls"}


def test_outer_shell_is_an_ordinary_entry_with_the_shell_weight_predicate():
    spec = fit_input("outer_shell")
    assert spec.kind == "directory"
    assert spec.runtime_impact == "new_fit"
    assert spec.dependencies == FULL_REBUILD_DEPENDENCIES
    # Enabled by either shell loss weight; the outer weight defaults on.
    assert spec.required({}) is True
    assert spec.required({"loss_weight_shell_outer": 0.0,
                          "loss_weight_shell_patch_radius": 0.0}) is False
    assert spec.required({"loss_weight_shell_outer": 0.0,
                          "loss_weight_shell_patch_radius": 2.0}) is True


def test_every_input_path_change_requires_a_full_host_rebuild():
    assert input_change_impact("outer_shell") == (
        "new_fit", list(FULL_REBUILD_DEPENDENCIES))
    assert input_change_impact("verified_patches") == (
        "new_fit", list(FULL_REBUILD_DEPENDENCIES))
    assert input_change_impact("checkpoint") == (
        "new_fit", list(FULL_REBUILD_DEPENDENCIES))


def test_config_catalog_paths_derive_from_the_input_catalog():
    assert input_path_schema() == {}
    assert Config.catalog()["schema"]["paths"] == input_path_schema()


def test_input_paths_and_model_configuration_are_new_fit_changes():
    assert not any(spec.checkpoint_domain for spec in FIT_INPUT_CATALOG)
    assert all(input_change_impact(spec.key)[0] == "new_fit"
               for spec in FIT_INPUT_CATALOG)
    fields = Config.catalog()["schema"]["fields"]
    assert fields["z_begin"]["runtime_impact"] == "new_fit"
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

    invalid = {"dense_spacing_mode": "crossing_count",
               "loss_weight_dense_normals": 0.0}
    assert not any(fit_input(key).required(invalid)
                   for key in ("normal_x", "normal_y",
                               "gradient_magnitude", "surf_sdt"))


def test_patch_inputs_follow_the_disable_switch():
    verified = fit_input("verified_patches")
    unverified = fit_input("unverified_patches")
    assert verified.required({}) is True
    assert verified.required({"input_disable_patches": True}) is False
    # verified_patches is still validated (as optional) when disabled;
    # unverified_patches drops out of validation entirely.
    assert verified.enabled({"input_disable_patches": True}) is True
    assert unverified.enabled({"input_disable_patches": True}) is False
    assert unverified.enabled({}) is True
    assert unverified.required({}) is False


def test_pcl_role_conventions_carry_discovery_flags():
    roles = {role.value: (filename, discovered)
             for role, filename, discovered in PCL_ROLE_CONVENTIONS}
    assert roles["absolute"] == ("abs_winding.json", True)
    # Conventional for the headless CLI, but not probed by dataset
    # resolution (the historical _PCL_ENTRIES omission).
    assert roles["patch_overlap"] == ("patch-overlap-pcls.json", False)
