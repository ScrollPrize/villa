"""The declarative fit-input catalog shared by validation and planning."""

from config import Config
from fit_session import (FIT_INPUT_CATALOG, PCL_ROLE_CONVENTIONS, PclRole,
                         SCROLL_SPEC_PATH_OVERRIDE_KEYS, SpiralInputPaths,
                         fit_input)


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
    # Enabled by either shell loss weight; the outer weight defaults on.
    assert spec.required({}) is True
    assert spec.required({"loss_weight_shell_outer": 0.0,
                          "loss_weight_shell_patch_radius": 0.0}) is False
    assert spec.required({"loss_weight_shell_outer": 0.0,
                          "loss_weight_shell_patch_radius": 2.0}) is True


def test_no_input_path_is_advertised_as_takeable_by_a_resident_session():
    assert Config.catalog()["schema"]["paths"] == {}


def test_model_configuration_is_a_new_fit_change():
    assert not any(spec.checkpoint_domain for spec in FIT_INPUT_CATALOG)
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
    assert verified.required({}) is True
    assert verified.required({"input_disable_patches": True}) is False
    # verified_patches is still validated (as optional) when disabled;
    # unverified_patches drops out of validation entirely.
    assert verified.enabled({"input_disable_patches": True}) is True
    assert unverified.enabled({"input_disable_patches": True}) is False
    assert unverified.enabled({}) is True
    assert unverified.required({}) is False


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
