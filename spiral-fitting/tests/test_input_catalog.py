"""The declarative fit-input catalog shared by validation and planning."""

from config import Config
from fit_session import (FIT_INPUT_CATALOG, INPUT_MANIFEST_KEYS,
                         SpiralInputPaths, fit_input, input_source_enabled,
                         lasagna_packs_required, outer_shell_required,
                         winding_inference_enabled)


def test_catalog_covers_every_fit_input_path_field():
    keys = {spec.key for spec in FIT_INPUT_CATALOG}
    input_fields = set(SpiralInputPaths.__dataclass_fields__) - {
        # Not fit inputs: identity, provenance, deployment, and resume
        # locations.
        "volume_uuid", "provenance", "scroll_zarr", "checkpoint",
        "output_directory", "cache_directory",
    }
    assert keys == input_fields
    # The command-line manifest names every one of them.
    assert keys <= set(INPUT_MANIFEST_KEYS)
    # Patches, fibers, point collections, tracks and the outer shell come
    # from the object store.
    assert not {"verified_patches", "fibers", "pcls", "tracks", "tracks_dbm",
                "outer_shell"} & (keys | set(INPUT_MANIFEST_KEYS))


def test_omitted_spacing_mode_uses_fitter_default():
    explicit = {"dense_spacing_mode": Config().dense_spacing_mode}
    for spec in FIT_INPUT_CATALOG:
        assert spec.enabled({}) == spec.enabled(explicit), spec.key
        assert spec.required({}) == spec.required(explicit), spec.key


def test_outer_shell_is_required_by_shell_losses_or_winding_model():
    # The store's tagged surface, not a dataset path: required by the outer
    # shell loss weight (which defaults on) or by winding-model supervision
    # even when the shell loss is disabled.
    assert "outer_shell" not in {spec.key for spec in FIT_INPUT_CATALOG}
    assert outer_shell_required({}) is True
    assert outer_shell_required({"dense_spacing_mode": "phase",
                                 "loss_weight_shell_outer": 0.0}) is False
    assert outer_shell_required({"dense_spacing_mode": "winding_model",
                                 "loss_weight_shell_outer": 0.0}) is True


def test_no_input_path_is_advertised_as_takeable_by_a_resident_session():
    assert Config.catalog()["schema"]["paths"] == {}


def test_model_configuration_is_a_new_fit_change():
    assert not any(spec.checkpoint_domain for spec in FIT_INPUT_CATALOG)
    schema = Config.catalog()["schema"]
    fields = schema["fields"]
    assert schema["run_fields"]["z_begin"]["runtime_impact"] == "new_fit"
    assert fields["model_num_flow_stages"]["runtime_impact"] == "new_fit"


def test_lasagna_store_predicates_reproduce_the_mode_contract():
    # grad_mag requires the gradient pack only with a positive spacing
    # weight; normals are required only with a positive normals weight; an
    # invalid mode enables nothing.
    grad = {"dense_spacing_mode": "grad_mag",
            "loss_weight_dense_normals": 0.0}
    assert fit_input("lasagna").required(grad) is True
    assert lasagna_packs_required(grad) == ["grad_mag"]
    assert lasagna_packs_required(
        {**grad, "loss_weight_dense_normals": 100.0}) == ["normals", "grad_mag"]
    assert fit_input("lasagna").required(
        {**grad, "loss_weight_dense_spacing": 0.0}) is False

    winding_model = {"dense_spacing_mode": "winding_model",
                     "loss_weight_dense_normals": 0.0}
    assert fit_input("winding_inference").required(winding_model) is True
    assert fit_input("winding_inference").enabled(winding_model) is True
    assert fit_input("lasagna").required(winding_model) is False

    invalid = {"dense_spacing_mode": "phase",
               "loss_weight_dense_normals": 0.0}
    assert not any(fit_input(key).required(invalid)
                   for key in ("lasagna", "winding_inference"))


def test_source_toggles_and_dependencies_are_centralized():
    assert input_source_enabled({}, "tracks") is True
    assert input_source_enabled({"input_use_tracks": False}, "tracks") is False
    assert fit_input("tracks_dbm") is None

    winding = {"dense_spacing_mode": "winding_model"}
    assert winding_inference_enabled(winding) is True
    assert winding_inference_enabled({
        **winding, "input_use_winding_inference": False,
    }) is False
    assert winding_inference_enabled({
        **winding, "input_use_outer_shell": False,
    }) is False
