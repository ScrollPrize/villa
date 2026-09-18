import pytest
import torch
from config import Config, FitConfig
from fit_session import PclInputSpec, PclRole, ScrollSpec, SpiralInputPaths
from fit_spiral import FitContext
import losses
from fit_session import (
    fit_input,
    input_source_enabled,
    pcl_input_enabled,
    phase_bundle_enabled,
    winding_inference_enabled,
)


def make_context(config, paths):
    return FitContext(
        FitConfig(config),
        scroll=ScrollSpec(name="test", voxel_size_um=1.0,
                          spiral_outward_sense="CW"),
        paths=paths,
    )


def test_disabled_sources_are_removed_before_any_loader_can_see_them():
    disabled = {
        key: False
        for key in Config().as_dict()
        if key.startswith("input_use_")
    }
    config = Config({
        **disabled,
        "dense_spacing_mode": "winding_model",
        # These remain deliberately nonzero: toggles gate execution without
        # destroying the user's tuning.
        "loss_weight_track_radius": 73.0,
        "sample_count_tracks_per_step": 1234,
    }).as_dict()
    paths = SpiralInputPaths(
        umbilicus="/inputs/umbilicus.json",
        verified_patches="/inputs/verified",
        unverified_patches="/inputs/unverified",
        fibers="/inputs/fibers",
        tracks_dbm="/inputs/tracks.dbm",
        normal_x="/inputs/nx.zarr",
        normal_y="/inputs/ny.zarr",
        gradient_magnitude="/inputs/grad.zarr",
        surf_sdt="/inputs/sdt.zarr",
        winding_inference="/inputs/winding",
        outer_shell="/inputs/shell",
        pcls=(
            PclInputSpec("/inputs/absolute.json", PclRole.ABSOLUTE),
            PclInputSpec("/inputs/relative.json", PclRole.RELATIVE),
            PclInputSpec("/inputs/same.json", PclRole.SAME_WINDING),
            PclInputSpec("/inputs/drawn.json", PclRole.DRAWN_CONTROL_POINTS),
        ),
    )

    context = make_context(config, paths)

    assert context.verified_patches_path is None
    assert context.unverified_patches_path is None
    assert context.fibers_path is None
    assert context.tracks_dbm_path is None
    assert context.normal_nx_zarr_path is None
    assert context.normal_ny_zarr_path is None
    assert context.grad_mag_zarr_path is None
    assert context.surf_sdt_zarr_path is None
    assert context.winding_inference_path is None
    assert context.shell_path is None
    assert context.pcl_input_specs == []
    assert context._configured_pcl_sources == paths.pcls
    assert context.config["loss_weight_track_radius"] == 73.0
    assert context.config["sample_count_tracks_per_step"] == 1234


def test_pcl_role_toggles_filter_documents_independently():
    config = Config({
        "input_use_pcl_relative": False,
        "input_use_pcl_drawn_control_points": False,
    }).as_dict()
    paths = SpiralInputPaths(
        umbilicus="/inputs/umbilicus.json",
        verified_patches="/inputs/verified",
        pcls=(
            PclInputSpec("/inputs/absolute.json", PclRole.ABSOLUTE),
            PclInputSpec("/inputs/relative.json", PclRole.RELATIVE),
            PclInputSpec("/inputs/same.json", PclRole.SAME_WINDING),
            PclInputSpec("/inputs/drawn.json", PclRole.DRAWN_CONTROL_POINTS),
        ),
    )

    context = make_context(config, paths)

    assert context.pcl_input_specs == [
        ("/inputs/absolute.json", "absolute"),
        ("/inputs/same.json", "same_winding"),
    ]


def test_disabling_normals_skips_the_normal_loss_graph(monkeypatch):
    class IdentityTransform:
        def inv(self, points):
            return points

    monkeypatch.setattr(
        losses, 'get_radial_normal_in_scroll_space',
        lambda *args, **kwargs: pytest.fail('normal loss graph was constructed'))
    volume = {
        'backend': 'dense_test',
        'volume': torch.ones([3, 4, 8, 8], dtype=torch.uint8),
        'shape': (4, 8, 8),
        'z_origin': 0,
        'y_origin': 0,
        'x_origin': 0,
        'lasagna_scale': 1,
    }

    values = list(losses.iter_lasagna_losses(
        IdentityTransform(), torch.tensor(1.0), volume, 2, 8,
        compute_spacing=True, compute_normals=False,
        cfg=Config().as_dict(), z_begin=1, z_end=3))

    assert [name for name, _ in values] == ['dense_spacing']


def test_outer_shell_is_required_by_shell_losses_or_winding_model():
    spec = fit_input("outer_shell")
    assert spec.kind == "directory"
    # Required by either shell loss weight (the outer weight defaults on) or
    # by winding-model supervision even when both shell losses are disabled.
    assert spec.required({}) is True
    assert spec.required({"dense_spacing_mode": "phase",
                          "loss_weight_shell_outer": 0.0,
                          "loss_weight_shell_patch_radius": 0.0}) is False
    assert spec.required({"loss_weight_shell_outer": 0.0,
                          "loss_weight_shell_patch_radius": 2.0}) is True
    assert spec.required({"dense_spacing_mode": "winding_model",
                          "loss_weight_shell_outer": 0.0,
                          "loss_weight_shell_patch_radius": 0.0}) is True


def test_lasagna_store_predicates_reproduce_the_mode_contract():
    # Phase requires normals and the SDT even at zero
    # sub-weights; grad_mag requires the gradient store only with a positive
    # spacing weight and never the SDT; an invalid mode enables nothing.
    phase = {"dense_spacing_mode": "phase"}
    assert fit_input("normal_x").required(phase) is True
    assert fit_input("surf_sdt").required(phase) is True
    assert fit_input("gradient_magnitude").required(phase) is False

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
