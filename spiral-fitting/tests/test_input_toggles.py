
import pytest
import torch

from config import Config, FitConfig
from fit_session import (PclInputSpec, PclRole, ScrollSpec, SpiralInputPaths,
                         conventional_input_paths,
                         conventional_pcl_document_path)
from fit_spiral import FitContext
import losses


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
        fibers="/inputs/fibers",
        tracks_dbm="/inputs/tracks.dbm",
        normal_x="/inputs/nx.zarr",
        normal_y="/inputs/ny.zarr",
        gradient_magnitude="/inputs/grad.zarr",
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

    assert context.tracks_dbm_path is None
    assert context.normal_nx_zarr_path is None
    assert context.normal_ny_zarr_path is None
    assert context.grad_mag_zarr_path is None
    assert context.winding_inference_path is None
    assert context.shell_path is None
    assert context.config["loss_weight_track_radius"] == 73.0
    assert context.config["sample_count_tracks_per_step"] == 1234


def test_conventional_pcl_document_path_matches_dataset_resolution(tmp_path):
    (tmp_path / "same_windings.json").write_text("{}")
    assert conventional_pcl_document_path(tmp_path, PclRole.SAME_WINDING) == \
        str((tmp_path / "same_windings.json").resolve())
    assert conventional_pcl_document_path("", "same_winding") == ""


def test_cli_paths_name_no_patch_fiber_or_pcl_files(tmp_path):
    # The CLI reads patches, fibers and point collections from an object
    # store; the conventional dataset layout no longer names them.
    paths = conventional_input_paths(
        tmp_path, ScrollSpec(name="test", voxel_size_um=1.0,
                             spiral_outward_sense="CW"))
    assert paths.verified_patches == ""
    assert paths.fibers == ""
    assert paths.pcls == ()
    assert paths.umbilicus == f"{tmp_path}/umbilicus.json"


def test_empty_patch_sets_have_no_sampling_distribution():
    context = object.__new__(FitContext)
    assert context._patch_sampling_probabilities([]) is None


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
