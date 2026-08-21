from unittest import mock

import pytest
import torch

from config import Config, FitConfig
from fit_session import PclInputSpec, PclRole, ScrollSpec, SpiralInputPaths
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


@pytest.mark.parametrize(('override', 'record', 'message'), [
    ('input_use_verified_patches',
     {'kind': 'patch', 'id': 'p', 'path': '/unused'},
     'verified-patch inputs are disabled'),
    ('input_use_fibers',
     {'kind': 'fiber', 'id': 'f', 'path': '/unused'},
     'fiber inputs are disabled'),
    ('input_use_pcl_same_winding',
     {'kind': 'pcl', 'id': 'c', 'role': 'same_winding', 'path': '/unused'},
     'same_winding PCL inputs are disabled'),
])
def test_disabled_interactive_inputs_are_rejected_before_loading(
    override, record, message,
):
    context = object.__new__(FitContext)
    context.config = FitConfig(Config({override: False}).as_dict())
    context.clear_interactive_influence = lambda: None
    with mock.patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
            mock.patch.object(torch.cuda, 'set_rng_state_all'):
        with pytest.raises(RuntimeError, match=message):
            context.incorporate_interactive_inputs(
                [record], current_iteration=0, target_iteration=1)
