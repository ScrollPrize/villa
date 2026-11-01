
import pytest
import torch

from config import Config, FitConfig
from fit_session import ScrollSpec, SpiralInputPaths
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
        lasagna="/inputs/s.lasagna.json",
        winding_inference="/inputs/winding",
    )

    context = make_context(config, paths)

    assert context.lasagna_source is None
    assert context.winding_inference_path is None
    assert context.config["loss_weight_track_radius"] == 73.0
    assert context.config["sample_count_tracks_per_step"] == 1234


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
