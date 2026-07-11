import pytest

from koine_machines.training.train import (
    _disable_z_projection_for_normal_pooled_3d,
    _full_3d_dilation_distances_for_level,
)


def test_full_3d_dilation_distances_scale_with_native_volume_level():
    config = {
        "full_3d": {
            "label_dilation_distance": 4.0,
            "supervision_dilation_distance": 8.0,
        },
        "datasets": [{"volume_scale": "2"}, {"volume_scale": 2}],
    }

    assert _full_3d_dilation_distances_for_level(config) == (1.0, 2.0)


def test_full_3d_dilation_distances_disabled_without_reading_datasets():
    assert _full_3d_dilation_distances_for_level({"full_3d": {}}) == (0.0, 0.0)


def test_full_3d_dilation_distances_reject_mixed_volume_scales():
    config = {
        "full_3d": {"label_dilation_distance": 4.0},
        "datasets": [{"volume_scale": "0"}, {"volume_scale": "2"}],
    }

    with pytest.raises(ValueError, match="single volume_scale"):
        _full_3d_dilation_distances_for_level(config)


def test_disable_z_projection_for_normal_pooled_3d_forces_projection_off():
    config = {
        "mode": "normal_pooled_3d",
        "model_config": {"z_projection_mode": "logsumexp"},
        "targets": {
            "ink": {
                "z_projection_mode": "mean",
                "z_projection": {
                    "mode": "learned_mlp",
                    "z_projection_mlp_hidden": 64,
                },
            },
            "aux": {
                "z_projection_mode": "max",
            },
        },
    }

    _disable_z_projection_for_normal_pooled_3d(config)

    assert config["model_config"]["z_projection_mode"] == "none"
    assert config["targets"]["ink"]["z_projection_mode"] == "none"
    assert config["targets"]["ink"]["z_projection"]["mode"] == "none"
    assert config["targets"]["aux"]["z_projection_mode"] == "none"
