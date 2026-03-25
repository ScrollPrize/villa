from koine_machines.training.train import _disable_z_projection_for_normal_pooled_3d


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
