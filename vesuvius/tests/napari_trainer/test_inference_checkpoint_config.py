"""The napari trainer's inference builds the network a checkpoint was trained with."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("napari")

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.napari_trainer.inference import ModelLoader


def _manager(features, targets):
    model_config = {
        "features_per_stage": features,
        "n_blocks_per_stage": [1] * len(features),
        "n_conv_per_stage_decoder": [1] * (len(features) - 1),
        "kernel_sizes": [[3, 3]] * len(features),
        "strides": [[1, 1]] + [[2, 2]] * (len(features) - 1),
        "patch_size": [32, 32],
        "targets": targets,
        "conv_op": "nn.Conv2d",
        "norm_op": "nn.InstanceNorm2d",
    }
    return SimpleNamespace(model_config=model_config, inference_config=model_config, targets=targets,
                           train_patch_size=(32, 32), train_batch_size=1, in_channels=1, autoconfigure=False,
                           op_dims=2, enable_deep_supervision=False, spacing=[1, 1], model_name="m")


def _differing_tensors(model, state):
    loaded = model.state_dict()
    return [k for k, v in state.items()
            if k not in loaded or loaded[k].shape != v.shape or not torch.equal(loaded[k], v)]


def test_checkpoint_config_wins_over_the_gui_config(tmp_path):
    torch.manual_seed(0)
    trained = NetworkFromConfig(_manager([8, 16, 32], {"ink": {"out_channels": 2, "activation": "none"}}))
    path = tmp_path / "model.pth"
    # what the trainer saves
    torch.save({"model": trained.state_dict(), "model_config": trained.final_config}, path)
    gui = _manager([16, 32, 64], {"ink": {"out_channels": 3, "activation": "none"}})

    model, _ = ModelLoader(path, config_manager=gui).load()

    assert _differing_tensors(model, trained.state_dict()) == []
    outputs = model(torch.zeros(1, 1, 32, 32))
    assert outputs["ink"].shape[1] == 2


def test_gui_config_is_used_for_a_checkpoint_without_one(tmp_path):
    gui = _manager([8, 16, 32], {"ink": {"out_channels": 2, "activation": "none"}})
    torch.manual_seed(0)
    trained = NetworkFromConfig(gui)
    path = tmp_path / "weights.pth"
    torch.save({"model": trained.state_dict()}, path)

    model, _ = ModelLoader(path, config_manager=gui).load()

    assert _differing_tensors(model, trained.state_dict()) == []
