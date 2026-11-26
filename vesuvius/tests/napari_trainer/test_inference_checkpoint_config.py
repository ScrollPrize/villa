"""The napari trainer's inference builds the network a checkpoint was trained with."""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("napari")

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.napari_trainer.inference import ModelLoader, run_inference


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


def _outputs(model, shape=(1, 1, 32, 32)):
    torch.manual_seed(1)
    x = torch.randn(*shape)
    model.eval()
    with torch.no_grad():
        return model(x)


def test_per_target_decoder_layout_is_restored(tmp_path):
    # ink uses the shared decoder, surface its own: the saved config says separate_decoders: true
    targets = {"ink": {"out_channels": 2, "activation": "none"},
               "surface": {"out_channels": 2, "activation": "none", "separate_decoder": True}}
    torch.manual_seed(0)
    trained = NetworkFromConfig(_manager([8, 16, 32], targets))
    assert trained.final_config["separate_decoders"] is True
    path = tmp_path / "mixed.pth"
    torch.save({"model": trained.state_dict(), "model_config": trained.final_config}, path)

    model, _ = ModelLoader(path).load()

    assert _differing_tensors(model, trained.state_dict()) == []
    expected, got = _outputs(trained), _outputs(model)
    for name in targets:
        torch.testing.assert_close(got[name], expected[name])


def test_nested_legacy_config_is_unwrapped(tmp_path):
    trained_mgr = _manager([8, 16, 32], {"ink": {"out_channels": 2, "activation": "none"}})
    torch.manual_seed(0)
    trained = NetworkFromConfig(trained_mgr)
    model_config = {k: v for k, v in trained_mgr.model_config.items() if k not in ("targets", "patch_size")}
    legacy = {"model_config": model_config, "patch_size": [32, 32], "batch_size": 1, "in_channels": 1,
              "targets": {"ink": {"out_channels": 2, "activation": "none"}}}
    path = tmp_path / "legacy.pth"
    torch.save({"model": trained.state_dict(), "config": legacy}, path)

    model, config = ModelLoader(path).load()

    assert _differing_tensors(model, trained.state_dict()) == []
    assert list(config["patch_size"]) == [32, 32]


class _Viewer:
    def __init__(self):
        self.added = []

    def add_image(self, data, name, **kwargs):
        self.added.append((name, data))
        return SimpleNamespace(name=name, data=data)


@pytest.fixture
def no_gui_config(monkeypatch):
    main_window = types.ModuleType("vesuvius.napari_trainer.main_window")
    main_window._config_manager = None
    monkeypatch.setitem(sys.modules, "vesuvius.napari_trainer.main_window", main_window)


def test_primus_checkpoint_infers_with_its_input_shape(tmp_path, no_gui_config):
    mgr = SimpleNamespace(targets={"ink": {"out_channels": 2, "activation": "none"}}, train_patch_size=(16, 16, 16),
                          train_batch_size=1, in_channels=1, autoconfigure=False, model_name="primus",
                          enable_deep_supervision=False,
                          model_config={"architecture_type": "primus_s", "input_shape": (16, 16, 16),
                                        "patch_embed_size": (8, 8, 8)})
    torch.manual_seed(0)
    trained = NetworkFromConfig(mgr)
    assert "patch_size" not in trained.final_config
    path = tmp_path / "primus.pth"
    torch.save({"model": trained.state_dict(), "model_config": trained.final_config}, path)
    viewer = _Viewer()
    image = SimpleNamespace(name="vol", data=np.random.default_rng(0).random((20, 24, 16)).astype(np.float32))

    run_inference(viewer, image, str(path))

    assert [(name, data.shape) for name, data in viewer.added] == [("vol_primus_ink", (1, 2, 20, 24, 16))]


def test_missing_weights_stop_before_a_prediction(tmp_path, no_gui_config):
    torch.manual_seed(0)
    trained = NetworkFromConfig(_manager([8, 16, 32], {"ink": {"out_channels": 2, "activation": "none"}}))
    state = trained.state_dict()
    dropped = next(k for k in state if k.startswith("task_heads.ink."))
    del state[dropped]
    path = tmp_path / "incomplete.pth"
    torch.save({"model": state, "model_config": trained.final_config}, path)
    viewer = _Viewer()
    image = SimpleNamespace(name="vol", data=np.zeros((32, 32), np.float32))

    with pytest.raises(RuntimeError, match="no weights"):
        run_inference(viewer, image, str(path))
    assert viewer.added == []


def test_extra_checkpoint_tensors_are_not_rejected(tmp_path):
    torch.manual_seed(0)
    trained = NetworkFromConfig(_manager([8, 16, 32], {"ink": {"out_channels": 2, "activation": "none"}}))
    state = dict(trained.state_dict())
    # e.g. the segmentation layers an older shared decoder or a deep-supervision decoder saved
    state["shared_decoder.seg_layers.0.weight"] = torch.zeros(2, 16, 1, 1)
    state["shared_decoder.seg_layers.0.bias"] = torch.zeros(2)
    path = tmp_path / "extra.pth"
    torch.save({"model": state, "model_config": trained.final_config}, path)

    model, _ = ModelLoader(path).load()

    assert _differing_tensors(model, trained.state_dict()) == []
