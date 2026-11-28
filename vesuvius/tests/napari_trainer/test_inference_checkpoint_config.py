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


def _checkpoint(tmp_path, **saved):
    """A small 2D U-Net checkpoint (32x32 patch) as the trainer saves it, plus `saved`."""
    torch.manual_seed(0)
    trained = NetworkFromConfig(_manager([4, 8], {"ink": {"out_channels": 2, "activation": "none"}}))
    path = tmp_path / "unet2d.pth"
    torch.save({"model": trained.state_dict(), "model_config": trained.final_config, **saved}, path)
    return str(path)


@pytest.fixture
def model_inputs(monkeypatch):
    """Every batch the loaded network is called with, and whether it was contiguous."""
    seen = []
    real_load = ModelLoader.load

    def load(self):
        model, config = real_load(self)
        model.register_forward_pre_hook(
            lambda module, args: seen.append((args[0].detach().cpu().clone(), args[0].is_contiguous())))
        return model, config

    monkeypatch.setattr(ModelLoader, "load", load)
    return seen


def _first_window(model_inputs):
    return model_inputs[0][0][0, 0].numpy()


def test_zscore_checkpoint_sees_zscored_windows(tmp_path, no_gui_config, model_inputs):
    # save_checkpoint stores the training dataset's normalization_scheme and intensity_properties
    path = _checkpoint(tmp_path, normalization_scheme="zscore", intensity_properties={})
    data = np.random.default_rng(0).normal(100, 10, (20, 48)).astype(np.float32)

    run_inference(_Viewer(), SimpleNamespace(name="vol", data=data), path)

    window = _first_window(model_inputs)  # the window at (0, 0); the image is 20 rows, the patch 32
    inside = window[:20]
    assert abs(inside.mean()) < 1e-4 and abs(inside.std() - 1) < 1e-3
    np.testing.assert_allclose(inside, (data[:, :32] - data[:, :32].mean()) / data[:, :32].std(), atol=1e-4)
    assert np.all(window[20:] == 0)  # padded after normalization, as the trainer pads a patch


def test_ct_checkpoint_sees_its_saved_intensity_properties(tmp_path, no_gui_config, model_inputs):
    props = {"mean": 100.0, "std": 10.0, "percentile_00_5": 85.0, "percentile_99_5": 115.0}
    path = _checkpoint(tmp_path, normalization_scheme="ct", intensity_properties=props)
    data = np.random.default_rng(0).normal(100, 10, (48, 48)).astype(np.float32)

    run_inference(_Viewer(), SimpleNamespace(name="vol", data=data), path)

    np.testing.assert_allclose(_first_window(model_inputs), (np.clip(data[:32, :32], 85, 115) - 100) / 10, atol=1e-5)


def test_checkpoint_without_saved_preprocessing_keeps_whole_image_minmax(tmp_path, no_gui_config, model_inputs):
    import vc_reader.transforms as A

    path = _checkpoint(tmp_path)
    data = np.random.default_rng(0).normal(100, 10, (48, 48)).astype(np.float32)

    run_inference(_Viewer(), SimpleNamespace(name="vol", data=data), path)

    expected = A.Normalize("minmax", eps=1e-5)(torch.as_tensor(data)[None]).numpy()[0]
    np.testing.assert_array_equal(_first_window(model_inputs), expected[:32, :32])


@pytest.mark.parametrize("saved", [{}, {"normalization_scheme": "zscore", "intensity_properties": {}}],
                         ids=["no-saved-preprocessing", "zscore"])
def test_the_network_gets_contiguous_windows(tmp_path, no_gui_config, model_inputs, saved):
    path = _checkpoint(tmp_path, **saved)
    data = np.random.default_rng(0).random((3, 50, 70)).astype(np.float32)[1]  # a strided view

    run_inference(_Viewer(), SimpleNamespace(name="vol", data=data), path, batch_size=3)

    assert len(model_inputs) > 1 and all(contiguous for _, contiguous in model_inputs)


class _NapariViewer:
    def __init__(self):
        import napari

        self.layers, self._image = [], napari.layers.Image

    def add_image(self, data, **kwargs):
        layer = self._image(data, **kwargs)
        self.layers.append(layer)
        return layer


def test_prediction_layers_keep_the_source_layers_transform(tmp_path, no_gui_config):
    import napari

    path = _checkpoint(tmp_path, normalization_scheme="zscore", intensity_properties={})
    source = napari.layers.Image(
        np.random.default_rng(0).random((40, 48)).astype(np.float32), name="vol",
        scale=(2.0, 0.5), translate=(10.0, -4.0), rotate=30.0, shear=[0.2],
        affine=np.array([[1.0, 0.1, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]))
    viewer = _NapariViewer()

    run_inference(viewer, source, path)

    (prediction,) = viewer.layers
    assert prediction.data.shape == (1, 2, 40, 48)
    for point in [(0, 0), (5, 7), (39, 47)]:
        for channel in (0, 1):
            np.testing.assert_allclose(prediction.data_to_world((0, channel) + point),
                                       (0, channel) + tuple(source.data_to_world(point)), atol=1e-9)
    np.testing.assert_allclose(prediction.scale[-2:], source.scale)
    np.testing.assert_allclose(prediction.translate[-2:], source.translate)
