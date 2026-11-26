"""vesuvius.napari_trainer starts with its default config and trains on the viewer's layers.

napari and magicgui are replaced by small stand-ins so no Qt window is needed; the
config manager, the layer export and the trainer's ZarrDataset are the real ones.
"""
import importlib
import inspect
import sys
import tomllib
import tempfile
import types
from copy import deepcopy
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("vc_reader")


class _Layer:
    def __init__(self, data, name):
        self.data, self.name = data, name


class _Image(_Layer):
    pass


class _Labels(_Layer):
    pass


@pytest.fixture
def main_window(monkeypatch, tmp_path):
    napari = types.ModuleType("napari")
    layers = types.ModuleType("napari.layers")
    layers.Layer, layers.Image, layers.Labels = _Layer, _Image, _Labels
    viewer_module = types.ModuleType("napari.viewer")

    class Viewer:
        def __init__(self):
            self.layers = []
            self.window = types.SimpleNamespace(add_dock_widget=lambda *a, **k: None)
            napari._current = self

    viewer_module.Viewer = Viewer
    napari.layers, napari.viewer, napari.Viewer = layers, viewer_module, Viewer
    napari._current = None
    napari.current_viewer = lambda: napari._current
    napari.run = lambda: None

    magicgui = types.ModuleType("magicgui")

    def fake_magicgui(*args, **options):
        def wrap(function):
            function.options = options
            return function
        return wrap

    magicgui.magicgui = fake_magicgui
    magicgui.widgets = types.ModuleType("magicgui.widgets")

    for name, module in {"napari": napari, "napari.layers": layers, "napari.viewer": viewer_module,
                         "magicgui": magicgui, "magicgui.widgets": magicgui.widgets}.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in [n for n in sys.modules if n.startswith("vesuvius.napari_trainer")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.chdir(tmp_path)  # the default config creates ./checkpoints
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return importlib.import_module("vesuvius.napari_trainer.main_window")


def test_main_loads_the_default_config(main_window):
    main_window.main()

    config_path = Path(main_window._config_manager._config_path)
    assert config_path.is_file()
    assert config_path.name == "ps128_dicece.yaml"
    assert "ink" in main_window._config_manager.targets


def test_run_training_trains_on_the_viewer_layers(main_window, monkeypatch):
    from vesuvius.models.datasets import ZarrDataset
    import vesuvius.models.training.train as train

    seen = {}

    class RecordingTrainer:
        def __init__(self, mgr, verbose=True):
            self.mgr = mgr

        def train(self):
            seen["dataset"] = ZarrDataset(mgr=self.mgr, is_training=True)
            seen["targets"] = {name: dict(info) for name, info in self.mgr.targets.items()}
            # the exported layers are removed when the run ends
            seen["arrays"] = [(np.asarray(v.image_array[:]), np.asarray(v.label_arrays["ink"][:]))
                              for v in seen["dataset"]._volumes]

    monkeypatch.setattr(train, "BaseTrainer", RecordingTrainer)

    import vesuvius
    from vesuvius.models.configuration.config_manager import ConfigManager

    mgr = ConfigManager(verbose=False)
    mgr.load_config(Path(vesuvius.__file__).parent / "models/configuration/single_task/ps128_dicece.yaml")
    main_window._config_manager = mgr
    viewer = sys.modules["napari"].Viewer()
    image = np.random.default_rng(0).integers(0, 255, size=(64, 64)).astype(np.uint8)
    label = np.zeros((64, 64), dtype=np.uint8)
    label[8:56, 8:56] = 1
    viewer.layers = [_Image(image, "vol01"), _Labels(label, "vol01_ink")]

    main_window.run_training(patch_size_z=0, patch_size_x=32, patch_size_y=32,
                             min_labeled_percentage=10.0, max_epochs=1)

    volumes = seen["arrays"]
    assert len(volumes) == 1
    np.testing.assert_array_equal(volumes[0][0], image)
    np.testing.assert_array_equal(volumes[0][1], label)
    assert list(seen["targets"]) == ["ink"]
    assert seen["targets"]["ink"]["losses"][0]["name"] == "nnUNet_DC_and_CE_loss"


def test_loss_choices_are_losses_the_trainer_knows(main_window):
    from vesuvius.models.training.loss.losses import _create_loss

    widget = main_window.run_training.options["loss_function"]
    names = set(widget["choices"]) | {widget["value"],
                                      inspect.signature(main_window.run_training).parameters["loss_function"].default}
    for name in sorted(names):
        _create_loss(name, {}, None, -100, None)


def _default_manager():
    import vesuvius
    from vesuvius.models.configuration.config_manager import ConfigManager

    mgr = ConfigManager(verbose=False)
    mgr.load_config(Path(vesuvius.__file__).parent / "models/configuration/single_task/ps128_dicece.yaml")
    return mgr


@pytest.fixture
def recorded(main_window, monkeypatch):
    """BaseTrainer replaced by a recorder of the config and dataset each run trains on."""
    from vesuvius.models.datasets import ZarrDataset
    import vesuvius.models.training.train as train

    runs = []

    class RecordingTrainer:
        def __init__(self, mgr, verbose=True):
            self.mgr = mgr

        def train(self):
            dataset = ZarrDataset(mgr=self.mgr, is_training=True)
            arrays = [(np.asarray(v.image_array[:]),
                       {t: np.asarray(a[:]) for t, a in v.label_arrays.items() if a is not None})
                      for v in dataset._volumes]  # read now: the export is removed when the run ends
            runs.append(SimpleNamespace(mgr=self.mgr, data_path=Path(self.mgr.data_path), dataset=dataset,
                                        arrays=arrays, targets=deepcopy(self.mgr.targets)))

    monkeypatch.setattr(train, "BaseTrainer", RecordingTrainer)
    main_window._config_manager = _default_manager()
    return runs


def _viewer(*layers):
    viewer = sys.modules["napari"].Viewer()
    viewer.layers = list(layers)
    return viewer


def _pair(name="vol01", target="ink", shape=(64, 64)):
    image = np.random.default_rng(0).integers(0, 255, size=shape).astype(np.uint8)
    label = np.zeros(shape, dtype=np.uint8)
    label[..., 8:56, 8:56] = 1
    return _Image(image, name), _Labels(label, f"{name}_{target}")


def _train(main_window, z=0, percent=10.0):
    main_window.run_training(patch_size_z=z, patch_size_x=32, patch_size_y=32,
                             min_labeled_percentage=percent, max_epochs=1)


def test_training_uses_a_copy_of_the_selected_config(main_window, recorded):
    selected = main_window._config_manager
    before = (deepcopy(selected.convert_to_dict()), selected.data_path, selected.train_patch_size)

    _viewer(*_pair(target="surface"))
    _train(main_window)
    _viewer(*_pair(target="ink"))
    _train(main_window)

    assert main_window._config_manager is selected
    assert (selected.convert_to_dict(), selected.data_path, selected.train_patch_size) == before
    assert all(run.mgr is not selected for run in recorded)
    assert list(recorded[0].targets) == ["surface"]
    # ink -> surface -> ink keeps the YAML's ink settings
    assert list(recorded[1].targets) == ["ink"]
    assert recorded[1].targets["ink"]["ignore_label"] == 2


def test_label_layer_pairs_with_the_longest_matching_image_name(main_window, recorded):
    rng = np.random.default_rng(1)
    vol = _Image(rng.integers(0, 255, size=(64, 64)).astype(np.uint8), "vol")
    crop, label = _pair("vol_crop")

    _viewer(vol, crop, label)
    _train(main_window)

    volumes = recorded[0].arrays
    assert len(volumes) == 1
    np.testing.assert_array_equal(volumes[0][0], crop.data)
    np.testing.assert_array_equal(volumes[0][1]["ink"], label.data)
    assert list(recorded[0].targets) == ["ink"]


@pytest.mark.parametrize("change, message", [
    (lambda image, label: setattr(label, "translate", np.array([5.0, 0.0])), "translate"),
    (lambda image, label: setattr(label, "scale", np.array([2.0, 2.0])), "scale"),
    (lambda image, label: setattr(label, "data", label.data[:, :32]), "shape"),
])
def test_label_layer_on_another_grid_is_rejected(main_window, recorded, change, message):
    image, label = _pair()
    image.translate, image.scale = np.zeros(2), np.ones(2)
    label.translate, label.scale = np.zeros(2), np.ones(2)
    change(image, label)

    _viewer(image, label)
    with pytest.raises(ValueError, match=message):
        _train(main_window)
    assert recorded == []


@pytest.mark.parametrize("dataset_config, message", [
    ({"volumes": [{"image": "elsewhere.zarr", "label": "elsewhere_ink.zarr"}]}, "dataset_config.volumes"),
    ({"dataset_type": "cross_frame", "image_zarr_url": "file:///nowhere/image.zarr",
      "labels_zarr_url": "file:///nowhere/labels.zarr", "transform_json_url": "file:///nowhere/t.json"},
     "cross_frame"),
])
def test_configs_that_read_other_data_are_rejected(main_window, recorded, dataset_config, message):
    main_window._config_manager.dataset_config.update(dataset_config)

    _viewer(*_pair())
    with pytest.raises(ValueError, match=message):
        _train(main_window)
    assert recorded == []


def test_exports_stay_in_a_run_directory_removed_after_the_run(main_window, recorded, monkeypatch, tmp_path):
    import vesuvius.models.training.train as train

    _viewer(*_pair("../../escaped"))
    _train(main_window)

    data_path = recorded[0].data_path
    assert data_path.is_relative_to(tmp_path)
    assert not data_path.exists()
    assert [p for p in tmp_path.rglob("*") if "escaped" in p.name] == []
    assert [p for p in tmp_path.iterdir() if p.name.startswith("napari_trainer_")] == []

    class FailingTrainer:
        def __init__(self, mgr, verbose=True):
            self.mgr = mgr

        def train(self):
            recorded.append(Path(self.mgr.data_path))
            raise RuntimeError("training failed")

    monkeypatch.setattr(train, "BaseTrainer", FailingTrainer)
    with pytest.raises(RuntimeError, match="training failed"):
        _train(main_window)
    assert recorded[-1].is_relative_to(tmp_path) and not recorded[-1].exists()

    image, label = _pair()
    label.name = "vol01_../../escaped"
    _viewer(image, label)
    with pytest.raises(ValueError, match="target"):
        _train(main_window)


def test_min_labeled_percentage_selects_the_patches(main_window, recorded):
    image, label = _pair()
    label.data = np.zeros_like(label.data)
    label.data[4:8, 4:8] = 1  # 16 of the 1024 pixels of one 32x32 patch

    _viewer(image, label)
    _train(main_window, percent=0.0)
    _train(main_window, percent=1.0)

    assert [len(run.dataset._patches) for run in recorded] == [4, 1]
    assert [p.position for p in recorded[1].dataset._patches] == [(0, 0)]
    with pytest.raises(ValueError, match="Min Labeled Percentage"):
        _train(main_window, percent=10.0)


def test_3d_layers_with_patch_size_z_0_are_rejected(main_window, recorded):
    _viewer(*_pair(shape=(8, 64, 64)))

    with pytest.raises(ValueError, match="Patch Size Z"):
        _train(main_window, z=0)
    assert recorded == []


def test_offered_losses_fit_the_default_target_and_keep_its_ignore_label(main_window):
    import torch
    from vesuvius.models.training.train import BaseTrainer

    widget = main_window.run_training.options["loss_function"]
    torch.manual_seed(0)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    target[:, :, :4] = 2  # ignore label of the default config
    fg = (target == 1).float()
    correct = torch.cat([10 - 20 * fg, 20 * fg - 10], dim=1)  # two-channel logits that match the target
    for name in widget["choices"]:
        task = {"ignore_label": 2, "losses": [{"name": name, "weight": 1.0, "kwargs": {}}]}
        trainer = SimpleNamespace(mgr=SimpleNamespace(targets={"ink": task}, enable_deep_supervision=False))
        loss_fn = BaseTrainer._build_loss(trainer)["ink"][0][0]

        loss = loss_fn(correct, target)
        assert torch.isfinite(loss), name
        assert loss < loss_fn(correct.flip(1), target), name
        changed = correct.clone()
        changed[:, :, :4] = torch.randn_like(changed[:, :, :4]) * 10  # only where the label is ignored
        torch.testing.assert_close(loss_fn(changed, target), loss, msg=name)


def test_default_config_is_package_data(main_window):
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    patterns = tomllib.loads(pyproject.read_text())["tool"]["setuptools"]["package-data"]["vesuvius"]
    package_dir = Path(main_window.__file__).resolve().parents[1]
    relative = main_window.DEFAULT_CONFIG_PATH.relative_to(package_dir).as_posix()

    assert any(fnmatch(relative, pattern) for pattern in patterns)
