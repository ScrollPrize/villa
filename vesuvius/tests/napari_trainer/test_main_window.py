"""vesuvius.napari_trainer starts with its default config and trains on the viewer's layers.

napari and magicgui are replaced by small stand-ins so no Qt window is needed; the
config manager, the layer export and the trainer's ZarrDataset are the real ones.
"""
import importlib
import inspect
import sys
import tempfile
import types
from pathlib import Path

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

    volumes = seen["dataset"]._volumes
    assert [v.volume_id for v in volumes] == ["vol01"]
    np.testing.assert_array_equal(np.asarray(volumes[0].image_array[:]), image)
    np.testing.assert_array_equal(np.asarray(volumes[0].label_arrays["ink"][:]), label)
    assert list(seen["targets"]) == ["ink"]
    assert seen["targets"]["ink"]["losses"][0]["name"] == "nnUNet_DC_and_CE_loss"


def test_loss_choices_are_losses_the_trainer_knows(main_window):
    from vesuvius.models.training.loss.losses import _create_loss

    widget = main_window.run_training.options["loss_function"]
    names = set(widget["choices"]) | {widget["value"],
                                      inspect.signature(main_window.run_training).parameters["loss_function"].default}
    for name in sorted(names):
        _create_loss(name, {}, None, -100, None)
