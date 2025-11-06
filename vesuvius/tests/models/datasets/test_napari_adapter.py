import numpy as np
import pytest

from vesuvius.models.datasets.adapters import AdapterConfig, NapariAdapter


class _StubLayer:
    def __init__(self, name, data, layer_type):
        self.name = name
        self.data = data
        self.layer_type = layer_type


class _StubViewer:
    def __init__(self, layers):
        self.layers = layers


def test_napari_adapter_discovers_and_streams(tmp_path):
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    labels = (image > image.mean()).astype(np.uint8)

    layers = [
        _StubLayer("sample", image, "image"),
        _StubLayer("sample_ink", labels, "labels"),
    ]

    viewer = _StubViewer(layers)
    config = AdapterConfig(data_path=tmp_path, targets=("ink",), allow_unlabeled=False)
    adapter = NapariAdapter(config, viewer=viewer)

    discovered = adapter.discover()
    assert len(discovered) == 1
    assert discovered[0].volume_id == "sample"

    adapter.prepare(discovered)
    volumes = list(adapter.iter_volumes())
    assert len(volumes) == 1

    volume = volumes[0]
    assert np.array_equal(volume.image.read(), image)
    assert np.array_equal(volume.labels["ink"].read(), labels)

    window = volume.image.read_window((1, 1), (2, 3))
    expected = image[1:3, 1:4]
    assert np.array_equal(window, expected)


def test_napari_adapter_respects_allow_unlabeled(tmp_path):
    image = np.ones((3, 3), dtype=np.float32)
    viewer = _StubViewer([_StubLayer("sample", image, "image")])

    config = AdapterConfig(data_path=tmp_path, targets=("ink",), allow_unlabeled=True)
    adapter = NapariAdapter(config, viewer=viewer)

    discovered = adapter.discover()
    adapter.prepare(discovered)
    volumes = list(adapter.iter_volumes())
    assert volumes[0].labels["ink"] is None


def test_napari_adapter_requires_labels_when_not_allowed(tmp_path):
    image = np.zeros((2, 2), dtype=np.float32)
    viewer = _StubViewer([_StubLayer("sample", image, "image")])

    config = AdapterConfig(data_path=tmp_path, targets=("ink",), allow_unlabeled=False)
    adapter = NapariAdapter(config, viewer=viewer)

    with pytest.raises(ValueError):
        adapter.discover()

