import numpy as np
import pytest
import zarr

from vesuvius.models.datasets.adapters import AdapterConfig, ZarrAdapter


def _write_zarr(path, array):
    store = zarr.open(str(path), mode="w", shape=array.shape, dtype=array.dtype)
    store[...] = array


def test_zarr_adapter_discovers_and_streams_labeled_volume(tmp_path):
    image_dir = tmp_path / "images" / "sample.zarr"
    label_dir = tmp_path / "labels" / "sample_ink.zarr"

    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    labels = (data > data.mean()).astype(np.uint8)

    _write_zarr(image_dir, data)
    _write_zarr(label_dir, labels)

    config = AdapterConfig(data_path=tmp_path, targets=("ink",), allow_unlabeled=False)
    adapter = ZarrAdapter(config)

    discovered = adapter.discover()
    assert len(discovered) == 1
    assert discovered[0].volume_id == "sample"

    adapter.prepare(discovered)

    volumes = list(adapter.iter_volumes())
    assert len(volumes) == 1

    volume = volumes[0]
    assert volume.metadata.volume_id == "sample"
    assert np.array_equal(volume.image.read(), data)

    window = volume.image.read_window((2, 3), (3, 4))
    expected = data[2:5, 3:7]
    assert np.array_equal(window, expected)

    label_handle = volume.labels["ink"]
    assert label_handle is not None
    assert np.array_equal(label_handle.read(), labels)


def test_zarr_adapter_allows_unlabeled(tmp_path):
    image_dir = tmp_path / "images" / "sample.zarr"
    data = np.ones((4, 4), dtype=np.uint8)
    _write_zarr(image_dir, data)

    config = AdapterConfig(data_path=tmp_path, targets=("ink",), allow_unlabeled=True)
    adapter = ZarrAdapter(config)

    discovered = adapter.discover()
    adapter.prepare(discovered)
    volumes = list(adapter.iter_volumes())

    assert len(volumes) == 1
    assert volumes[0].labels["ink"] is None


def test_zarr_adapter_requires_labels_when_not_allowed(tmp_path):
    image_dir = tmp_path / "images" / "sample.zarr"
    data = np.zeros((4, 4), dtype=np.uint8)
    _write_zarr(image_dir, data)

    config = AdapterConfig(data_path=tmp_path, targets=("ink",), allow_unlabeled=False)
    adapter = ZarrAdapter(config)

    with pytest.raises(FileNotFoundError):
        adapter.discover()


def test_zarr_adapter_shape_mismatch(tmp_path):
    image_dir = tmp_path / "images" / "sample.zarr"
    label_dir = tmp_path / "labels" / "sample_ink.zarr"

    data = np.zeros((4, 4), dtype=np.uint8)
    labels = np.zeros((5, 5), dtype=np.uint8)

    _write_zarr(image_dir, data)
    _write_zarr(label_dir, labels)

    config = AdapterConfig(data_path=tmp_path, targets=("ink",), allow_unlabeled=False)
    adapter = ZarrAdapter(config)

    discovered = adapter.discover()
    with pytest.raises(ValueError):
        adapter.prepare(discovered)
