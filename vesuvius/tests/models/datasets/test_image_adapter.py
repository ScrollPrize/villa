import cv2
import numpy as np
import pytest
import tifffile

from vesuvius.models.datasets.adapters import AdapterConfig, ImageAdapter


def _write_tiff(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), array, photometric="minisblack")


def _write_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), array)
    if not ok:
        raise RuntimeError(f"Failed to write PNG file: {path}")


def test_image_adapter_discovers_and_streams_labeled_volume(tmp_path):
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    labels = (data > data.mean()).astype(np.uint8)

    image_path = tmp_path / "images" / "sample.tif"
    label_path = tmp_path / "labels" / "sample_ink.tif"

    _write_tiff(image_path, data)
    _write_tiff(label_path, labels)

    config = AdapterConfig(data_path=tmp_path, targets=("ink",), allow_unlabeled=False)
    adapter = ImageAdapter(config)

    discovered = adapter.discover()
    assert len(discovered) == 1
    assert discovered[0].volume_id == "sample"

    adapter.prepare(discovered)

    volumes = list(adapter.iter_volumes())
    assert len(volumes) == 1

    volume = volumes[0]
    metadata = volume.metadata
    assert metadata.volume_id == "sample"
    assert metadata.spatial_shape == data.shape
    assert "ink" in metadata.targets_with_labels

    image_full = volume.image.read()
    assert np.array_equal(image_full, data)

    window = volume.image.read_window((2, 3), (3, 4))
    expected = data[2:5, 3:7]
    assert np.array_equal(window, expected)

    label_handle = volume.labels["ink"]
    assert label_handle is not None
    label_full = label_handle.read()
    assert np.array_equal(label_full, labels)


def test_image_adapter_allows_unlabeled_volumes(tmp_path):
    data = np.ones((3, 6, 6), dtype=np.uint8)
    image_path = tmp_path / "images" / "sample.tif"
    _write_tiff(image_path, data)

    config = AdapterConfig(data_path=tmp_path, targets=("ink",), allow_unlabeled=True)
    adapter = ImageAdapter(config)

    discovered = adapter.discover()
    adapter.prepare(discovered)
    volumes = list(adapter.iter_volumes())

    assert len(volumes) == 1
    assert volumes[0].labels["ink"] is None


def test_image_adapter_respects_chunk_shape(tmp_path):
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    labels = (data > data.mean()).astype(np.uint8)

    image_path = tmp_path / "images" / "sample.tif"
    label_path = tmp_path / "labels" / "sample_ink.tif"

    _write_tiff(image_path, data)
    _write_tiff(label_path, labels)

    config = AdapterConfig(
        data_path=tmp_path,
        targets=("ink",),
        allow_unlabeled=False,
        tiff_chunk_shape=(4, 4),
    )
    adapter = ImageAdapter(config)

    discovered = adapter.discover()
    adapter.prepare(discovered)
    volume = next(adapter.iter_volumes())

    assert getattr(volume.image, "_chunk_shape", None) == (4, 4)


def test_image_adapter_requires_labels_when_not_allowed(tmp_path):
    data = np.zeros((2, 4, 4), dtype=np.uint8)
    image_path = tmp_path / "images" / "sample.tif"
    _write_tiff(image_path, data)

    config = AdapterConfig(data_path=tmp_path, targets=("ink",), allow_unlabeled=False)
    adapter = ImageAdapter(config)

    with pytest.raises(FileNotFoundError):
        adapter.discover()


def test_image_adapter_rejects_label_shape_mismatch(tmp_path):
    image = np.zeros((4, 4), dtype=np.uint8)
    label = np.zeros((5, 5), dtype=np.uint8)

    image_path = tmp_path / "images" / "sample.tif"
    label_path = tmp_path / "labels" / "sample_ink.tif"

    _write_tiff(image_path, image)
    _write_tiff(label_path, label)

    config = AdapterConfig(data_path=tmp_path, targets=("ink",), allow_unlabeled=False)
    adapter = ImageAdapter(config)

    discovered = adapter.discover()

    with pytest.raises(ValueError):
        adapter.prepare(discovered)


def test_image_adapter_handles_png_inputs(tmp_path):
    data = (np.arange(64, dtype=np.uint8).reshape(8, 8))
    label = (data > data.mean()).astype(np.uint8) * 255

    image_path = tmp_path / "images" / "sample.png"
    label_path = tmp_path / "labels" / "sample_ink.png"

    _write_png(image_path, data)
    _write_png(label_path, label)

    config = AdapterConfig(data_path=tmp_path, targets=("ink",), allow_unlabeled=False)
    adapter = ImageAdapter(config)

    discovered = adapter.discover()
    adapter.prepare(discovered)
    volume = next(adapter.iter_volumes())

    assert volume.metadata.spatial_shape == data.shape
    assert np.array_equal(volume.image.read(), data)
    label_handle = volume.labels["ink"]
    assert label_handle is not None
    assert np.array_equal(label_handle.read(), label)
