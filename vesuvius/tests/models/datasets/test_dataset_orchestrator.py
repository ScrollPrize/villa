import numpy as np
import torch
import tifffile
import zarr

from types import SimpleNamespace

from vesuvius.models.datasets import DatasetOrchestrator


def _write_tiff(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), array, photometric="minisblack")


def _write_zarr(path, array):
    store = zarr.open(str(path), mode="w", shape=array.shape, dtype=array.dtype)
    store[...] = array


def _make_mgr(data_path, *, allow_unlabeled=False, patch_size=None):
    return SimpleNamespace(
        model_name="test-model",
        targets={"ink": {"loss": "bce"}},
        train_patch_size=patch_size or [4, 4],
        min_labeled_ratio=0.0,
        min_bbox_percent=0.0,
        skip_patch_validation=False,
        allow_unlabeled_data=allow_unlabeled,
        normalization_scheme="zscore",
        intensity_properties={},
        slice_sampling_enabled=False,
        slice_sample_planes=[],
        slice_plane_weights={},
        slice_plane_patch_sizes={},
        slice_random_rotation_planes={},
        slice_random_tilt_planes={},
        slice_label_interpolation={},
        slice_save_plane_masks=False,
        slice_plane_mask_mode="plane",
        cache_valid_patches=False,
        dataset_config={},
        skip_intensity_sampling=True,
        data_path=data_path,
        num_workers=0,
        downsample_level=1,
        train_batch_size=2,
    )


def test_dataset_orchestrator_with_image_adapter(tmp_path):
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    labels = (data > data.mean()).astype(np.uint8)

    image_path = tmp_path / "images" / "sample.tif"
    label_path = tmp_path / "labels" / "sample_ink.tif"

    _write_tiff(image_path, data)
    _write_tiff(label_path, labels)

    mgr = _make_mgr(tmp_path, patch_size=[4, 4])
    mgr.data_format = "image"

    dataset = DatasetOrchestrator(mgr=mgr, adapter="image", is_training=False)

    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].ndim in (3, 4)
    assert "ink" in sample

    labeled, unlabeled = dataset.get_labeled_unlabeled_patch_indices()
    assert len(labeled) + len(unlabeled) == len(dataset.valid_patches)


def test_dataset_orchestrator_with_zarr_unlabeled(tmp_path):
    data = np.ones((8, 8), dtype=np.float32)
    image_dir = tmp_path / "images" / "sample.zarr"
    _write_zarr(image_dir, data)

    mgr = _make_mgr(tmp_path, allow_unlabeled=True, patch_size=[4, 4])
    mgr.data_format = "zarr"

    dataset = DatasetOrchestrator(mgr=mgr, adapter="zarr", is_training=False)

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["is_unlabeled"] is True

    labeled, unlabeled = dataset.get_labeled_unlabeled_patch_indices()
    assert labeled == []
    assert len(unlabeled) == len(dataset.valid_patches)


def test_dataset_orchestrator_with_zarr_labeled(tmp_path):
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    labels = (image > image.mean()).astype(np.uint8)

    image_dir = tmp_path / "images" / "sample.zarr"
    label_dir = tmp_path / "labels" / "sample_ink.zarr"

    _write_zarr(image_dir, image)
    _write_zarr(label_dir, labels)

    mgr = _make_mgr(tmp_path, allow_unlabeled=False, patch_size=[4, 4])
    mgr.data_format = "zarr"

    dataset = DatasetOrchestrator(mgr=mgr, adapter="zarr", is_training=False)

    assert len(dataset) > 0
    labeled, unlabeled = dataset.get_labeled_unlabeled_patch_indices()
    assert len(labeled) == len(dataset.valid_patches)
    assert unlabeled == []


def test_dataset_orchestrator_with_napari_adapter(tmp_path):
    image = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    labels = (image > image.mean()).astype(np.uint8)

    class _StubLayer:
        def __init__(self, name, data, layer_type):
            self.name = name
            self.data = data
            self.layer_type = layer_type

    class _StubViewer:
        def __init__(self, layers):
            self.layers = layers

    viewer = _StubViewer([
        _StubLayer("sample", image, "image"),
        _StubLayer("sample_ink", labels, "labels"),
    ])

    mgr = _make_mgr(tmp_path, patch_size=[4, 4, 4])
    mgr.data_format = "napari"
    mgr.napari_viewer = viewer

    dataset = DatasetOrchestrator(
        mgr=mgr,
        adapter="napari",
        adapter_kwargs={"viewer": viewer},
        is_training=False,
    )

    assert len(dataset) > 0
    sample = dataset[0]
    image_tensor = sample["image"]
    assert image_tensor.ndim == 4
    assert image_tensor.shape[1:] == image.shape
    assert torch.isfinite(image_tensor).all()

    labeled, unlabeled = dataset.get_labeled_unlabeled_patch_indices()
    assert len(labeled) == len(dataset.valid_patches)
    assert unlabeled == []
