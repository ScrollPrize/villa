import numpy as np
import torch
from types import SimpleNamespace

from vesuvius.models.datasets.base_dataset import BaseDataset


def _make_mgr():
    return SimpleNamespace(
        model_name="test-model",
        targets={"ink": {"losses": [{"name": "mse", "weight": 1.0}]}},
        train_patch_size=[4, 4],
        min_labeled_ratio=0.0,
        min_bbox_percent=0.0,
        skip_patch_validation=False,
        allow_unlabeled_data=False,
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
        data_path=".",
    )


class _MinimalDataset(BaseDataset):
    def __init__(self, mgr, image_patch, label_patch):
        self._image_patch = image_patch
        self._label_patch = label_patch
        super().__init__(mgr, is_training=False)

    def _initialize_volumes(self):
        entry = {
            "volume_id": "vol0",
            "image": self._image_patch,
            "label": self._label_patch,
            "label_path": None,
            "label_source": None,
            "has_label": True,
        }
        self.target_volumes = {"ink": [entry]}
        self.valid_patches = [
            {"volume_index": 0, "start": (0, 0), "shape": self._image_patch.shape[-2:]}
        ]
        self.patch_weights = None
        self.zarr_arrays = []
        self.zarr_names = []
        self.data_paths = []

    def _setup_chunk_slicer(self):
        self.chunk_slicer = None

    def _setup_plane_slicer(self):
        self.plane_slicer = None

    def _get_valid_patches(self):
        return self.valid_patches

    def __getitem__(self, idx):
        return {
            "image": torch.from_numpy(self._image_patch.astype(np.float32)).unsqueeze(0),
            "ink": torch.from_numpy(self._label_patch.astype(np.float32)).unsqueeze(0),
            "is_unlabeled": False,
            "patch_info": {"volume_index": 0},
        }


def test_base_dataset_minimal_flow():
    mgr = _make_mgr()
    image = np.ones((4, 4), dtype=np.float32)
    label = np.ones((4, 4), dtype=np.float32)

    dataset = _MinimalDataset(mgr, image, label)

    assert dataset.is_2d_dataset is True
    assert len(dataset) == 1

    sample = dataset[0]
    assert sample["image"].shape == (1, 4, 4)
    assert sample["ink"].shape == (1, 4, 4)

    assert dataset.valid_patches[0]["volume_index"] == 0
