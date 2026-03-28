from __future__ import annotations

from pathlib import Path

import yaml
import pytest
import torch
import numpy as np
from scipy import ndimage as ndi

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.datasets.find_valid_patches import _filter_z_starts_by_ignore_bounds
from vesuvius.models.datasets.zarr_dataset import ZarrDataset
from vesuvius.models.evaluation.base_metric import prediction_to_discrete_labels
from vesuvius.models.preprocessing.patches.cache import build_cache_params, cache_filename
from vesuvius.models.preprocessing.patches.generate import _full_resolution_patch_size
from vesuvius.models.training.loss.losses import BinaryBCEAndDiceLoss
from vesuvius.utils.plotting import _resolve_visualization_activation


def _write_config(
    tmp_path: Path,
    *,
    ome_zarr_resolution: int,
    valid_patch_find_resolution: int,
) -> Path:
    config = {
        "tr_setup": {
            "model_name": "surface-test",
        },
        "tr_config": {
            "patch_size": [128, 128, 128],
            "batch_size": 2,
        },
        "dataset_config": {
            "data_path": str(tmp_path),
            "ome_zarr_resolution": ome_zarr_resolution,
            "valid_patch_find_resolution": valid_patch_find_resolution,
            "targets": {
                "surface": {
                    "activation": "none",
                    "ignore_label": 2,
                }
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_config_manager_loads_ome_zarr_resolution(tmp_path: Path) -> None:
    mgr = ConfigManager(verbose=False)
    mgr.load_config(_write_config(tmp_path, ome_zarr_resolution=2, valid_patch_find_resolution=3))
    assert mgr.ome_zarr_resolution == 2
    assert mgr.valid_patch_find_resolution == 3


def test_config_manager_rejects_patch_find_resolution_below_training_resolution(tmp_path: Path) -> None:
    mgr = ConfigManager(verbose=False)
    with pytest.raises(ValueError, match="valid_patch_find_resolution"):
        mgr.load_config(_write_config(tmp_path, ome_zarr_resolution=2, valid_patch_find_resolution=1))


def test_patch_cache_filename_varies_by_training_resolution(tmp_path: Path) -> None:
    common_kwargs = {
        "data_path": tmp_path,
        "volume_ids": ["sample"],
        "patch_size": [128, 128, 128],
        "min_labeled_ratio": 0.001,
        "bbox_threshold": 0.35,
        "valid_patch_find_resolution": 3,
    }
    scale0 = build_cache_params(ome_zarr_resolution=0, **common_kwargs)
    scale2 = build_cache_params(ome_zarr_resolution=2, **common_kwargs)
    assert cache_filename(scale0) != cache_filename(scale2)


def test_patch_cache_filename_varies_by_ignore_label(tmp_path: Path) -> None:
    common_kwargs = {
        "data_path": tmp_path,
        "volume_ids": ["sample"],
        "patch_size": [128, 128, 128],
        "min_labeled_ratio": 0.001,
        "bbox_threshold": 0.35,
        "valid_patch_find_resolution": 3,
    }
    ignore0 = build_cache_params(ignore_label=0, **common_kwargs)
    ignore2 = build_cache_params(ignore_label=2, **common_kwargs)
    assert cache_filename(ignore0) != cache_filename(ignore2)


def test_cached_positions_scale_to_training_level() -> None:
    assert ZarrDataset._cached_position_to_training_level((256, 128, 64), 2) == (64, 32, 16)
    with pytest.raises(ValueError, match="not divisible"):
        ZarrDataset._cached_position_to_training_level((258, 128, 64), 2)


def test_zero_components_without_foreground_become_ignore() -> None:
    ds = ZarrDataset.__new__(ZarrDataset)
    ds._zero_cc_to_ignore_enabled = True
    ds._zero_cc_to_ignore_structure = ndi.generate_binary_structure(2, 2)
    label = np.array(
        [
            [0, 0, 2, 2],
            [0, 1, 2, 0],
            [0, 0, 2, 0],
            [2, 2, 2, 0],
        ],
        dtype=np.float32,
    )
    cleaned = ds._zero_components_without_foreground_to_ignore(
        label,
        valid_patch_value=1,
        ignore_label=2,
    )
    expected = np.array(
        [
            [0, 0, 2, 2],
            [0, 1, 2, 2],
            [0, 0, 2, 2],
            [2, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(cleaned, expected)


def test_all_zero_patch_becomes_ignore_when_no_foreground_present() -> None:
    ds = ZarrDataset.__new__(ZarrDataset)
    ds._zero_cc_to_ignore_enabled = True
    ds._zero_cc_to_ignore_structure = ndi.generate_binary_structure(3, 3)
    label = np.zeros((2, 2, 2), dtype=np.float32)
    cleaned = ds._zero_components_without_foreground_to_ignore(
        label,
        valid_patch_value=1,
        ignore_label=2,
    )
    np.testing.assert_array_equal(cleaned, np.full_like(label, 2.0))


def test_filter_z_starts_by_ignore_bounds_keeps_intersecting_patches() -> None:
    z_starts = [0, 64, 128, 192]
    filtered = _filter_z_starts_by_ignore_bounds(
        z_starts,
        patch_depth=64,
        ignore_bounds=(96, 159),
    )
    assert filtered == [64, 128]


def test_full_resolution_patch_size_uses_training_scale() -> None:
    assert _full_resolution_patch_size((128, 128, 128), 0) == (128, 128, 128)
    assert _full_resolution_patch_size((128, 128, 128), 2) == (512, 512, 512)


def test_binary_bce_and_dice_loss_ignores_ignore_label() -> None:
    loss_fn = BinaryBCEAndDiceLoss(ignore_label=2)
    logits = torch.tensor([[[[[0.0, 1.0], [2.0, -1.0]]]]], dtype=torch.float32)
    target = torch.tensor([[[[[0.0, 1.0], [2.0, 0.0]]]]], dtype=torch.float32)
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_prediction_to_discrete_labels_thresholds_single_channel_logits() -> None:
    logits = np.array([[[[[-2.0, 2.0], [0.1, -0.1]]]]], dtype=np.float32)
    labels = prediction_to_discrete_labels(logits)
    assert labels.shape == (1, 1, 2, 2)
    assert labels.tolist() == [[[[0, 1], [1, 0]]]]


def test_visualization_activation_uses_sigmoid_for_binary_bce_dice() -> None:
    task_cfg = {
        "out_channels": 1,
        "activation": "none",
        "losses": [{"name": "BinaryBCEAndDiceLoss"}],
    }
    assert _resolve_visualization_activation(task_cfg) == "sigmoid"
