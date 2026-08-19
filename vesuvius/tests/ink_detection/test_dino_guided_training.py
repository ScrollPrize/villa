"""Integration seams for the staged DINO-guided ink recipes."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import pytest
import torch

from vesuvius.ink_detection.config import (
    DinoGuidedLabelConfig,
    NormalizationConfig,
    SelfDistillLabelConfig,
    TrainingConfig,
    resolve_training_mapping,
)
from vesuvius.ink_detection.data.coord_patch_dataset import CoordPatchDataset
from vesuvius.ink_detection.training.train import (
    apply_dynamic_label_substitution,
    prepare_loss_inputs,
    prepare_validation_loss_inputs,
    should_save_checkpoint,
)

from .test_model_foundation import _config_mapping


def _training_mapping() -> dict:
    mapping = _config_mapping(depth=256, side=256)
    mapping.update(
        {
            "mode": "full_3d",
            "num_iterations": 80_001,
            "out_dir": "/tmp/ink-output",
            "seed": 17,
        }
    )
    return mapping


def _parse(mapping: dict) -> TrainingConfig:
    return TrainingConfig.from_mapping(resolve_training_mapping(mapping))


def test_dino_guided_v1_keeps_the_raw_intensity_gate_disabled():
    mapping = _training_mapping()
    mapping["dynamic_label"] = {
        "enabled": True,
        "kind": "dino_guided",
        "unet_ckpt": "teacher.pth",
        "dino_ckpt": "dino.pt",
        "ref_embedding": "ink.npy",
        "dino_stride": 128,
        "dino_minibatch": 8,
        "dino_blend_sigma": 4.0,
        "threshold": 0.5,
    }

    config = _parse(mapping)

    assert isinstance(config.dynamic_label, DinoGuidedLabelConfig)
    assert config.dynamic_label.input_mask_threshold is None
    assert config.dynamic_label.dino_stride == 128


def test_v3_extra_patches_require_self_distillation_and_uniform_sampling():
    mapping = _training_mapping()
    mapping["dynamic_label"] = {
        "enabled": True,
        "kind": "self_distill",
        "primary_ckpt": "v2-77k.pth",
        "ensemble_ckpt": "v2-64k.pth",
        "primary_threshold": 0.17647,
        "ensemble_threshold": 0.15686,
        "mean_hi": 105,
        "std_lo": 30,
        "input_mask_threshold": 50,
    }
    mapping["extra_patches"] = {
        "enabled": True,
        "coords_xyz": [[1, 2, 3]],
        "jitter": 1024,
        "fraction": 0.25,
    }

    config = _parse(mapping)

    assert isinstance(config.dynamic_label, SelfDistillLabelConfig)
    assert config.extra_patches.coords_xyz == ((1, 2, 3),)

    mapping["sampling_strategy"] = "scroll_segment_balanced"
    with pytest.raises(ValueError, match="sampling_strategy='uniform'"):
        _parse(mapping)


def test_dynamic_labels_reject_non_native_or_wrong_size_inputs():
    mapping = _training_mapping()
    mapping["dynamic_label"] = {
        "enabled": True,
        "kind": "dino_guided",
        "unet_ckpt": "teacher.pth",
        "dino_ckpt": "dino.pt",
        "ref_embedding": "ink.npy",
    }
    mapping["mode"] = "flat"
    with pytest.raises(ValueError, match="only in mode='full_3d'"):
        _parse(mapping)

    mapping["mode"] = "full_3d"
    mapping["patch_size"] = [128, 256, 256]
    with pytest.raises(ValueError, match="requires patch_size"):
        _parse(mapping)


def test_unlabeled_dynamic_labels_require_full_supervision():
    mapping = _training_mapping()
    mapping["patch_discovery_mode"] = "unlabeled"
    mapping["unlabeled_datasets"] = mapping["datasets"]
    mapping["dynamic_label"] = {
        "enabled": True,
        "kind": "self_distill",
        "primary_ckpt": "v2-77k.pth",
        "ensemble_ckpt": "v2-64k.pth",
        "primary_threshold": 0.17647,
        "ensemble_threshold": 0.15686,
        "mean_hi": 105,
        "std_lo": 30,
    }

    with pytest.raises(
        ValueError,
        match="patch_discovery_mode='unlabeled'.*force_full_supervision=true",
    ):
        _parse(mapping)

    mapping["force_full_supervision"] = True
    assert _parse(mapping).force_full_supervision is True


def test_force_full_supervision_zeroes_native_ignore_mask():
    predictions = torch.zeros(1, 1, 2, 2, 2)
    batch = {
        "image": torch.zeros_like(predictions),
        "inklabels": torch.zeros_like(predictions),
        "supervision_mask": torch.zeros_like(predictions),
    }

    _, _, ordinary_ignore = prepare_loss_inputs(
        predictions, batch, mode="full_3d"
    )
    _, _, full_ignore = prepare_loss_inputs(
        predictions,
        batch,
        mode="full_3d",
        force_full_supervision=True,
    )

    assert torch.all(ordinary_ignore == 1)
    assert torch.all(full_ignore == 0)


def test_validation_always_preserves_the_stored_supervision_mask():
    predictions = torch.zeros(1, 1, 2, 2, 2)
    batch = {
        "image": torch.zeros_like(predictions),
        "inklabels": torch.zeros_like(predictions),
        "supervision_mask": torch.zeros_like(predictions),
    }

    _, _, validation_ignore = prepare_validation_loss_inputs(
        predictions,
        batch,
        mode="full_3d",
    )

    assert torch.all(validation_ignore == 1)


def test_dynamic_label_substitution_uses_clean_image_and_removes_helper_keys():
    class Generator:
        def generate(self, image, mask_b1zyx=None, **kwargs):
            assert torch.equal(image, torch.full_like(image, 2))
            assert torch.equal(mask_b1zyx, torch.ones_like(image))
            assert set(kwargs) == {"raw_mean", "raw_std"}
            return torch.ones_like(image)

    image = torch.zeros(1, 1, 2, 2, 2)
    batch = {
        "image": image,
        "image_for_label": torch.full_like(image, 2),
        "image_mask_for_label": torch.ones_like(image),
        "image_raw_mean": torch.tensor([110.0]),
        "image_raw_std": torch.tensor([20.0]),
        "inklabels": torch.zeros_like(image),
    }

    apply_dynamic_label_substitution(batch, Generator(), kind="self_distill")

    assert torch.all(batch["inklabels"] == 1)
    assert not any(key.startswith("image_") for key in batch)


def test_coordinate_dataset_interprets_authored_coordinates_as_xyz():
    volume = np.arange(6 * 6 * 6, dtype=np.uint8).reshape(6, 6, 6)
    dataset = CoordPatchDataset(
        volume_path="unused.zarr",
        resolution=0,
        coords_xyz=[(3, 2, 1)],
        jitter=0,
        length=2,
        patch_size=(2, 2, 2),
        normalization=NormalizationConfig(mode="divide", divisor=255.0),
        input_mask_threshold=20,
    )
    dataset._volume = volume

    sample = dataset[0]
    raw_expected = torch.from_numpy(volume[0:2, 1:3, 2:4].astype(np.float32))
    expected = raw_expected / 255.0

    assert torch.equal(sample["image"][0], expected)
    assert torch.equal(sample["image_for_label"], sample["image"])
    assert sample["image_raw_mean"].item() == pytest.approx(
        float(raw_expected.mean())
    )
    assert torch.equal(
        sample["image_mask_for_label"][0], (raw_expected > 20).float()
    )


def test_explicit_save_iterations_use_current_completed_iteration_names():
    assert should_save_checkpoint(60_000, save_every=10_000, save_iterations=(60_001,))
    assert not should_save_checkpoint(
        59_999, save_every=100_000, save_iterations=(60_001,)
    )
    assert should_save_checkpoint(9, save_every=10, save_iterations=())


def test_all_shipped_phase_recipes_parse_and_preserve_the_lineage():
    config_dir = (
        Path(__file__).parents[2]
        / "src"
        / "vesuvius"
        / "ink_detection"
        / "configs"
    )
    parsed = {}
    for name in ("teacher", "v1", "v2", "v3", "v3_fullsup"):
        with (config_dir / f"dino_guided_{name}.json").open(
            encoding="utf-8"
        ) as stream:
            parsed[name] = _parse(json.load(stream))

    assert parsed["teacher"].dynamic_label is None
    assert parsed["teacher"].save_iterations == (60_001,)
    assert parsed["v1"].save_iterations == (63_001,)
    assert parsed["v2"].save_iterations == (64_001, 77_001)
    assert parsed["v3"].save_iterations == (79_001,)
    assert parsed["v3_fullsup"].save_iterations == (78_001,)
    assert parsed["v3"].force_full_supervision is False
    assert parsed["v3_fullsup"].force_full_supervision is True
