"""Excluded annotation values must not affect native targets or discovery."""

from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from vesuvius.ink_detection.config import InkConfig, TrainingConfig, resolve_training_mapping
from vesuvius.ink_detection.data.dataset import InkDataset
from vesuvius.ink_detection.data.patch_cache import load_patch_cache, patch_cache_path, save_patch_cache
from vesuvius.ink_detection.data.patch_finding_default import find_segment_patches as find_default
from vesuvius.ink_detection.data.patch_finding_subtiling import find_segment_patches as find_subtiling
from vesuvius.ink_detection.training.deep_supervision import (
    DeepSupervisionWrapper,
    concatenate_deep_supervision_ignore,
    deep_supervision_weights,
)
from vesuvius.ink_detection.training.dilation import apply_label_dilation
from vesuvius.ink_detection.training.losses import create_loss
from vesuvius.ink_detection.training.train import prepare_loss_inputs, prepare_model_input
from vesuvius.ink_detection.types import Patch, Segment

from .test_data_foundation import _FakeTifxyz
from .test_model_foundation import _config_mapping


def _native_fixture(tmp_path, mode, is_validation, *, augment=False):
    mapping = _config_mapping(depth=5, side=4)
    mapping.update(mode=mode, datasets=[{"segments_path": str(tmp_path), "volume_scale": 0}])
    config = InkConfig.from_mapping(mapping)
    arrays = {
        "image": np.arange(10**3, dtype=np.uint16).reshape(10, 10, 10),
        "labels": np.zeros((5, 4, 4), dtype=np.uint8),
        "supervision": np.full((5, 4, 4), 255, dtype=np.uint8),
        "validation": np.zeros((5, 4, 4), dtype=np.uint8),
    }
    arrays["labels"][:, :, 1:] = 255
    arrays["supervision"][:, :, 3] = 0
    arrays["validation"][:, :, 2] = 255
    segment = Segment(config.data, config.data.datasets[0], 0, "primary", tmp_path,
                      "primary", "image", "labels", "supervision", "validation")
    patch = Patch(segment=segment, bbox=(0, 0, 0, 5, 4, 4), is_validation=is_validation,
                  supervision_mask_override="validation" if is_validation else None)
    dataset = InkDataset(config.data, do_augmentations=augment, patches=[patch], segments=[segment])
    dataset._open = lambda path, resolution: arrays[str(path)]
    yy, xx = np.indices((4, 4), dtype=np.float32)
    positions = np.stack([np.full_like(yy, 4), yy + 2, xx + 2], axis=-1)
    dataset._tifxyz_cache[str(tmp_path)] = _FakeTifxyz(positions)
    return config, dataset, arrays


def _load_and_loss(config, dataset):
    random.seed(431)
    np.random.seed(431)
    torch.manual_seed(431)
    sample = dataset[0]
    batch = {name: value.unsqueeze(0) for name, value in sample.items() if isinstance(value, torch.Tensor)}
    batch = apply_label_dilation(batch, 0.0, 0.0)
    logits = torch.linspace(-1.3, 1.7, sample["inklabels"].numel()).reshape_as(batch["inklabels"])
    logits.requires_grad_()
    prediction, targets, ignore = prepare_loss_inputs(logits, batch, mode=config.data.mode)
    packed = concatenate_deep_supervision_ignore(targets, ignore, prediction)
    loss = create_loss(config)(prediction, packed)
    loss.backward()
    return sample, loss.detach(), logits.grad


@pytest.mark.parametrize("mode", ["full_3d", "full_3d_single_wrap"])
@pytest.mark.parametrize("is_validation", [False, True])
@pytest.mark.parametrize("augment", [False, True])
def test_excluded_values_do_not_change_native_sample_loss_or_gradient(tmp_path, mode, is_validation, augment):
    config, dataset, arrays = _native_fixture(tmp_path, mode, is_validation, augment=augment)
    excluded = arrays["validation"] == 0 if is_validation else arrays["validation"] > 0
    original_labels = arrays["labels"].copy()
    original, loss, gradient = _load_and_loss(config, dataset)
    for value in (0, 255):
        arrays["labels"] = original_labels.copy()
        arrays["labels"][excluded] = value
        mutated, other_loss, other_gradient = _load_and_loss(config, dataset)
        for name in original:
            if isinstance(original[name], torch.Tensor):
                torch.testing.assert_close(mutated[name], original[name], rtol=0, atol=0)
            else:
                assert mutated[name] == original[name]
        torch.testing.assert_close(other_loss, loss, rtol=0, atol=0)
        torch.testing.assert_close(other_gradient, gradient, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["full_3d", "full_3d_single_wrap"])
def test_permitted_positive_outside_supervision_is_preserved(tmp_path, mode):
    config, dataset, arrays = _native_fixture(tmp_path, mode, False)
    original, _, _ = _load_and_loss(config, dataset)
    # Column 3 is outside ordinary supervision but is not validation. Native
    # projection deliberately treats these positive labels as supervised.
    arrays["labels"][:, :, 3] = 0
    changed, _, _ = _load_and_loss(config, dataset)
    assert original["inklabels"].sum() > changed["inklabels"].sum()
    assert original["supervision_mask"].sum() > changed["supervision_mask"].sum()
    if mode == "full_3d_single_wrap":
        torch.testing.assert_close(original["surface_mask"], changed["surface_mask"], rtol=0, atol=0)


@pytest.mark.parametrize("finder", [find_default, find_subtiling])
@pytest.mark.parametrize("role", ["train", "validation"])
@pytest.mark.parametrize("mask_value", [1, 255])
def test_discovery_membership_ignores_excluded_ink(tmp_path, finder, role, mask_value):
    mapping = _config_mapping(depth=3, side=4)
    mapping.update(patch_overlap=0.5, patch_min_labeled_coverage=0.1,
                   datasets=[{"segments_path": str(tmp_path), "volume_scale": 0}])
    if finder is find_subtiling:
        mapping.update(patch_finding_type="subtiling", patch_finding_filter_empty_tile=True)
    config = InkConfig.from_mapping(mapping)
    segment = Segment(config.data, config.data.datasets[0], 0, "discovery", tmp_path,
                      "discovery", "image", "labels", "supervision", "validation")
    arrays = {"image": np.ones((3, 8, 8), dtype=np.uint8),
              "supervision": np.full((3, 8, 8), mask_value, dtype=np.uint8),
              "validation": np.zeros((3, 8, 8), dtype=np.uint8),
              "labels": np.zeros((3, 8, 8), dtype=np.uint8)}
    arrays["validation"][:, :, 4:] = mask_value
    arrays["labels"][:, 1:3, 1:3] = 255
    arrays["labels"][:, 1:3, 5:7] = 255
    excluded = arrays["validation"] > 0 if role == "train" else arrays["validation"] == 0
    original_labels = arrays["labels"].copy()
    def discover():
        training, validation = finder(segment, lambda path, resolution: arrays[str(path)])
        return [p.bbox for p in (training if role == "train" else validation)]
    original = discover()
    assert original
    for value in (0, 255):
        arrays["labels"] = original_labels.copy()
        arrays["labels"][excluded] = value
        assert discover() == original


def test_intersecting_segment_already_excludes_heldout_values(tmp_path):
    config, dataset, arrays = _native_fixture(tmp_path, "full_3d", False)
    primary = dataset.segments[0]
    other = replace(primary, segment_relpath="other", segment_dir=tmp_path / "other",
                    inklabels="other_labels", supervision_mask="other_supervision",
                    validation_mask="other_validation")
    for kind in ("labels", "supervision", "validation"):
        arrays[f"other_{kind}"] = arrays[kind].copy()
    arrays["labels"].fill(0)
    dataset._tifxyz_cache[str(other.segment_dir)] = dataset._tifxyz_cache[str(tmp_path)]
    key = (primary.dataset_idx, str(primary.image_volume), primary.scale)
    dataset._segments_by_volume[key] = [primary, other]
    original, loss, gradient = _load_and_loss(config, dataset)
    arrays["other_labels"][arrays["other_validation"] > 0] = 0
    mutated, other_loss, other_gradient = _load_and_loss(config, dataset)
    for name in original:
        torch.testing.assert_close(mutated[name], original[name], rtol=0, atol=0)
    torch.testing.assert_close(other_loss, loss, rtol=0, atol=0)
    torch.testing.assert_close(other_gradient, gradient, rtol=0, atol=0)
    arrays["other_labels"][:, :, 1] = 0
    permitted_change, _, permitted_gradient = _load_and_loss(config, dataset)
    assert not torch.equal(permitted_change["inklabels"], original["inklabels"])
    assert not torch.equal(permitted_gradient, gradient)


@pytest.mark.parametrize("mode", ["full_3d", "full_3d_single_wrap"])
def test_warm_cache_preserves_ordinary_training_scope(tmp_path, mode):
    config, dataset, arrays = _native_fixture(tmp_path, mode, False)
    cold, loss, gradient = _load_and_loss(config, dataset)
    path = tmp_path / "patches.json"
    save_patch_cache(path, dataset.patches)
    restored = load_patch_cache(path, config=config.data, segments=dataset.segments)
    assert restored is not None
    assert str(restored[0].supervision_mask_override) == str(restored[0].segment.supervision_mask)
    dataset.patches = restored
    warm, other_loss, other_gradient = _load_and_loss(config, dataset)
    for name in cold:
        torch.testing.assert_close(warm[name], cold[name], rtol=0, atol=0)
    torch.testing.assert_close(other_loss, loss, rtol=0, atol=0)
    torch.testing.assert_close(other_gradient, gradient, rtol=0, atol=0)
    arrays["labels"][:, :, 3] = 0
    changed, _, changed_gradient = _load_and_loss(config, dataset)
    assert changed["inklabels"].sum() < warm["inklabels"].sum()
    assert not torch.equal(changed_gradient, other_gradient)


def test_distinct_training_override_is_a_restrictive_scope(tmp_path):
    config, dataset, arrays = _native_fixture(tmp_path, "full_3d", False)
    arrays["scope"] = arrays["supervision"].copy()
    dataset.patches = [replace(dataset.patches[0], supervision_mask_override="scope")]
    original, loss, gradient = _load_and_loss(config, dataset)
    arrays["labels"][arrays["scope"] == 0] = 0
    changed, other_loss, other_gradient = _load_and_loss(config, dataset)
    for name in original:
        torch.testing.assert_close(changed[name], original[name], rtol=0, atol=0)
    torch.testing.assert_close(other_loss, loss, rtol=0, atol=0)
    torch.testing.assert_close(other_gradient, gradient, rtol=0, atol=0)


def test_empty_cache_is_recomputed(tmp_path):
    config, dataset, _ = _native_fixture(tmp_path, "full_3d", False)
    path = tmp_path / "empty.json"
    path.write_text("[]")
    assert load_patch_cache(path, config=config.data, segments=dataset.segments) is None


@pytest.mark.parametrize("is_validation", [False, True])
def test_intersecting_segment_without_validation_has_only_training_scope(tmp_path, is_validation):
    config, dataset, arrays = _native_fixture(tmp_path, "full_3d", is_validation)
    primary = dataset.segments[0]
    other = replace(primary, segment_relpath="other", segment_dir=tmp_path / "other",
                    inklabels="other_labels", supervision_mask="other_supervision",
                    validation_mask=None)
    arrays["labels"].fill(0)
    arrays["other_labels"] = np.zeros_like(arrays["labels"])
    arrays["other_labels"][:, :, 1] = 255
    arrays["other_supervision"] = arrays["supervision"].copy()
    arrays["other_validation"] = np.zeros_like(arrays["validation"])
    dataset._tifxyz_cache[str(other.segment_dir)] = dataset._tifxyz_cache[str(tmp_path)]
    key = (primary.dataset_idx, str(primary.image_volume), primary.scale)
    dataset._segments_by_volume[key] = [primary, other]
    original, loss, gradient = _load_and_loss(config, dataset)
    arrays["other_labels"].fill(0)
    erased, erased_loss, erased_gradient = _load_and_loss(config, dataset)
    if not is_validation:
        # Ordinary training must still receive this training-only segment.
        assert original["inklabels"].sum() > erased["inklabels"].sum()
        assert not torch.equal(gradient, erased_gradient)
        return
    for name in original:
        torch.testing.assert_close(erased[name], original[name], rtol=0, atol=0)
    torch.testing.assert_close(erased_loss, loss, rtol=0, atol=0)
    torch.testing.assert_close(erased_gradient, gradient, rtol=0, atol=0)

    arrays["other_labels"][:, :, 1] = 255
    other = replace(other, validation_mask="other_validation")
    dataset._segments_by_volume[key] = [primary, other]
    empty_validation, _, _ = _load_and_loss(config, dataset)
    for name in original:
        torch.testing.assert_close(empty_validation[name], original[name], rtol=0, atol=0)
    # A real held-out positive is retained once the other segment assigns it V.
    arrays["other_validation"][:, :, 1] = 255
    allowed, _, allowed_gradient = _load_and_loss(config, dataset)
    assert allowed["inklabels"].sum() > original["inklabels"].sum()
    assert not torch.equal(allowed_gradient, gradient)
    assert torch.any((allowed["inklabels"] == 0) & (allowed["supervision_mask"] > 0))


def test_subtiling_cache_rejects_a_changed_coverage_threshold(tmp_path):
    mapping = _config_mapping(depth=3, side=2)
    mapping.update(patch_finding_type="subtiling", patch_finding_filter_empty_tile=True,
                   patch_finding_tile_size=4, patch_finding_stride=4,
                   datasets=[{"segments_path": str(tmp_path), "volume_scale": 0}])
    config = InkConfig.from_mapping(mapping)
    segment = Segment(config.data, config.data.datasets[0], 0, "cache", tmp_path,
                      "cache", "image", "labels", "supervision", None)
    arrays = {"image": np.ones((3, 4, 4), dtype=np.uint8),
              "supervision": np.full((3, 4, 4), 255, dtype=np.uint8),
              "labels": np.zeros((3, 4, 4), dtype=np.uint8)}
    arrays["labels"][:, 0, 0] = 255
    opener = lambda path, resolution: arrays[str(path)]
    patches, _ = find_subtiling(segment, opener)
    assert len(patches) == 4
    path = tmp_path / "subtiling.json"
    save_patch_cache(path, patches)
    warm = load_patch_cache(path, config=config.data, segments=[segment])
    assert warm is not None
    assert [p.bbox for p in warm] == [p.bbox for p in patches]

    mapping["patch_min_labeled_coverage"] = 0.1
    changed_config = InkConfig.from_mapping(mapping)
    changed_segment = replace(segment, data_config=changed_config.data,
                              source=changed_config.data.datasets[0])
    changed, _ = find_subtiling(changed_segment, opener)
    assert len(changed) == 1
    assert patch_cache_path(config.data) != patch_cache_path(changed_config.data)
    assert load_patch_cache(path, config=changed_config.data, segments=[changed_segment]) is None


@pytest.mark.parametrize("finder", [find_default, find_subtiling])
def test_positive_mask_encodings_do_not_create_fully_heldout_training_patches(tmp_path, finder):
    mapping = _config_mapping(depth=3, side=4)
    mapping.update(patch_overlap=0.5, patch_min_labeled_coverage=0.0,
                   datasets=[{"segments_path": str(tmp_path), "volume_scale": 0}])
    if finder is find_subtiling:
        mapping.update(patch_finding_type="subtiling", patch_finding_filter_empty_tile=True)
    config = InkConfig.from_mapping(mapping)
    segment = Segment(config.data, config.data.datasets[0], 0, "masks", tmp_path,
                      "masks", "image", "labels", "supervision", "validation")
    arrays = {"image": np.ones((3, 8, 8), dtype=np.uint8),
              "supervision": np.full((3, 8, 8), 255, dtype=np.uint8),
              "validation": np.ones((3, 8, 8), dtype=np.uint8),
              "labels": np.full((3, 8, 8), 255, dtype=np.uint8)}
    training, validation = finder(segment, lambda path, resolution: arrays[str(path)])
    assert not training
    assert validation


@pytest.mark.parametrize("mode", ["full_3d", "full_3d_single_wrap"])
@pytest.mark.parametrize("is_validation", [False, True])
def test_excluded_values_do_not_change_model_input_or_deep_supervision(tmp_path, mode, is_validation):
    config, dataset, arrays = _native_fixture(tmp_path, mode, is_validation)
    authored = config.to_mapping()
    authored.update(num_iterations=3, seed=431, out_dir=str(tmp_path),
                    in_channels=2 if mode == "full_3d_single_wrap" else 1)
    training = TrainingConfig.from_mapping(resolve_training_mapping(authored))

    def objective():
        sample = dataset[0]
        batch = {name: value.unsqueeze(0) for name, value in sample.items()}
        model_input = prepare_model_input(batch, training)
        heads = [
            torch.linspace(-1.3 + index * 0.2, 1.7, int(np.prod(shape)))
            .reshape(1, 1, *shape).requires_grad_()
            for index, shape in enumerate(((5, 4, 4), (3, 2, 2), (2, 1, 1)))
        ]
        predictions, targets, ignore = prepare_loss_inputs(heads, batch, mode=mode)
        packed = concatenate_deep_supervision_ignore(targets, ignore, predictions)
        loss = DeepSupervisionWrapper(create_loss(config), deep_supervision_weights(3))(
            predictions, packed
        )
        loss.backward()
        gradients = [None if head.grad is None else head.grad.clone() for head in heads]
        assert all(torch.count_nonzero(gradient) for gradient in gradients[:2])
        assert gradients[2] is None  # Production gives the last decoder zero weight.
        return model_input, loss.detach(), gradients

    original_input, original_loss, original_gradients = objective()
    original_labels = arrays["labels"].copy()
    excluded = arrays["validation"] == 0 if is_validation else arrays["validation"] > 0
    for value in (0, 255):
        arrays["labels"] = original_labels.copy()
        arrays["labels"][excluded] = value
        model_input, loss, gradients = objective()
        torch.testing.assert_close(model_input, original_input, rtol=0, atol=0)
        torch.testing.assert_close(loss, original_loss, rtol=0, atol=0)
        for gradient, original_gradient in zip(gradients[:2], original_gradients[:2], strict=True):
            torch.testing.assert_close(gradient, original_gradient, rtol=0, atol=0)
