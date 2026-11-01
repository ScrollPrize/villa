"""Whole-dataset held-out validation: split, partition, isolation guard, cache, trainer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.data.dataset import InkDataset
from vesuvius.ink_detection.data.heldout import (
    assert_heldout_isolated,
    partition_heldout,
)
from vesuvius.ink_detection.types import Patch

from .test_data_foundation import _write_pyramid

DEPTH, SIDE = 3, 16


def _plane_volume(plane: np.ndarray) -> np.ndarray:
    volume = np.zeros((DEPTH, SIDE, SIDE), dtype=np.uint8)
    volume[DEPTH // 2] = plane
    return volume


def _write_segment(root: Path, name: str, *, supervision, validation=None, seed=0) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    image = rng.integers(1, 255, size=(DEPTH, SIDE, SIDE), dtype=np.uint8)
    _write_pyramid(directory / "surface.zarr", image)
    ink = (rng.random((SIDE, SIDE)) < 0.3).astype(np.uint8) * 255
    _write_pyramid(directory / f"{name}_inklabels.zarr", _plane_volume(ink))
    _write_pyramid(
        directory / f"{name}_supervision_mask.zarr", _plane_volume(supervision)
    )
    if validation is not None:
        _write_pyramid(
            directory / f"{name}_validation_mask.zarr", _plane_volume(validation)
        )


def _mask(rows: slice, cols: slice) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=np.uint8)
    mask[rows, cols] = 255
    return mask


@pytest.fixture
def roots(tmp_path):
    a, b = tmp_path / "scroll-a", tmp_path / "scroll-b"
    # Training segment: supervision on the left, its own validation mask on the right.
    _write_segment(
        a, "a", supervision=_mask(slice(None), slice(0, 9)),
        validation=_mask(slice(None), slice(10, 16)), seed=1,
    )
    # Held-out segment: supervision on top, a validation mask at the bottom.
    _write_segment(
        b, "b", supervision=_mask(slice(0, 8), slice(None)),
        validation=_mask(slice(12, 16), slice(None)), seed=2,
    )
    return a, b


def _source(root: Path, name: str, scroll: str | None, split: str | None = None,
            surface: Path | None = None) -> dict:
    source = {
        "segments_path": str(root),
        "volume_scale": 0,
        "segments": [name],
        "surface_volume_paths": {name: str(surface or root / name / "surface.zarr")},
    }
    if scroll is not None:
        source["sampling_scroll"] = scroll
    if split is not None:
        source["split"] = split
    return source


def _mapping(out_dir: Path, sources: list[dict], **extra) -> dict:
    mapping = {
        "mode": "flat",
        "patch_size": [DEPTH, 4, 4],
        "patch_overlap": 0.5,
        "patch_min_labeled_coverage": 0.0,
        "image_normalization": "none",
        "out_dir": str(out_dir),
        "dataloader_workers": 1,
        "datasets": sources,
    }
    mapping.update(extra)
    return mapping


def _dataset(out_dir: Path, sources: list[dict]) -> InkDataset:
    return InkDataset(InkDataConfig.from_mapping(_mapping(out_dir, sources)), do_augmentations=False)


def _signature(patch: Patch) -> tuple:
    return (
        patch.segment.segment_name,
        patch.bbox,
        patch.is_validation,
        None if patch.supervision_mask_override is None else str(patch.supervision_mask_override),
    )


def test_split_defaults_to_train_and_unknown_values_are_refused(tmp_path, roots):
    a, _ = roots
    config = InkDataConfig.from_mapping(_mapping(tmp_path, [_source(a, "a", "A")]))
    assert config.datasets[0].split == "train"
    with pytest.raises(ValueError, match="split must be 'train' or 'validation'"):
        InkDataConfig.from_mapping(_mapping(tmp_path, [_source(a, "a", "A", "test")]))


def test_unlabeled_datasets_cannot_be_held_out(tmp_path, roots):
    a, _ = roots
    with pytest.raises(ValueError, match=r"unlabeled_datasets\[0\]\.split must be 'train'"):
        InkDataConfig.from_mapping(_mapping(
            tmp_path, [], patch_discovery_mode="unlabeled",
            unlabeled_datasets=[_source(a, "a", "A", "validation")],
        ))


def test_heldout_validation_is_refused_outside_flat_mode(tmp_path, roots):
    a, b = roots
    with pytest.raises(ValueError, match="implemented for mode 'flat' only"):
        InkDataConfig.from_mapping(_mapping(
            tmp_path, [_source(a, "a", "A"), _source(b, "b", "B", "validation")],
            mode="full_3d",
        ))


def test_explicit_train_split_changes_nothing(tmp_path, roots):
    a, _ = roots
    implicit = _dataset(tmp_path / "implicit", [_source(a, "a", "A")])
    explicit = _dataset(tmp_path / "explicit", [_source(a, "a", "A", "train")])
    assert [_signature(p) for p in implicit.patches] == [_signature(p) for p in explicit.patches]
    # The default keeps validation windows inside the training segment.
    assert implicit.validation_patches and implicit.training_patches


def test_heldout_dataset_is_scored_whole_and_never_trained_on(tmp_path, roots):
    a, b = roots
    only_a = _dataset(tmp_path / "only-a", [_source(a, "a", "A")])
    only_b = _dataset(tmp_path / "only-b", [_source(b, "b", "B")])
    held = _dataset(tmp_path / "held", [_source(a, "a", "A"), _source(b, "b", "B", "validation")])

    # Training side: exactly the training-only run, same patches in the same order.
    assert [_signature(p) for p in held.training_patches] == [
        _signature(p) for p in only_a.training_patches
    ]
    # Validation side: every window b would have trained on, scored on its supervision
    # mask, plus b's validation-mask windows; nothing from a.
    b_supervision = str(only_b.segments[0].supervision_mask)
    expected = {
        (p.segment.segment_name, p.bbox, True, b_supervision) for p in only_b.training_patches
    } | {_signature(p) for p in only_b.validation_patches}
    assert {_signature(p) for p in held.validation_patches} == expected
    assert len(held.validation_patches) == len(expected)
    assert {p.segment.segment_name for p in held.validation_patches} == {"b"}


def test_partition_drops_validation_windows_inside_training_data(tmp_path, roots):
    a, _ = roots
    only_a = _dataset(tmp_path / "only-a", [_source(a, "a", "A")])
    assert only_a.validation_patches
    assert partition_heldout(only_a.patches) == only_a.training_patches


def test_promoted_patches_preserve_supervision_excluding_validation(tmp_path, roots):
    a, b = roots
    _write_segment(
        b, "overlapping", supervision=_mask(slice(None), slice(None)),
        validation=_mask(slice(None), slice(8, 16)), seed=3,
    )
    original = _dataset(tmp_path / "original", [_source(b, "overlapping", "B")])
    held = _dataset(tmp_path / "held", [
        _source(a, "a", "A"), _source(b, "overlapping", "B", "validation"),
    ])
    expected = {
        (p.bbox, p.is_validation): original[index]["supervision_mask"]
        for index, p in enumerate(original.patches)
    }
    promoted = {
        p.bbox: held[index]["supervision_mask"]
        for index, p in enumerate(held.patches)
        if p.exclude_validation_mask
    }
    checked_overlap = False
    for index, patch in enumerate(held.patches):
        if not patch.is_validation:
            continue
        was_validation = not patch.exclude_validation_mask
        actual = held[index]["supervision_mask"]
        assert torch.equal(actual, expected[patch.bbox, was_validation])
        if was_validation and patch.bbox in promoted:
            assert not torch.any((actual > 0) & (promoted[patch.bbox] > 0))
            checked_overlap = True
    assert checked_overlap


def test_isolation_guard_passes_and_reports(tmp_path, roots):
    a, b = roots
    held = _dataset(tmp_path / "held", [_source(a, "a", "A"), _source(b, "b", "B", "validation")])
    report = assert_heldout_isolated(
        held.training_patches, held.validation_patches, held.config, held.segments
    )
    assert report["shared_assets"] == 0
    assert report["training_scrolls"] == ["A"] and report["validation_scrolls"] == ["B"]
    assert report["validation_segments"] == ["1:b"]
    assert report["validation_patches"] == len(held.validation_patches)


@pytest.mark.parametrize(
    "scroll_a, scroll_b, surface_b, message",
    [
        ("A", "A", None, "scrolls that no training dataset uses"),
        (None, "B", None, r"datasets\[0\] must define sampling_scroll"),
        ("A", "B", "a", "shares image volumes"),
    ],
)
def test_isolation_guard_refuses_a_split_that_is_not_isolated(
    tmp_path, roots, scroll_a, scroll_b, surface_b, message
):
    a, b = roots
    surface = None if surface_b is None else a / "a" / "surface.zarr"
    held = _dataset(tmp_path / "held", [
        _source(a, "a", scroll_a), _source(b, "b", scroll_b, "validation", surface=surface),
    ])
    with pytest.raises(ValueError, match=message):
        assert_heldout_isolated(
            held.training_patches, held.validation_patches, held.config, held.segments
        )


@pytest.mark.parametrize("alias_kind", ["relative", "symlink"])
def test_isolation_guard_refuses_image_path_aliases(tmp_path, roots, monkeypatch, alias_kind):
    a, b = roots
    image = a / "a" / "surface.zarr"
    monkeypatch.chdir(tmp_path)
    if alias_kind == "relative":
        alias = image.relative_to(tmp_path)
    else:
        alias = tmp_path / "alias.zarr"
        alias.symlink_to(image, target_is_directory=True)
    held = _dataset(tmp_path / "held", [
        _source(a, "a", "A"), _source(b, "b", "B", "validation", surface=alias),
    ])
    with pytest.raises(ValueError, match="shares image volumes"):
        assert_heldout_isolated(
            held.training_patches, held.validation_patches, held.config, held.segments
        )


def test_isolation_guard_refuses_a_heldout_segment_with_no_patches(tmp_path, roots):
    a, b = roots
    held = _dataset(tmp_path / "held", [_source(a, "a", "A"), _source(b, "b", "B", "validation")])
    with pytest.raises(ValueError, match=r"found no patches for \['b'\]"):
        assert_heldout_isolated(held.training_patches, [], held.config, held.segments)


def test_isolation_guard_refuses_when_one_of_several_heldout_segments_is_missing(tmp_path, roots):
    a, b = roots
    _write_segment(b, "c", supervision=_mask(slice(4, 12), slice(None)), seed=3)
    two = _source(b, "b", "B", "validation")
    two["segments"] = ["b", "c"]
    two["surface_volume_paths"]["c"] = str(b / "c" / "surface.zarr")
    held = _dataset(tmp_path / "held", [_source(a, "a", "A"), two])
    only_b = [p for p in held.validation_patches if p.segment.segment_name == "b"]
    assert only_b and len(only_b) < len(held.validation_patches)
    with pytest.raises(ValueError, match=r"found no patches for \['c'\]"):
        assert_heldout_isolated(held.training_patches, only_b, held.config, held.segments)


def test_patch_cache_from_before_a_dataset_was_added_is_rebuilt(tmp_path, roots):
    a, b = roots
    out_dir = tmp_path / "shared-out"
    _dataset(out_dir, [_source(a, "a", "A")])  # writes a cache that knows only a
    held = _dataset(out_dir, [_source(a, "a", "A"), _source(b, "b", "B", "validation")])
    assert {p.segment.segment_name for p in held.validation_patches} == {"b"}
    assert_heldout_isolated(
        held.training_patches, held.validation_patches, held.config, held.segments
    )


def test_cache_written_while_a_dataset_trained_cannot_leak_it_into_training(tmp_path, roots):
    a, b = roots
    out_dir = tmp_path / "shared-out"
    both = _dataset(out_dir, [_source(a, "a", "A"), _source(b, "b", "B")])
    assert any(p.segment.segment_name == "b" for p in both.training_patches)
    held = _dataset(out_dir, [_source(a, "a", "A"), _source(b, "b", "B", "validation")])
    assert all(p.segment.segment_name == "a" for p in held.training_patches)
    assert {p.segment.segment_name for p in held.validation_patches} == {"b"}


def test_real_training_entry_validates_only_on_the_heldout_dataset(
    tmp_path, roots, monkeypatch, capsys
):
    import importlib.machinery
    import sys
    import types
    from types import SimpleNamespace

    import vesuvius.ink_detection.models.model as model_module
    import vesuvius.ink_detection.training.losses as losses_module
    import vesuvius.ink_detection.training.optimizers as optimizers_module
    import vesuvius.ink_detection.training.train as train_module
    from vesuvius.ink_detection.training.train import stage_training_request

    from .test_training_numerics import (
        _RecordingScheduler,
        _SyntheticTrainingLoss,
        _SyntheticTrainingModel,
        _training_mapping,
    )

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
    fake_wandb.run = None
    fake_wandb.init = lambda **kwargs: setattr(fake_wandb, "run", SimpleNamespace(id="fake"))
    fake_wandb.log = lambda values, step=None: None
    fake_wandb.Image = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("ACCELERATE_USE_CPU", "1")

    a, b = roots
    authored = _training_mapping()
    authored.update(_mapping(
        tmp_path / "output", [_source(a, "a", "A"), _source(b, "b", "B", "validation")],
        num_iterations=2, grad_acc_steps=1, mixed_precision="no", dataloader_workers=0,
        pin_memory=False, val_every=1, val_steps=99, val_preview_batches=0, save_every=2,
        log_every=1, wandb_project="test", wandb_entity=None,
    ))
    authored["model_config"]["pretrained_backbone"] = "synthetic"
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(authored), encoding="utf-8")

    monkeypatch.setattr(model_module, "make_model", lambda config: _SyntheticTrainingModel())
    monkeypatch.setattr(losses_module, "create_loss", lambda config: _SyntheticTrainingLoss())
    monkeypatch.setattr(optimizers_module, "create_training_optimizer",
                        lambda model, config: torch.optim.SGD(model.parameters(), lr=1e-9))
    monkeypatch.setattr(train_module, "create_training_scheduler",
                        lambda optimizer, config: _RecordingScheduler())

    assert train_module._run_training(stage_training_request(config_path)) == 0
    printed = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("heldout_validation=")
    ]
    assert len(printed) == 1, printed
    report = json.loads(printed[0].split("=", 1)[1])
    assert report["validation_segments"] == ["1:b"] and report["shared_assets"] == 0
    records = [
        json.loads(line)
        for line in (tmp_path / "output" / "validation_metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert records and all(np.isfinite(record["val_loss"]) for record in records)
