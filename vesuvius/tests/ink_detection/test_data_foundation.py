from __future__ import annotations

from collections import Counter
from pathlib import Path
import pickle
import random
import warnings

import numpy as np
import pytest
import torch
import zarr

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.dataset import InkDataset, flat_z_window_bbox
from vesuvius.ink_detection.geometry import (
    filter_support_components,
    read_tifxyz_on_flat_grid,
    select_flat_pixels_via_stored_resolution,
)
from vesuvius.ink_detection.patch_cache import (
    load_patch_cache,
    patch_finding_cache_token,
    save_patch_cache,
)
from vesuvius.ink_detection.patch_finding_default import (
    combined_patch_discovery_support,
    find_segment_patches,
    find_segment_unlabeled_patches,
    labeled_patch_coverage,
)
from vesuvius.ink_detection.patch_finding_subtiling import build_patch_index
from vesuvius.ink_detection.segment import (
    build_matching_label_asset_path,
    discover_segment_labels,
    gather_segments,
    parse_label_asset_path,
)
from vesuvius.ink_detection.types import Patch, Segment
from vesuvius.ink_detection.volume_io import (
    DiskCachedMapping,
    open_volume,
    read_bbox_with_padding,
)


def _config(tmp_path: Path, **overrides) -> InkDataConfig:
    authored = {
        "mode": "flat",
        "patch_size": [3, 2, 2],
        "patch_overlap": 0.5,
        "patch_min_labeled_coverage": 0.0,
        "image_normalization": "none",
        "out_dir": str(tmp_path),
        "dataloader_workers": 1,
        "datasets": [
            {
                "segments_path": str(tmp_path),
                "volume_scale": 0,
            }
        ],
    }
    authored.update(overrides)
    return InkDataConfig.from_mapping(authored)


def _segment(
    config: InkDataConfig,
    tmp_path: Path,
    *,
    image_volume: str | Path = "image.zarr",
) -> Segment:
    return Segment(
        data_config=config,
        source=config.datasets[0],
        dataset_idx=0,
        segment_relpath="segment-a",
        segment_dir=tmp_path / "segment-a",
        segment_name="segment-a",
        image_volume=image_volume,
    )


def _write_pyramid(path: Path, array: np.ndarray) -> None:
    root = zarr.open_group(path, mode="w")
    root.create_array("0", data=array, chunks=array.shape)


def test_config_rejects_undefined_subtiling_branch(tmp_path):
    with pytest.raises(
        ValueError,
        match="patch_finding_type='subtiling'.*patch_finding_filter_empty_tile=true",
    ):
        _config(tmp_path, patch_finding_type="subtiling")


def test_config_requires_authored_patch_overlap(tmp_path):
    with pytest.raises(KeyError, match="patch_overlap"):
        InkDataConfig.from_mapping(
            {
                "mode": "flat",
                "patch_size": [3, 2, 2],
                "patch_min_labeled_coverage": 0.0,
                "datasets": [
                    {"segments_path": str(tmp_path), "volume_scale": 0}
                ],
            }
        )


def test_config_requires_authored_patch_min_labeled_coverage(tmp_path):
    with pytest.raises(KeyError, match="patch_min_labeled_coverage"):
        InkDataConfig.from_mapping(
            {
                "mode": "flat",
                "patch_size": [3, 2, 2],
                "patch_overlap": 0.25,
                "datasets": [
                    {"segments_path": str(tmp_path), "volume_scale": 0}
                ],
            }
        )


def test_config_preserves_authored_patch_values_and_default_finder(tmp_path):
    config = _config(
        tmp_path,
        patch_overlap="0.375",
        patch_min_labeled_coverage="0.125",
    )
    assert config.patch_finding.kind == "default"
    assert config.patch_finding.overlap == 0.375
    assert config.patch_finding.min_labeled_coverage == 0.125


def test_typed_config_and_dataset_are_pickle_safe_for_spawn_workers(tmp_path):
    config = _config(
        tmp_path,
        sampling_strategy="fixed_scroll_prior_stratified",
        seed=7,
        fixed_scroll_prior={
            "seed": 7,
            "target_batch_counts": {"first": 2, "second": 1},
        },
    )
    restored = pickle.loads(pickle.dumps(config))
    assert restored == config
    assert list(restored.sampling.fixed_batch_quotas.items()) == [
        ("first", 2),
        ("second", 1),
    ]
    with pytest.raises(TypeError):
        restored.sampling.fixed_batch_quotas["first"] = 9
    for do_augmentations in (False, True):
        dataset = InkDataset(
            restored,
            do_augmentations=do_augmentations,
            patches=[],
        )
        round_tripped_dataset = pickle.loads(pickle.dumps(dataset))
        assert round_tripped_dataset.config == restored
        assert round_tripped_dataset.do_augmentations is do_augmentations


def test_segment_asset_parsing_and_independent_auto_versions(tmp_path):
    segment_dir = tmp_path / "segment-a"
    segment_dir.mkdir()
    paths = [
        segment_dir / "segment-a_inklabels.zarr",
        segment_dir / "segment-a_inklabels_v3.zarr",
        segment_dir / "segment-a_supervision_mask_v2.zarr",
        segment_dir / "segment-a_validation_mask_v4.zarr",
    ]
    for path in paths:
        path.mkdir()
    parsed = parse_label_asset_path(paths[1])
    assert parsed["label_kind"] == "inklabels"
    assert parsed["version_num"] == 3
    assert build_matching_label_asset_path(
        paths[1], label_kind="supervision_mask"
    ).name == "segment-a_supervision_mask_v3.zarr"

    discovered = discover_segment_labels(_segment(_config(tmp_path), tmp_path))
    assert discovered.inklabels == paths[1]
    assert discovered.supervision_mask == paths[2]
    assert discovered.validation_mask == paths[3]


def test_explicit_label_version_requires_matching_required_assets(tmp_path):
    segment_dir = tmp_path / "segment-a"
    segment_dir.mkdir()
    (segment_dir / "segment-a_inklabels_v2.zarr").mkdir()
    config = _config(tmp_path, label_version="v2")
    with pytest.raises(ValueError, match="matching .zarr labels for version v2"):
        discover_segment_labels(_segment(config, tmp_path))


def test_segment_gathering_preserves_remote_and_explicit_volume_paths(tmp_path):
    native_root = tmp_path / "native-segments"
    native_segment = native_root / "native-a"
    native_segment.mkdir(parents=True)
    (native_segment / "x.tif").touch()
    (native_segment / "native-a_inklabels.zarr").mkdir()
    (native_segment / "native-a_supervision_mask.zarr").mkdir()
    remote_native = "s3://vesuvius-challenge-open-data/native.zarr/"
    native_config = InkDataConfig.from_mapping(
        {
            "mode": "full_3d",
            "patch_size": [3, 2, 2],
            "patch_overlap": 0.25,
            "patch_min_labeled_coverage": 0.0,
            "datasets": [
                {
                    "segments_path": str(native_root),
                    "volume_path": remote_native,
                    "volume_scale": 0,
                }
            ],
        }
    )
    assert [segment.image_volume for segment in gather_segments(native_config)] == [
        remote_native
    ]

    flat_root = tmp_path / "flat-segments"
    flat_segment = flat_root / "flat-a"
    flat_segment.mkdir(parents=True)
    (flat_segment / "flat-a_inklabels.zarr").mkdir()
    (flat_segment / "flat-a_supervision_mask.zarr").mkdir()
    remote_surface = "s3://vesuvius-challenge-open-data/surface.zarr/"
    flat_config = InkDataConfig.from_mapping(
        {
            "mode": "flat",
            "patch_size": [3, 2, 2],
            "patch_overlap": 0.25,
            "patch_min_labeled_coverage": 0.0,
            "datasets": [
                {
                    "segments_path": str(flat_root),
                    "segments": ["flat-a"],
                    "surface_volume_paths": {"flat-a": remote_surface},
                    "volume_scale": 0,
                }
            ],
        }
    )
    assert [segment.image_volume for segment in gather_segments(flat_config)] == [
        remote_surface
    ]


def test_patch_discovery_math_and_filter_empty_tile_subtiling():
    label = np.zeros((4, 4), dtype=np.uint8)
    label[1, 1] = 1
    label[2, 3] = 1
    assert labeled_patch_coverage(label) == pytest.approx(6 / 16)
    supervision = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    validation = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(
        combined_patch_discovery_support(supervision, validation),
        np.array([[True, False], [False, True]]),
    )

    subtiling_labels = np.zeros((8, 8), dtype=np.uint8)
    subtiling_labels[:4, :4] = 3
    _, xyxys, indices = build_patch_index(
        subtiling_labels,
        np.ones((8, 8), dtype=np.uint8),
        size=2,
        tile_size=4,
        stride=4,
        filter_empty_tile=True,
    )
    np.testing.assert_array_equal(
        xyxys,
        np.array(
            [[0, 0, 2, 2], [2, 0, 4, 2], [0, 2, 2, 4], [2, 2, 4, 4]],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(indices, np.full(4, -1, dtype=np.int32))
    with pytest.raises(
        ValueError,
        match="patch_finding_type.*patch_finding_filter_empty_tile",
    ):
        build_patch_index(
            subtiling_labels,
            np.ones((8, 8), dtype=np.uint8),
            size=2,
            tile_size=4,
            stride=4,
            filter_empty_tile=False,
        )


def test_default_labeled_and_unlabeled_patch_origins(tmp_path):
    labeled_config = _config(tmp_path, patch_size=[3, 2, 2], patch_overlap=1.0)
    labeled_segment = _segment(labeled_config, tmp_path, image_volume="image").with_labels(
        inklabels=Path("labels"),
        supervision_mask=Path("supervision"),
        validation_mask=Path("validation"),
    )
    image = np.zeros((3, 6, 6), dtype=np.uint8)
    image[1] = 1
    labels = np.zeros_like(image)
    labels[1, 1, 1] = 1
    supervision = np.zeros_like(image)
    supervision[1, 1, 1] = 1
    supervision[1, 4, 4] = 1
    validation = np.zeros_like(image)
    validation[1, 4, 4] = 1
    volumes = {
        "image": image,
        "labels": labels,
        "supervision": supervision,
        "validation": validation,
    }
    opener = lambda path, resolution: volumes[str(path)]
    training, held_out = find_segment_patches(labeled_segment, opener)
    assert [patch.bbox for patch in training] == [(0, 0, 0, 3, 2, 2)]
    assert [patch.bbox for patch in held_out] == [(0, 4, 4, 3, 6, 6)]

    unlabeled_config = _config(
        tmp_path,
        patch_size=[3, 2, 2],
        patch_overlap=1.0,
        patch_discovery_mode="unlabeled",
        datasets=[],
        unlabeled_datasets=[
            {"segments_path": str(tmp_path), "volume_scale": 0}
        ],
    )
    unlabeled_segment = Segment(
        data_config=unlabeled_config,
        source=unlabeled_config.unlabeled_datasets[0],
        dataset_idx=0,
        segment_relpath="segment-a",
        segment_dir=tmp_path / "segment-a",
        segment_name="segment-a",
        image_volume="image",
        supervision_mask=Path("supervision"),
        validation_mask=Path("validation"),
    )
    training, held_out = find_segment_unlabeled_patches(unlabeled_segment, opener)
    assert [patch.bbox for patch in training] == [
        (0, 0, 2, 3, 2, 4),
        (0, 0, 4, 3, 2, 6),
        (0, 2, 0, 3, 4, 2),
        (0, 2, 2, 3, 4, 4),
        (0, 2, 4, 3, 4, 6),
        (0, 4, 0, 3, 6, 2),
        (0, 4, 2, 3, 6, 4),
    ]
    assert held_out == []


def test_v6_patch_cache_round_trip_and_stale_rejection(tmp_path):
    config = _config(tmp_path)
    segment = _segment(config, tmp_path).with_labels(
        inklabels=tmp_path / "ink.zarr",
        supervision_mask=tmp_path / "supervision.zarr",
        validation_mask=tmp_path / "validation.zarr",
    )
    patch = Patch(
        segment=segment,
        bbox=(1, 2, 3, 4, 5, 6),
        is_validation=True,
        supervision_mask_override=segment.validation_mask,
    )
    path = tmp_path / "patches.json"
    save_patch_cache(path, [patch])
    loaded = load_patch_cache(path, config=config, segments=[segment])
    assert len(loaded) == 1
    assert loaded[0].segment is segment
    assert loaded[0].bbox == patch.bbox
    assert loaded[0].is_validation
    assert loaded[0].supervision_mask == str(segment.validation_mask)
    assert "v6" in patch_finding_cache_token(config)

    changed = _config(tmp_path, patch_overlap=0.25)
    assert load_patch_cache(path, config=changed, segments=[segment]) is None


def test_volume_resolution_padding_and_zarr3_disk_cache(tmp_path):
    volume_path = tmp_path / "volume.zarr"
    array = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    _write_pyramid(volume_path, array)
    volume = open_volume(volume_path, 0)
    np.testing.assert_array_equal(volume[:], array)
    crop, valid = read_bbox_with_padding(
        volume, (-1, 1, 3, 2, 5, 7), fill_value=9
    )
    assert crop.shape == (3, 4, 4)
    assert valid == (slice(1, 3), slice(0, 3), slice(0, 2))
    np.testing.assert_array_equal(crop[1:3, :3, :2], array[:2, 1:4, 3:5])

    cached = open_volume(
        volume_path,
        0,
        cache_dir=tmp_path / "cache",
        cache_max_gb=0.001,
    )
    np.testing.assert_array_equal(cached[:], array)
    assert any((tmp_path / "cache").rglob("*__*"))


def test_disk_cache_is_read_through_atomic_and_lru_bounded(tmp_path):
    class CountingSource(dict):
        def __init__(self):
            super().__init__({"a/chunk": b"a" * 8, "b/chunk": b"b" * 8})
            self.reads = Counter()

        def __getitem__(self, key):
            self.reads[key] += 1
            return super().__getitem__(key)

    source = CountingSource()
    cache = DiskCachedMapping(source, tmp_path / "bounded-cache", max_bytes=10)
    assert cache["a/chunk"] == b"a" * 8
    assert cache["b/chunk"] == b"b" * 8
    assert cache["b/chunk"] == b"b" * 8
    assert source.reads == {"a/chunk": 1, "b/chunk": 1}
    cached_files = [
        path
        for path in (tmp_path / "bounded-cache").iterdir()
        if path.name != ".cache.lock"
    ]
    assert [path.name for path in cached_files] == ["b__chunk"]
    assert not any(path.name.endswith(".tmp") for path in cached_files)


def test_flat_jitter_and_dataset_sample_are_self_contained(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        flat_z_window_jitter={
            "enabled": True,
            "window_depth": 3,
            "max_offset": 1,
            "probability": 1.0,
            "padding": "forbidden",
        },
    )
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randint", lambda low, high: -1)
    assert flat_z_window_bbox(
        (2, 0, 0, 5, 2, 2),
        config=config,
        do_augmentations=True,
        is_validation=False,
    ) == ((1, 0, 0, 4, 2, 2), -1)

    image_path = tmp_path / "image.zarr"
    labels_path = tmp_path / "labels.zarr"
    supervision_path = tmp_path / "supervision.zarr"
    validation_path = tmp_path / "validation.zarr"
    image = np.arange(7 * 2 * 2, dtype=np.uint8).reshape(7, 2, 2)
    labels = np.ones_like(image)
    supervision = np.ones_like(image)
    validation = np.zeros_like(image)
    validation[:, 0, 0] = 1
    for path, value in (
        (image_path, image),
        (labels_path, labels),
        (supervision_path, supervision),
        (validation_path, validation),
    ):
        _write_pyramid(path, value)
    segment = _segment(config, tmp_path, image_volume=image_path).with_labels(
        inklabels=labels_path,
        supervision_mask=supervision_path,
        validation_mask=validation_path,
    )
    patch = Patch(segment=segment, bbox=(2, 0, 0, 5, 2, 2))
    dataset = InkDataset(config, do_augmentations=False, patches=[patch])
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (1, 3, 2, 2)
    assert not bool(sample["is_unlabeled"])
    assert torch.count_nonzero(sample["supervision_mask"][:, :, 0, 0]) == 0


class _FakeTifxyz:
    def __init__(self, positions_zyx: np.ndarray):
        self.positions_zyx = positions_zyx
        self.full_resolution_shape = positions_zyx.shape[:2]

    def get_zyxs(self, *, stored_resolution: bool):
        assert stored_resolution
        return self.positions_zyx

    def __getitem__(self, index):
        positions = self.positions_zyx[index]
        valid = np.ones(positions.shape[:2], dtype=bool)
        return positions[..., 2], positions[..., 1], positions[..., 0], valid

    def get_normals(self, row_start, row_end, column_start, column_end):
        shape = row_end - row_start, column_end - column_start
        return (
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.ones(shape, dtype=np.float32),
        )


class _PyramidFakeTifxyz:
    def __init__(self, full_positions_zyx: np.ndarray, *, stride: int):
        self.full_positions_zyx = full_positions_zyx
        self.stored_positions_zyx = full_positions_zyx[::stride, ::stride]
        self.full_resolution_shape = full_positions_zyx.shape[:2]

    def __getitem__(self, index):
        positions = self.full_positions_zyx[index]
        valid = np.ones(positions.shape[:2], dtype=bool)
        return positions[..., 2], positions[..., 1], positions[..., 0], valid


def test_ragged_tifxyz_pyramid_refines_to_exact_flat_grid():
    side = 15
    rows = np.arange(side, dtype=np.float32)[:, None]
    columns = np.arange(side, dtype=np.float32)[None, :]
    full_positions = np.stack(
        [
            np.full((side, side), 8.0, dtype=np.float32),
            rows.repeat(side, axis=1),
            columns.repeat(side, axis=0),
        ],
        axis=-1,
    )
    tifxyz = _PyramidFakeTifxyz(full_positions, stride=4)
    sampled, sampled_valid = read_tifxyz_on_flat_grid(
        tifxyz,
        y0=0,
        y1=4,
        x0=0,
        x1=4,
        flat_grid_stride=4,
        native_coordinate_scale=0.25,
    )
    assert sampled.shape == (4, 4, 3)
    np.testing.assert_array_equal(
        sampled[3, 3], np.array([2.0, 3.0, 3.0], dtype=np.float32)
    )
    support_bbox, support, support_valid = select_flat_pixels_via_stored_resolution(
        tifxyz,
        (2, 3, 3, 3, 4, 4),
        coarse_native_pad=1,
        coarse_positions_zyx=tifxyz.stored_positions_zyx,
        coarse_valid=np.ones((4, 4), dtype=bool),
        native_coordinate_scale=0.25,
        flat_grid_stride=4,
    )
    assert support_bbox == (3, 4, 3, 4)
    assert sampled_valid.all() and support_valid.all()
    np.testing.assert_array_equal(
        support, np.array([[[2.0, 3.0, 3.0]]], dtype=np.float32)
    )


def test_patch_bbox_seeds_only_its_connected_support_component():
    positions = np.array(
        [[[0, 0, 0], [0, 0, 1], [0, 0, 5], [0, 0, 10], [0, 0, 11]]],
        dtype=np.float32,
    )
    valid = np.array([[True, True, False, True, True]])
    supervision = np.array([[1, 0, 0, 0, 1]], dtype=np.uint8)
    labels = supervision.copy()
    support_bbox, kept_positions, kept_valid, kept_labels, kept_supervision = (
        filter_support_components(
            support_bbox_yx=(0, 1, 0, 5),
            positions_zyx=positions,
            valid_mask=valid,
            inklabels_flat=labels,
            supervision_flat=supervision,
            crop_bbox_zyx=(0, 0, 0, 1, 1, 12),
            patch_bbox_zyx=(0, 0, 0, 1, 1, 2),
            max_supervision_grid_distance=None,
        )
    )
    assert support_bbox == (0, 1, 0, 2)
    np.testing.assert_array_equal(kept_positions, positions[:, :2])
    np.testing.assert_array_equal(kept_valid, np.array([[True, True]]))
    np.testing.assert_array_equal(kept_labels, np.array([[1, 0]], dtype=np.uint8))
    np.testing.assert_array_equal(
        kept_supervision, np.array([[1, 0]], dtype=np.uint8)
    )


@pytest.mark.parametrize(
    ("mode", "has_surface_mask"),
    [("full_3d", False), ("full_3d_single_wrap", True)],
)
def test_native_dataset_modes_use_shared_geometry(
    tmp_path, mode, has_surface_mask
):
    config = InkDataConfig.from_mapping(
        {
            "mode": mode,
            "patch_size": [3, 2, 2],
            "patch_overlap": 0.25,
            "patch_min_labeled_coverage": 0.0,
            "image_normalization": "none",
            "datasets": [
                {
                    "segments_path": str(tmp_path),
                    "volume_path": str(tmp_path / "native.zarr"),
                    "volume_scale": 0,
                }
            ],
        }
    )
    native_path = tmp_path / "native.zarr"
    labels_path = tmp_path / "labels.zarr"
    supervision_path = tmp_path / "supervision.zarr"
    _write_pyramid(native_path, np.ones((6, 6, 6), dtype=np.uint8) * 7)
    _write_pyramid(labels_path, np.ones((3, 2, 2), dtype=np.uint8))
    _write_pyramid(supervision_path, np.ones((3, 2, 2), dtype=np.uint8))
    segment_dir = tmp_path / "segment-native"
    segment_dir.mkdir()
    segment = Segment(
        data_config=config,
        source=config.datasets[0],
        dataset_idx=0,
        segment_relpath="segment-native",
        segment_dir=segment_dir,
        segment_name="segment-native",
        image_volume=native_path,
        inklabels=labels_path,
        supervision_mask=supervision_path,
    )
    patch = Patch(segment=segment, bbox=(0, 0, 0, 3, 2, 2))
    positions = np.array(
        [[[2, 2, 2], [2, 2, 3]], [[2, 3, 2], [2, 3, 3]]],
        dtype=np.float32,
    )
    dataset = InkDataset(config, do_augmentations=False, patches=[patch])
    dataset._tifxyz_cache[str(segment_dir)] = _FakeTifxyz(positions)
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (1, 3, 2, 2)
    assert tuple(sample["inklabels"].shape) == (1, 3, 2, 2)
    assert ("surface_mask" in sample) is has_surface_mask
    if has_surface_mask:
        assert tuple(sample["surface_mask"].shape) == (1, 3, 2, 2)
        assert sample["surface_mask"].max() == 1.0


def test_native_sample_retry_follows_replacement_chain(tmp_path, monkeypatch):
    config = InkDataConfig.from_mapping(
        {
            "mode": "full_3d",
            "patch_size": [1, 1, 1],
            "patch_overlap": 0.25,
            "patch_min_labeled_coverage": 0.0,
            "seed": 17,
            "datasets": [
                {
                    "segments_path": str(tmp_path),
                    "volume_path": "unused",
                    "volume_scale": 0,
                }
            ],
        }
    )
    segment = Segment(
        data_config=config,
        source=config.datasets[0],
        dataset_idx=0,
        segment_relpath="segment",
        segment_dir=tmp_path / "segment",
        segment_name="segment",
        image_volume="unused",
    )
    patches = [
        Patch(segment=segment, bbox=(0, 0, index, 1, 1, index + 1))
        for index in range(3)
    ]
    dataset = InkDataset(config, do_augmentations=False, patches=patches)
    attempts = []

    def scripted_sample(patch):
        patch_index = patch.bbox[2]
        attempts.append(patch_index)
        if patch_index < 2:
            raise ValueError("No valid tifxyz points found for patch")
        return {"patch_index": torch.tensor(patch_index)}

    class ScriptedRandom:
        def __init__(self, seed):
            self.replacement = 1 if seed == config.seed else 2

        def randrange(self, count):
            assert count == 3
            return self.replacement

    monkeypatch.setattr(dataset, "_native_sample", scripted_sample)
    monkeypatch.setattr("vesuvius.ink_detection.dataset.random.Random", ScriptedRandom)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sample = dataset[0]
    assert sample["patch_index"].item() == 2
    assert attempts == [0, 1, 2]
    assert len(caught) == 2
    assert "patch idx 0" in str(caught[0].message)
    assert "resampling idx 1" in str(caught[0].message)
    assert "patch idx 1" in str(caught[1].message)
    assert "resampling idx 2" in str(caught[1].message)


def test_full_3d_merges_intersecting_segment_supervision(tmp_path, monkeypatch):
    config = InkDataConfig.from_mapping(
        {
            "mode": "full_3d",
            "patch_size": [3, 3, 4],
            "patch_overlap": 0.25,
            "patch_min_labeled_coverage": 0.0,
            "full_3d": {"projection_half_thickness": 0},
            "datasets": [
                {
                    "segments_path": str(tmp_path),
                    "volume_path": "volume-a",
                    "volume_scale": 0,
                }
            ],
        }
    )
    source = config.datasets[0]
    base = Segment(
        config,
        source,
        0,
        "base",
        tmp_path / "base",
        "base",
        "volume-a",
        Path("base-ink"),
        Path("base-supervision"),
    )
    other = Segment(
        config,
        source,
        0,
        "other",
        tmp_path / "other",
        "other",
        "volume-a",
        Path("other-ink"),
        Path("other-supervision"),
    )
    patch = Patch(base, (4, 10, 20, 7, 13, 24))
    rows = np.arange(3, dtype=np.float32)[:, None]
    columns = np.arange(4, dtype=np.float32)[None, :]
    positions = np.stack(
        [
            np.full((3, 4), 5.0, dtype=np.float32),
            (10.0 + rows).repeat(4, axis=1),
            (20.0 + columns).repeat(3, axis=0),
        ],
        axis=-1,
    )
    other_supervision = np.zeros((3, 3, 4), dtype=np.uint8)
    other_supervision[1, 1, 1] = 1
    other_supervision[1, 2, 2] = 1
    other_ink = np.zeros_like(other_supervision)
    other_ink[1, 2, 2] = 1
    arrays = {
        "other-supervision": other_supervision,
        "other-ink": other_ink,
    }
    dataset = InkDataset(
        config,
        do_augmentations=False,
        patches=[patch],
        segments=[base, other],
    )
    dataset._tifxyz_cache[str(other.segment_dir)] = _FakeTifxyz(positions)
    monkeypatch.setattr(dataset, "_open", lambda path, resolution: arrays[str(path)])
    labels, supervision = dataset._merge_intersecting(
        patch,
        patch.bbox,
        np.zeros((3, 3, 4), dtype=np.float32),
        np.zeros((3, 3, 4), dtype=np.float32),
    )
    expected_supervision = np.zeros((3, 3, 4), dtype=np.float32)
    expected_supervision[1, 1, 1] = 1
    expected_supervision[1, 2, 2] = 1
    expected_labels = np.zeros_like(expected_supervision)
    expected_labels[1, 2, 2] = 1
    np.testing.assert_array_equal(supervision, expected_supervision)
    np.testing.assert_array_equal(labels, expected_labels)
