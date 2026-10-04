from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import pickle
import random
import shutil
import warnings

import numpy as np
import pytest
import torch
import zarr

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.data.dataset import InkDataset, flat_z_window_bbox
from vesuvius.ink_detection.data.geometry import (
    StoredResolutionIndex,
    catmull_rom_vertex_mask,
    filter_support_bands,
    filter_support_components,
    maybe_select_flat_pixels,
    native_tifxyz_pyramid_params,
    paste_bands,
    project_labels_and_supervision,
    read_tifxyz_on_flat_grid,
    select_flat_pixel_bands_via_stored_resolution,
    select_flat_pixels_via_stored_resolution,
)
from vesuvius.ink_detection.data.patch_cache import (
    label_asset_fingerprint,
    load_patch_cache,
    patch_finding_cache_token,
    save_patch_cache,
)
from vesuvius.ink_detection.data.patch_finding_default import (
    combined_patch_discovery_support,
    find_segment_patches,
    find_segment_unlabeled_patches,
    labeled_patch_coverage,
)
from vesuvius.ink_detection.data.patch_finding_subtiling import build_patch_index
from vesuvius.ink_detection.data.segment import (
    discover_segment_labels,
    gather_segments,
    parse_label_asset_path,
)
from vesuvius.ink_detection.types import Patch, Segment
from vesuvius.tifxyz import Tifxyz
from vesuvius.ink_detection.volume_io import (
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
    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        root.create_array("0", data=array, chunks=array.shape)
    else:
        root.create_dataset("0", data=array, chunks=array.shape)


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
    labeled_segment = replace(
        _segment(labeled_config, tmp_path, image_volume="image"),
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
    segment = replace(
        _segment(config, tmp_path),
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
    assert replace(patch, supervision_mask_override="").supervision_mask == ""
    assert "v6" in patch_finding_cache_token(config)

    changed = _config(tmp_path, patch_overlap=0.25)
    assert load_patch_cache(path, config=changed, segments=[segment]) is None


def _write_mask(path: Path, chunks: dict[str, bytes]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, payload in chunks.items():
        (path / name).write_bytes(payload)


def test_patch_cache_is_rejected_when_a_label_changes_under_the_same_path(tmp_path):
    """Paths alone cannot see a mask regenerated in place, which is how a split goes stale."""

    config = _config(tmp_path)
    mask = tmp_path / "supervision.zarr"
    _write_mask(mask, {".zarray": b"{}", "0.0.0": b"chunk-one", "0.0.1": b"chunk-two"})
    segment = replace(_segment(config, tmp_path), supervision_mask=mask)
    path = tmp_path / "patches.json"
    save_patch_cache(path, [Patch(segment=segment, bbox=(1, 2, 3, 4, 5, 6))])

    assert load_patch_cache(path, config=config, segments=[segment]) is not None

    # regenerated in place: same path, one chunk fewer
    (mask / "0.0.1").unlink()
    assert load_patch_cache(path, config=config, segments=[segment]) is None

    # rewritten to a different size under the same name
    _write_mask(mask, {"0.0.1": b"chunk-two"})
    save_patch_cache(path, [Patch(segment=segment, bbox=(1, 2, 3, 4, 5, 6))])
    assert load_patch_cache(path, config=config, segments=[segment]) is not None
    (mask / "0.0.1").write_bytes(b"chunk-two-but-longer")
    assert load_patch_cache(path, config=config, segments=[segment]) is None

    # rewritten to the same size under the same name: only the modification time moves
    save_patch_cache(path, [Patch(segment=segment, bbox=(1, 2, 3, 4, 5, 6))])
    assert load_patch_cache(path, config=config, segments=[segment]) is not None
    chunk = mask / "0.0.1"
    chunk.write_bytes(b"chunk-two-but-edited")
    os.utime(chunk, ns=(0, os.stat(chunk).st_mtime_ns + 1_000_000_000))
    assert load_patch_cache(path, config=config, segments=[segment]) is None


def test_patch_cache_still_hits_when_nothing_changed(tmp_path):
    """The fingerprint must not cost the fast path: an untouched tree keeps hitting."""

    config = _config(tmp_path)
    mask = tmp_path / "supervision.zarr"
    _write_mask(mask, {".zarray": b"{}", "0.0.0": b"chunk"})
    segment = replace(_segment(config, tmp_path), supervision_mask=mask)
    path = tmp_path / "patches.json"
    save_patch_cache(path, [Patch(segment=segment, bbox=(1, 2, 3, 4, 5, 6))])

    for _ in range(3):
        loaded = load_patch_cache(path, config=config, segments=[segment])
        assert loaded is not None and len(loaded) == 1


def test_label_fingerprint_is_stable_and_notices_each_asset(tmp_path):
    ink = tmp_path / "ink.zarr"
    supervision = tmp_path / "supervision.zarr"
    _write_mask(ink, {"0.0.0": b"a"})
    _write_mask(supervision, {"0.0.0": b"b"})

    baseline = label_asset_fingerprint([ink, supervision])
    assert baseline == label_asset_fingerprint([supervision, ink]), "order must not matter"
    assert baseline == label_asset_fingerprint([ink, supervision, None])

    (ink / "0.0.0").write_bytes(b"aa")
    assert label_asset_fingerprint([ink, supervision]) != baseline

    # a copy that kept modification times fingerprints as its source; a touch does not
    copied = tmp_path / "elsewhere" / "ink.zarr"
    shutil.copytree(ink, copied)
    assert label_asset_fingerprint([copied]) == label_asset_fingerprint([ink])
    stamp = os.stat(copied / "0.0.0").st_mtime_ns + 1_000_000_000
    os.utime(copied / "0.0.0", ns=(stamp, stamp))
    assert label_asset_fingerprint([copied]) != label_asset_fingerprint([ink])

    missing = label_asset_fingerprint([tmp_path / "absent.zarr"])
    assert missing and missing != label_asset_fingerprint([ink])


def test_unlabeled_coverage_key_is_rejected_and_cache_token_stays_compatible(tmp_path):
    unlabeled = _config(
        tmp_path,
        patch_discovery_mode="unlabeled",
        unlabeled_datasets=[
            {"segments_path": str(tmp_path), "volume_scale": 0}
        ],
    )
    assert patch_finding_cache_token(unlabeled) == (
        "unlabeled-default-v6-po-0.5-mdc-0.15-pfs-"
    )

    with pytest.raises(
        ValueError,
        match="threshold is fixed at 0.25.*not honored",
    ):
        _config(tmp_path, unlabeled_patch_min_data_coverage=0.9)


def test_volume_resolution_padding_and_disk_cache_boundary(tmp_path):
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

    if int(zarr.__version__.split(".", 1)[0]) < 3:
        with pytest.raises(
            NotImplementedError, match="volume disk cache requires zarr 3"
        ):
            open_volume(volume_path, 0, cache_dir=tmp_path / "cache")
        return

    cached = open_volume(
        volume_path,
        0,
        cache_dir=tmp_path / "cache",
        cache_max_gb=0.001,
    )
    np.testing.assert_array_equal(cached[:], array)
    assert any(path.is_file() for path in (tmp_path / "cache").rglob("*"))


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
    segment = replace(
        _segment(config, tmp_path, image_volume=image_path),
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
    indexed = select_flat_pixels_via_stored_resolution(
        tifxyz,
        (2, 3, 3, 3, 4, 4),
        coarse_native_pad=1,
        coarse_positions_zyx=tifxyz.stored_positions_zyx,
        coarse_valid=np.ones((4, 4), dtype=bool),
        native_coordinate_scale=0.25,
        flat_grid_stride=4,
        coarse_index=StoredResolutionIndex(
            tifxyz.stored_positions_zyx,
            np.ones((4, 4), dtype=bool),
            native_coordinate_scale=0.25,
        ),
    )
    assert indexed[0] == support_bbox
    np.testing.assert_array_equal(indexed[1], support)
    np.testing.assert_array_equal(indexed[2], support_valid)


@pytest.mark.parametrize("scale", [1.0, 0.25])
def test_stored_resolution_index_matches_full_grid_scan(scale):
    # A wavy sheet with NaN holes and invalid points; the index must return the
    # exact window a scan of the whole stored grid returns, including misses.
    rng = np.random.default_rng(7)
    height, width = 70, 90
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    positions = np.stack(
        [
            40 + 6 * np.sin(columns / 9.0) + rng.normal(0, 0.5, (height, width)),
            2.0 * rows + rng.normal(0, 0.5, (height, width)),
            2.0 * columns,
        ],
        axis=-1,
    ).astype(np.float32)
    positions[rng.random((height, width)) < 0.05] = np.nan
    positions[rng.random((height, width)) < 0.05] = -1
    valid = np.isfinite(positions).all(axis=-1) & (positions >= 0).all(axis=-1)
    scaled = positions * scale if scale != 1.0 else positions
    extent = np.array([60, 150, 190]) * scale
    for tile in (1, 16, 32, 128):
        index = StoredResolutionIndex(
            positions, valid, native_coordinate_scale=scale, tile=tile
        )
        for _ in range(200):
            start = rng.integers(-10, extent.astype(int))
            stop = start + rng.integers(1, (extent / 3).astype(int) + 2)
            bbox = (*start.tolist(), *stop.tolist())
            expected = maybe_select_flat_pixels(scaled, valid, bbox)
            assert index.window(bbox) == (None if expected is None else expected[0])


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


def _spiral_tifxyz(
    rng, scale: float, *, shift: float = 0.0, non_finite: bool = True
) -> Tifxyz:
    # A sheet wound several times around an axis, with invalid and non-finite
    # stored points, as a real Tifxyz (Catmull-Rom full resolution).
    step = 1.0 / scale
    height, width = round(120 * scale), round(1800 * scale)
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    start, pitch = 30.0 + shift, 25.0 / (2 * np.pi)
    theta = (np.sqrt(start**2 + 2 * pitch * columns * step) - start) / pitch
    radius = start + pitch * theta
    x = 140 + radius * np.cos(theta) + rng.normal(0, 0.3, theta.shape)
    y = 140 + radius * np.sin(theta) + rng.normal(0, 0.3, theta.shape)
    z = 10 + rows * step + 3 * np.sin(columns / 7.0) + rng.normal(0, 0.3, theta.shape)
    x, y, z = (values.astype(np.float32) for values in (x, y, z))
    for _ in range(3):
        row, column = rng.integers(0, height), rng.integers(0, width)
        hole = slice(row, row + 2), slice(column, column + 4)
        x[hole] = y[hole] = z[hole] = -1.0
    if non_finite:
        x[rng.integers(0, height), rng.integers(0, width)] = np.nan
    return Tifxyz(_x=x, _y=y, _z=z, _scale=(scale, scale)).use_full_resolution()


def _rippled_tifxyz(rng, scale: float) -> Tifxyz:
    # A sheet with diagonal ripples: a thin crop meets it in slanted stripes
    # whose bounding boxes overlap.
    step = 1.0 / scale
    height = width = round(240 * scale)
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = 20 + columns * step + rng.normal(0, 0.2, rows.shape)
    y = 20 + rows * step + rng.normal(0, 0.2, rows.shape)
    z = 80 + 40 * np.sin((rows + columns) * step / 26.0)
    x, y, z = (values.astype(np.float32) for values in (x, y, z))
    return Tifxyz(_x=x, _y=y, _z=z, _scale=(scale, scale)).use_full_resolution()


def _spiral_selection(tifxyz, resolution: int) -> dict:
    stride, coordinate_scale, coarse_pad = native_tifxyz_pyramid_params(resolution)
    coarse = np.asarray(tifxyz.get_zyxs(stored_resolution=True), dtype=np.float32)
    coarse_valid = np.isfinite(coarse).all(axis=-1) & (coarse >= 0).all(axis=-1)
    return dict(
        coarse_native_pad=coarse_pad,
        coarse_positions_zyx=coarse,
        coarse_valid=coarse_valid,
        native_coordinate_scale=coordinate_scale,
        flat_grid_stride=stride,
        required=False,
        coarse_index=StoredResolutionIndex(
            coarse, coarse_valid, native_coordinate_scale=coordinate_scale
        ),
    )


def _random_spiral_crop(rng, coordinate_scale: float = 1.0, *, slab: bool = False):
    if slab:  # thin in z, wide in y and x
        start = rng.integers((50, 20, 20), (110, 120, 120))
        size = rng.integers((4, 80, 80), (12, 140, 140))
    else:
        start = rng.integers((0, 0, 0), (130, 240, 240))
        size = rng.integers(8, 70, 3)
    start = (start * coordinate_scale).astype(int)
    size = np.maximum(2, (size * coordinate_scale).astype(int))
    return (*start.tolist(), *(start + size).tolist())


@pytest.mark.parametrize(
    ("surface", "resolution", "scale"),
    [
        (_spiral_tifxyz, 0, 0.1),
        (_spiral_tifxyz, 0, 0.05),
        (_spiral_tifxyz, 2, 0.1),
        (_spiral_tifxyz, 1, 0.25),
        (_rippled_tifxyz, 0, 0.25),
        (_rippled_tifxyz, 1, 0.25),
    ],
)
def test_support_bands_hold_exactly_the_points_of_the_whole_window(
    surface, resolution, scale
):
    # Crops through a spiral meet several windings; the bands must hold the same
    # in-crop points, with the same positions, as the one window spanning them.
    # Thin slabs through a rippled sheet meet it in slanted stripes whose boxes
    # overlap, which must end up in one band.
    rng = np.random.default_rng(11)
    tifxyz = surface(rng, scale)
    selection = _spiral_selection(tifxyz, resolution)
    vertex_mask = catmull_rom_vertex_mask(tifxyz)
    several = 0
    for _ in range(50):
        crop = _random_spiral_crop(
            rng, selection["native_coordinate_scale"], slab=surface is _rippled_tifxyz
        )
        whole = select_flat_pixels_via_stored_resolution(tifxyz, crop, **selection)
        bands = select_flat_pixel_bands_via_stored_resolution(
            tifxyz, crop, vertex_mask=vertex_mask, **selection
        )
        assert (whole is None) == (not bands)
        if whole is None:
            continue
        bbox, positions, valid = whole
        boxes = [band[0] for band in bands]
        assert bbox == (
            min(box[0] for box in boxes),
            max(box[1] for box in boxes),
            min(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        np.testing.assert_array_equal(
            paste_bands(boxes, [band[2] for band in bands], bbox, fill=False), valid
        )
        np.testing.assert_array_equal(
            paste_bands(boxes, [band[1] for band in bands], bbox, fill=np.nan)[valid],
            positions[valid],
        )
        for index, a in enumerate(boxes):
            for b in boxes[index + 1 :]:
                assert max(b[0] - a[1], a[0] - b[1], b[2] - a[3], a[2] - b[3]) >= 1
        several += len(bands) > 1
        (window,) = select_flat_pixel_bands_via_stored_resolution(
            tifxyz, crop, **selection
        )
        assert window[0] == bbox
        np.testing.assert_array_equal(window[2], valid)
    if surface is _spiral_tifxyz:
        assert several >= 5


def test_support_filter_drops_native_components_without_supervision():
    # Flat neighbours on two sheets that are apart in the volume: only the
    # supervised sheet is kept, although the flat grid connects them.
    positions = np.array(
        [[[0, 0, 0], [0, 0, 1], [0, 5, 2], [0, 5, 3]]], dtype=np.float32
    )
    supervision = np.array([[1, 0, 0, 0]], dtype=np.uint8)
    kept_bbox, keep = filter_support_bands(
        [((0, 1, 0, 4), positions, np.ones((1, 4), dtype=bool))],
        [supervision],
        crop_bbox_zyx=(0, 0, 0, 1, 6, 4),
        patch_bbox_zyx=(0, 0, 0, 1, 1, 4),
        max_supervision_grid_distance=None,
    )
    assert kept_bbox == (0, 1, 0, 2)
    np.testing.assert_array_equal(keep[0], np.array([[True, True, False, False]]))


def _flat_normals(tifxyz, bbox):
    nx, ny, nz = tifxyz.get_normals(*bbox)
    return np.stack([nz, ny, nx], axis=-1).astype(np.float32)


@pytest.mark.parametrize("max_distance", [None, 0.0, 3.0, 64.0])
def test_support_filter_and_projection_per_band_match_the_whole_window(max_distance):
    rng = np.random.default_rng(5)
    tifxyz = _spiral_tifxyz(rng, 0.1)
    selection = _spiral_selection(tifxyz, 0)
    vertex_mask = catmull_rom_vertex_mask(tifxyz)
    several = dropped = 0
    for _ in range(40):
        crop = _random_spiral_crop(rng)
        whole = select_flat_pixels_via_stored_resolution(tifxyz, crop, **selection)
        if whole is None:
            continue
        bbox, positions, valid = whole
        bands = select_flat_pixel_bands_via_stored_resolution(
            tifxyz, crop, vertex_mask=vertex_mask, **selection
        )
        boxes = [band[0] for band in bands]
        supervision = rng.random(valid.shape) < rng.choice([0.0, 0.001, 0.02, 0.9])
        supervision = supervision.astype(np.uint8)
        labels = (rng.random(valid.shape) < 0.3).astype(np.uint8)

        def in_band(flat, box):
            return flat[
                box[0] - bbox[0] : box[1] - bbox[0], box[2] - bbox[2] : box[3] - bbox[2]
            ]

        # Mostly a patch on one of the bands, sometimes one that misses them all.
        rows, columns = np.nonzero(valid)
        point = rng.integers(len(rows)) if rng.random() < 0.75 else None
        patch_y0 = bbox[0] - 60 if point is None else bbox[0] + int(rows[point]) - 10
        patch_x0 = bbox[2] - 60 if point is None else bbox[2] + int(columns[point]) - 10
        filtering = dict(
            crop_bbox_zyx=crop,
            patch_bbox_zyx=(
                0,
                patch_y0,
                patch_x0,
                1,
                patch_y0 + int(rng.integers(11, 40)),
                patch_x0 + int(rng.integers(11, 40)),
            ),
            max_supervision_grid_distance=max_distance,
        )
        expected_bbox, _, expected_valid, _, _ = filter_support_components(
            support_bbox_yx=bbox,
            positions_zyx=positions,
            valid_mask=valid,
            inklabels_flat=labels,
            supervision_flat=supervision,
            **filtering,
        )
        kept_bbox, keep = filter_support_bands(
            bands, [in_band(supervision, box) for box in boxes], **filtering
        )
        assert kept_bbox == expected_bbox
        np.testing.assert_array_equal(
            paste_bands(boxes, keep, kept_bbox, fill=False), expected_valid
        )
        several += len(bands) > 1
        dropped += np.count_nonzero(expected_valid) < np.count_nonzero(valid)

        projecting = dict(
            crop_bbox_zyx=crop, label_half_thickness=3.0, background_half_thickness=2.0
        )
        expected_labels, expected_supervision = project_labels_and_supervision(
            positions_zyx=positions,
            valid_mask=valid & (supervision > 0),
            inklabels_flat=labels,
            supervision_flat=supervision,
            normals_zyx=_flat_normals(tifxyz, bbox),
            **projecting,
        )
        band_labels = np.zeros_like(expected_labels)
        band_supervision = np.zeros_like(expected_supervision)
        for box, band_positions, band_valid in bands:
            projected = project_labels_and_supervision(
                positions_zyx=band_positions,
                valid_mask=band_valid & (in_band(supervision, box) > 0),
                inklabels_flat=in_band(labels, box),
                supervision_flat=in_band(supervision, box),
                normals_zyx=_flat_normals(tifxyz, box),
                **projecting,
            )
            np.maximum(band_labels, projected[0], out=band_labels)
            np.maximum(band_supervision, projected[1], out=band_supervision)
        np.testing.assert_array_equal(band_labels, expected_labels)
        np.testing.assert_array_equal(band_supervision, expected_supervision)
    assert several >= 5 and dropped >= 3


@pytest.mark.parametrize("max_distance", [None, 0.0, 2.0, 5.0, 64.0])
def test_filter_support_bands_matches_one_window_when_bands_are_close(max_distance):
    # A sheet cut into strips by narrow gaps: seeds of one strip are within reach
    # of its neighbours, so strips must be filtered together where it matters.
    rng = np.random.default_rng(23)
    dropped = 0
    for _ in range(60):
        height, width = int(rng.integers(6, 30)), int(rng.integers(20, 90))
        rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        positions = np.stack(
            [3.0 + rows, 12 + 6 * np.sin(columns / 5.0), 2 + 0.7 * columns], axis=-1
        ).astype(np.float32)
        valid = rng.random((height, width)) < 0.9
        cuts = np.zeros(width, dtype=bool)
        for start in rng.integers(1, width - 1, int(rng.integers(1, 5))):
            cuts[start : min(width - 1, start + int(rng.integers(1, 7)))] = True
        valid[:, cuts] = False
        edges = np.flatnonzero(np.diff(np.concatenate([[1], cuts, [1]]).astype(int)))
        bbox = (40, 40 + height, 100, 100 + width)
        strips = [
            (slice(None), slice(int(left), int(right)))
            for left, right in zip(edges[::2], edges[1::2])
        ]
        boxes = [
            (bbox[0], bbox[1], bbox[2] + strip[1].start, bbox[2] + strip[1].stop)
            for strip in strips
        ]
        supervision = rng.random((height, width)) < rng.choice([0.0, 0.01, 0.1, 0.9])
        supervision = supervision.astype(np.uint8)
        patch_x0 = bbox[2] + int(rng.integers(-10, width))
        crop_stop = int(rng.integers(8, 40)), 24, int(rng.integers(20, 70))
        filtering = dict(
            crop_bbox_zyx=(0, 0, 0, *crop_stop),
            patch_bbox_zyx=(0, bbox[0], patch_x0, 1, bbox[1], patch_x0 + 12),
            max_supervision_grid_distance=max_distance,
        )
        expected_bbox, _, expected_valid, _, _ = filter_support_components(
            support_bbox_yx=bbox,
            positions_zyx=positions,
            valid_mask=valid,
            inklabels_flat=supervision,
            supervision_flat=supervision,
            **filtering,
        )
        bands = [
            (box, positions[strip], valid[strip]) for box, strip in zip(boxes, strips)
        ]
        kept_bbox, keep = filter_support_bands(
            bands, [supervision[strip] for strip in strips], **filtering
        )
        assert kept_bbox == expected_bbox
        np.testing.assert_array_equal(
            paste_bands(boxes, keep, kept_bbox, fill=False), expected_valid
        )
        dropped += np.count_nonzero(expected_valid) < np.count_nonzero(valid)
    assert dropped >= 10


@pytest.mark.parametrize(
    ("mode", "resolution"),
    [("full_3d", 0), ("full_3d_single_wrap", 0), ("full_3d", 1)],
)
def test_native_samples_are_the_same_with_and_without_bands(
    tmp_path, monkeypatch, mode, resolution
):
    # Two spiral segments on one volume: samples assembled from per-band support
    # must equal the samples assembled from the window spanning the bands.
    rng = np.random.default_rng(3)
    config = InkDataConfig.from_mapping(
        {
            "mode": mode,
            "patch_size": [56, 56, 56],
            "patch_overlap": 0.25,
            "patch_min_labeled_coverage": 0.0,
            "image_normalization": "none",
            "full_3d": {"projection_half_thickness": 3 << resolution},
            "datasets": [
                {
                    "segments_path": str(tmp_path),
                    "volume_path": "volume",
                    "volume_scale": resolution,
                }
            ],
        }
    )
    stride = 1 << resolution
    volume_shape = (140 // stride + 1, 280 // stride + 1, 280 // stride + 1)
    arrays = {"volume": rng.integers(0, 255, volume_shape, dtype=np.uint8)}
    segments, surfaces = [], {}
    for name, shift in (("base", 0.0), ("other", 9.0)):
        tifxyz = _spiral_tifxyz(rng, 0.1, shift=shift, non_finite=False)
        flat_shape = tuple(-(-size // stride) for size in tifxyz.full_resolution_shape)
        arrays[f"{name}-ink"] = (rng.random((1, *flat_shape)) < 0.3).astype(np.uint8)
        arrays[f"{name}-supervision"] = (rng.random((1, *flat_shape)) < 0.6).astype(
            np.uint8
        )
        segment = Segment(
            config,
            config.datasets[0],
            0,
            name,
            tmp_path / name,
            name,
            "volume",
            Path(f"{name}-ink"),
            Path(f"{name}-supervision"),
        )
        segments.append(segment)
        surfaces[str(segment.segment_dir)] = tifxyz
    flat_height, flat_width = arrays["base-ink"].shape[1:]
    patches = [
        Patch(segments[0], (0, y, x, 56, y + 56, x + 56))
        for y in (0, flat_height - 56)
        for x in range(0, flat_width - 56, 150)
    ]

    def dataset(banded: bool) -> InkDataset:
        dataset = InkDataset(
            config, do_augmentations=False, patches=patches, segments=segments
        )
        dataset._tifxyz_cache.update(surfaces)
        if not banded:
            dataset._vertex_mask_cache.update(dict.fromkeys(surfaces))
        monkeypatch.setattr(
            dataset, "_open", lambda path, resolution: arrays[str(path)]
        )
        return dataset

    banded, whole = dataset(True), dataset(False)
    band_counts = []
    select = banded._support_bands

    def counted(segment, crop_bbox, *, required):
        bands = select(segment, crop_bbox, required=required)
        band_counts.append(len(bands))
        return bands

    monkeypatch.setattr(banded, "_support_bands", counted)
    produced = 0
    for patch in patches:
        expected, sample = whole._native_sample(patch), banded._native_sample(patch)
        assert (expected is None) == (sample is None)
        if sample is None:
            continue
        produced += 1
        assert sample.keys() == expected.keys()
        for key, value in sample.items():
            assert torch.equal(value, expected[key]), key
        assert sample["supervision_mask"].any()
    assert produced >= 5 and max(band_counts) > 1


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


def test_native_sample_retry_follows_replacement_chain(
    tmp_path, monkeypatch
):
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
            return None
        return {"patch_index": torch.tensor(patch_index)}

    class ScriptedRandom:
        def __init__(self, seed):
            self.replacement = 1 if seed == config.seed else 2

        def randrange(self, count):
            assert count == 3
            return self.replacement

    monkeypatch.setattr(dataset, "_native_sample", scripted_sample)
    monkeypatch.setattr("vesuvius.ink_detection.data.dataset.random.Random", ScriptedRandom)
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
