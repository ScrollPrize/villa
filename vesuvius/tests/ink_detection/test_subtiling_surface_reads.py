"""Reuse the two mask surfaces already needed by subtiling's index builder."""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.types import Segment
from vesuvius.ink_detection.data.patch_finding_subtiling import find_segment_patches


class CountingVolume:
    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self.values[key]


def fixture(tmp_path, validation=False, empty=False):
    config = InkDataConfig.from_mapping({
        "mode": "flat", "patch_size": [3, 2, 2], "patch_overlap": 1.0,
        "patch_min_labeled_coverage": 0.1, "image_normalization": "none",
        "out_dir": str(tmp_path), "dataloader_workers": 1,
        "datasets": [{"segments_path": str(tmp_path), "volume_scale": 0}],
        "patch_finding_type": "subtiling", "patch_finding_tile_size": 2,
        "patch_finding_stride": 2, "patch_finding_filter_empty_tile": True,
    })
    label = np.zeros((1, 8, 8), dtype=np.uint8) if empty else np.full((1, 8, 8), 3, dtype=np.uint8)
    supervision = np.ones((1, 8, 8), dtype=np.uint8)
    held_out = np.zeros_like(supervision)
    held_out[:, :, :4] = 1
    volumes = {"image": CountingVolume(label), "label": CountingVolume(label),
               "supervision": CountingVolume(supervision), "validation": CountingVolume(held_out)}
    segment = Segment(
        data_config=config, source=config.datasets[0], dataset_idx=0,
        segment_relpath="sample", segment_dir=tmp_path, segment_name="sample",
        image_volume="image", inklabels=Path("label"), supervision_mask=Path("supervision"),
        validation_mask=Path("validation") if validation else None,
    )
    return segment, volumes


def test_surfaces_read_once_and_patch_order_is_unchanged(tmp_path):
    segment, volumes = fixture(tmp_path)
    before = {name: volume.values.copy() for name, volume in volumes.items()}
    training, held_out = find_segment_patches(segment, lambda path, scale: volumes[str(path)])
    assert not held_out
    assert [p.bbox for p in training] == [(-1, y, x, 2, y + 2, x + 2) for y in range(0, 8, 2) for x in range(0, 8, 2)]
    assert volumes["label"].reads == [0]
    assert volumes["supervision"].reads == [0]
    for name, volume in volumes.items():
        np.testing.assert_array_equal(volume.values, before[name])


def test_validation_split_and_override_remain_identical(tmp_path):
    segment, volumes = fixture(tmp_path, validation=True)
    training, held_out = find_segment_patches(segment, lambda path, scale: volumes[str(path)])
    assert [p.bbox for p in training] == [(-1, y, x, 2, y + 2, x + 2) for y in range(0, 8, 2) for x in (4, 6)]
    assert [p.bbox for p in held_out] == [(-1, y, x, 2, y + 2, x + 2) for y in range(0, 8, 2) for x in (0, 2)]
    assert all(p.is_validation and p.supervision_mask_override == Path("validation") for p in held_out)
    assert volumes["label"].reads == [0] and volumes["supervision"].reads == [0]
    # Validation retains its existing per-patch access; no new whole-plane read.
    assert len(volumes["validation"].reads) == 16
    assert all(isinstance(key, tuple) for key in volumes["validation"].reads)


def test_empty_index_does_not_introduce_validation_reads(tmp_path):
    segment, volumes = fixture(tmp_path, validation=True, empty=True)
    with pytest.raises(ValueError, match="no valid patches"):
        find_segment_patches(segment, lambda path, scale: volumes[str(path)])
    assert volumes["validation"].reads == []
    assert volumes["label"].reads == [0] and volumes["supervision"].reads == [0]
