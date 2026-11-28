"""Focused regression coverage for unlabeled flat-patch discovery."""

from pathlib import Path

import numpy as np
import pytest

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.data.patch_finding_default import (
    _occupied_stride_origins,
    find_segment_unlabeled_patches,
)
from vesuvius.ink_detection.types import Segment


@pytest.mark.parametrize("stride", [1, 2, 4, 5, 128, 603])
def test_stride_origins_match_unique_coordinates(stride):
    rng = np.random.default_rng(2043)
    surfaces = [np.zeros(shape, np.uint8) for shape in ((0, 0), (0, 5), (1, 1), (7, 0), (11, 37))]
    surfaces += [
        rng.choice([0, -3, 2, -1], size=shape)
        for shape in ((1, 1), (1, 7), (8, 1), (15, 32), (31, 513), (544, 169))
    ]
    surfaces += [surfaces[-1][::-2, 1::3], np.array([[0.0, np.nan, -0.0, -np.inf]])]
    for surface in surfaces:
        y, x = np.nonzero(surface)
        expected = np.unique(np.stack((y // stride, x // stride), axis=1), axis=0) * stride
        np.testing.assert_array_equal(_occupied_stride_origins(surface, stride), expected)


def test_unlabeled_discovery_reads_image_plane_once_and_retains_mask_exclusion(tmp_path):
    config = InkDataConfig.from_mapping({
        "mode": "flat",
        "patch_size": [3, 2, 2],
        "patch_overlap": 1.0,
        "patch_min_labeled_coverage": 0.0,
        "patch_discovery_mode": "unlabeled",
        "unlabeled_datasets": [{"segments_path": str(tmp_path), "volume_scale": 0}],
    })

    class CountedImage:
        data = np.ones((3, 5, 7), np.uint8)
        shape = data.shape

        def __init__(self):
            self.reads = []

        def __getitem__(self, key):
            self.reads.append(key)
            return self.data[key]

    image = CountedImage()
    supervision = np.zeros((3, 5, 7), np.uint8)
    supervision[1, 0, 0] = 1
    validation = np.zeros_like(supervision)
    validation[1, 2, 4] = 1
    segment = Segment(
        data_config=config, source=config.unlabeled_datasets[0], dataset_idx=0,
        segment_relpath="segment", segment_dir=tmp_path, segment_name="segment",
        image_volume="image", supervision_mask=Path("supervision"), validation_mask=Path("validation"),
    )
    volumes = {"image": image, "supervision": supervision, "validation": validation}

    training, held_out = find_segment_unlabeled_patches(segment, lambda path, scale: volumes[str(path)])

    assert image.reads == [1]
    expected_origins = [(y, x) for y in range(0, 5, 2) for x in range(0, 7, 2) if (y, x) not in {(0, 0), (2, 4)}]
    assert [(patch.bbox[1], patch.bbox[2]) for patch in training] == expected_origins
    assert all(patch.is_unlabeled for patch in training)
    assert held_out == []
