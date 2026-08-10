"""CPU witnesses for the five ink-detection curation command leaves."""

from __future__ import annotations

import numpy as np

from vesuvius.ink_detection.preprocessing import clean_labels
from vesuvius.ink_detection.preprocessing import composite_from_zarr
from vesuvius.ink_detection.preprocessing import download_required_zarr_chunks
from vesuvius.ink_detection.preprocessing import merge_predictions
from vesuvius.ink_detection.preprocessing import validate_segments


def test_clean_labels_preserves_categorical_foreground() -> None:
    source = np.array([[0, 1, 255], [0, 0, 0]], dtype=np.uint8)
    assert clean_labels.normalize_mask_image(source).tolist() == [[0, 255, 255], [0, 0, 0]]


def test_composite_projection_encodes_reference_uint8_range() -> None:
    source = np.array([[[0, 100]], [[255, 200]]], dtype=np.uint8)
    assert composite_from_zarr._project_block(source, "max").tolist() == [[255, 200]]
    assert composite_from_zarr._to_uint8(np.array([[0.0, 255.0]])).tolist() == [[0, 255]]


def test_prediction_merge_and_chunk_coverage_are_deterministic() -> None:
    inputs = [np.array([[0, 255]], dtype=np.uint8), np.array([[255, 0]], dtype=np.uint8)]
    assert merge_predictions.merge_soft_mean_chunk(inputs).tolist() == [[128, 128]]
    assert download_required_zarr_chunks.collect_unique_chunk_ids(
        [{"world_bbox": (0, 1, 1, 4, 5, 5)}],
        chunk_shape_zyx=(2, 2, 2),
        array_shape_zyx=(4, 6, 6),
    ) == ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2))


def test_validate_segments_accepts_reference_binary_values() -> None:
    assert validate_segments._normalize_version_id("v2") == 2
    assert validate_segments.ALLOWED_BINARY_LABEL_VALUES == frozenset({0, 1, 255})
