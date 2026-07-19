from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from PIL import Image

from vesuvius.models.benchmarks.benchmark_support_mask import (
    evaluate_plane,
    render_support_mask_preview,
    run_benchmark,
)
from vesuvius.models.run.mask_predictions import (
    _spatial_chunks,
    mask_finalized_predictions,
)
from vesuvius.models.run.finalize_outputs import (
    FinalizeConfig,
    add_support_mask_arguments,
    apply_finalization,
    apply_support_mask,
    finalize_logits,
    resolve_support_mask,
    validate_support_volume,
)


def test_apply_support_mask_counts_multichannel_spatial_voxels_once() -> None:
    output = np.arange(1, 17, dtype=np.uint8).reshape(2, 2, 2, 2)
    support = np.array(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[0.0, 3.0], [4.0, 5.0]],
        ],
        dtype=np.float32,
    )

    masked, stats = apply_support_mask(output, support)

    expected = output.copy()
    expected[:, support <= 0.0] = 0
    np.testing.assert_array_equal(masked, expected)
    assert stats == {
        "spatial_voxels": 8,
        "supported_voxels": 5,
        "unsupported_voxels": 3,
        "nonzero_voxels_before": 8,
        "nonzero_voxels_after": 5,
        "nonzero_voxels_removed": 3,
    }


def test_benchmark_reports_phantom_removal_and_preservation() -> None:
    prediction = np.array(
        [[255, 255, 0, 255], [7, 0, 255, 255]],
        dtype=np.uint8,
    )
    support = np.array(
        [[0, 6, 0, 5], [9, 0, 10, np.nan]],
        dtype=np.float32,
    )

    metrics = evaluate_plane(
        prediction,
        support,
        prediction_threshold=0,
        support_threshold=5,
    )

    assert metrics == {
        'spatial_voxels': 8,
        'supported_voxels': 3,
        'support_fraction': 3 / 8,
        'positive_voxels_before': 6,
        'positive_voxels_after': 3,
        'phantom_positives_before': 3,
        'phantom_positives_after': 0,
        'phantom_fraction_before': 0.5,
        'phantom_fraction_after': 0.0,
        'supported_values_preserved': True,
        'nonzero_voxels_removed': 3,
    }


def test_benchmark_preview_renders_before_effect_and_after(tmp_path) -> None:
    prediction = np.array(
        [[255, 255, 0, 0], [255, 255, 0, 0]],
        dtype=np.uint8,
    )
    support = np.array(
        [[0, 6, 0, 0], [0, 6, 0, 0]],
        dtype=np.float32,
    )
    metrics = evaluate_plane(
        prediction,
        support,
        prediction_threshold=0,
        support_threshold=5,
    )
    output_path = tmp_path / 'support-mask-preview.png'

    render_support_mask_preview(
        prediction,
        support,
        metrics,
        output_path,
        z_index=7,
        prediction_threshold=0,
        support_threshold=5,
        label='fixture',
        panel_size=4,
    )

    with Image.open(output_path) as image:
        assert image.mode == 'RGB'
        colors = {
            tuple(pixel) for pixel in np.asarray(image).reshape(-1, 3)
        }
    assert (0, 0, 0) in colors
    assert (255, 255, 255) in colors
    assert (255, 0, 255) in colors


@pytest.mark.parametrize(
    ('output_image', 'image_plane', 'message'),
    [
        ('preview.png', None, 'must be provided together'),
        (None, 0, 'must be provided together'),
        ('preview.png', 2, 'must be one of the requested planes'),
    ],
)
def test_benchmark_rejects_invalid_image_options_before_opening_inputs(
    output_image: str | None,
    image_plane: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_benchmark(
            'unused-prediction.zarr',
            'unused-support.zarr',
            [0],
            output_image=None if output_image is None else Path(output_image),
            image_plane=image_plane,
        )


def test_apply_support_mask_accepts_singleton_channel_and_masks_nan() -> None:
    output = np.array([[[[7, 8, 9]]]], dtype=np.uint8)
    support = np.array([[[[1.0, np.nan, 2.0]]]], dtype=np.float32)

    masked, stats = apply_support_mask(output, support)

    np.testing.assert_array_equal(masked, np.array([[[[7, 0, 9]]]], dtype=np.uint8))
    assert stats == {
        "spatial_voxels": 3,
        "supported_voxels": 2,
        "unsupported_voxels": 1,
        "nonzero_voxels_before": 3,
        "nonzero_voxels_after": 2,
        "nonzero_voxels_removed": 1,
    }


def test_apply_support_mask_requires_values_strictly_above_threshold() -> None:
    output = np.full((1, 1, 1, 4), 255, dtype=np.uint8)
    support = np.array([[[0.49, 0.5, 0.5001, -1.0]]], dtype=np.float32)

    masked, stats = apply_support_mask(output, support, threshold=0.5)

    np.testing.assert_array_equal(
        masked,
        np.array([[[[0, 0, 255, 0]]]], dtype=np.uint8),
    )
    assert stats["supported_voxels"] == 1
    assert stats["unsupported_voxels"] == 3


def test_apply_support_mask_does_not_mutate_inputs() -> None:
    output = np.array([[[[1, 2], [3, 4]]]], dtype=np.uint8)
    support = np.array([[[[1.0, np.nan], [0.0, 2.0]]]], dtype=np.float32)
    output_before = output.copy()
    support_before = support.copy()

    apply_support_mask(output, support)

    np.testing.assert_array_equal(output, output_before)
    np.testing.assert_array_equal(support, support_before)


def test_support_mask_is_applied_after_low_probability_threshold() -> None:
    # Equal binary logits produce foreground probability 0.5. At the 0.2
    # threshold from issue #1114 that becomes a positive prediction, so masking
    # logits instead of the finalized result would recreate the phantom voxel.
    logits = np.array(
        [
            [[[-1.0, 0.0, 1.0]]],
            [[[1.0, 0.0, -1.0]]],
        ],
        dtype=np.float32,
    )
    finalized, is_empty = apply_finalization(
        logits,
        num_classes=2,
        config=FinalizeConfig(mode="binary", threshold=0.2),
    )

    assert not is_empty
    assert finalized[0, 0, 0, 1] == 255

    masked, _ = apply_support_mask(
        finalized,
        np.array([[[1, 0, 1]]], dtype=np.uint8),
    )
    assert masked[0, 0, 0, 1] == 0


def test_support_mask_catches_saturated_logits_at_gaussian_corner() -> None:
    from vesuvius.models.run.blending import generate_gaussian_map

    # The current Gaussian uses sigma=patch/8. At an outer 3-D corner its
    # weight is smaller than process_chunk's normalization epsilon, so even a
    # source-side +20/-20 background encoding is attenuated toward equal logits.
    epsilon = 1e-8
    corner_weight = float(generate_gaussian_map((16, 16, 16))[0, 0, 0, 0])
    assert 0.0 < corner_weight < epsilon

    saturated_background = np.array(
        [20.0, -20.0], dtype=np.float32
    ).reshape(2, 1, 1, 1)
    blended_logits = (
        saturated_background * corner_weight / (corner_weight + epsilon)
    )

    finalized, is_empty = apply_finalization(
        blended_logits,
        num_classes=2,
        config=FinalizeConfig(mode="binary", threshold=0.2),
    )
    assert not is_empty
    assert finalized.item() == 255

    masked, stats = apply_support_mask(
        finalized,
        np.zeros((1, 1, 1), dtype=np.uint8),
    )
    assert masked.item() == 0
    assert stats["nonzero_voxels_removed"] == 1


def test_constant_nonzero_logits_are_not_treated_as_empty() -> None:
    output, is_empty = apply_finalization(
        np.full((1, 2, 2, 2), 2.0, dtype=np.float32),
        num_classes=1,
        config=FinalizeConfig(mode='binary', threshold=0.5),
    )

    assert not is_empty
    np.testing.assert_array_equal(output, np.full((1, 2, 2, 2), 255, dtype=np.uint8))


def test_constant_nonzero_multiclass_label_is_not_treated_as_empty() -> None:
    logits = np.zeros((2, 2, 2, 2), dtype=np.float32)
    logits[1] = 2.0

    output, is_empty = apply_finalization(
        logits,
        num_classes=2,
        config=FinalizeConfig(mode='multiclass', threshold=0.5),
    )

    assert not is_empty
    np.testing.assert_array_equal(output, np.ones((1, 2, 2, 2), dtype=np.uint8))


def test_thresholded_multiclass_preserves_class_ids_across_chunks() -> None:
    homogeneous = np.zeros((3, 1, 1, 2), dtype=np.float32)
    homogeneous[1] = 2.0
    mixed = homogeneous.copy()
    mixed[:, 0, 0, 1] = (0.0, 0.0, 3.0)

    homogeneous_output, homogeneous_empty = apply_finalization(
        homogeneous,
        num_classes=3,
        config=FinalizeConfig(mode='multiclass', threshold=0.5),
    )
    mixed_output, mixed_empty = apply_finalization(
        mixed,
        num_classes=3,
        config=FinalizeConfig(mode='multiclass', threshold=0.5),
    )

    assert not homogeneous_empty
    assert not mixed_empty
    np.testing.assert_array_equal(
        homogeneous_output,
        np.array([[[[1, 1]]]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        mixed_output,
        np.array([[[[1, 2]]]], dtype=np.uint8),
    )


def test_mask_predictions_default_chunks_align_prediction_and_support() -> None:
    prediction = SimpleNamespace(
        shape=(8398, 3941, 3941),
        chunks=(192, 192, 192),
        dtype=np.dtype('uint8'),
    )
    support = SimpleNamespace(
        shape=(8398, 3941, 3941),
        chunks=(128, 128, 128),
        dtype=np.dtype('uint8'),
    )

    assert _spatial_chunks(prediction, support, None) == (384, 384, 384)


def test_mask_predictions_invalid_workers_preserve_existing_output(tmp_path) -> None:
    shape = (2, 2, 2)
    prediction_path = tmp_path / 'prediction.zarr'
    support_path = tmp_path / 'support.zarr'
    output_path = tmp_path / 'existing-output.zarr'

    prediction = zarr.open(
        str(prediction_path),
        mode='w',
        shape=shape,
        chunks=shape,
        dtype='u1',
        zarr_format=2,
    )
    prediction[:] = 1
    support = zarr.open(
        str(support_path),
        mode='w',
        shape=shape,
        chunks=shape,
        dtype='u1',
        zarr_format=2,
    )
    support[:] = 1
    existing_output = zarr.open(
        str(output_path),
        mode='w',
        shape=shape,
        chunks=shape,
        dtype='u1',
        zarr_format=2,
    )
    existing_output[:] = 37
    existing_output.attrs['sentinel'] = 'preserve'

    with pytest.raises(ValueError, match='num_workers must be at least 1'):
        mask_finalized_predictions(
            str(prediction_path),
            str(output_path),
            str(support_path),
            num_workers=0,
            verbose=False,
        )

    preserved_output = zarr.open(str(output_path), mode='r')
    assert preserved_output.attrs['sentinel'] == 'preserve'
    np.testing.assert_array_equal(
        preserved_output[:],
        np.full(shape, 37, dtype=np.uint8),
    )


@pytest.mark.parametrize(
    ('prediction_path', 'output_path'),
    [
        ('artifact.zarr', 'artifact.zarr/0'),
        ('artifact.zarr/0', 'artifact.zarr'),
        ('s3://bucket/artifact.zarr/0', 's3://bucket/artifact.zarr'),
        ('s3://bucket/artifact.zarr', 's3://bucket/artifact.zarr/0'),
    ],
)
def test_mask_predictions_rejects_overlapping_store_paths(
    prediction_path: str,
    output_path: str,
) -> None:
    with pytest.raises(ValueError, match='must not equal, contain, or be contained'):
        mask_finalized_predictions(
            prediction_path,
            output_path,
            'separate-support.zarr',
        )


def test_mask_predictions_rejects_resolved_local_alias(tmp_path) -> None:
    prediction_path = tmp_path / 'artifact.zarr'
    output_alias = prediction_path / '..' / prediction_path.name

    with pytest.raises(ValueError, match='must not equal, contain, or be contained'):
        mask_finalized_predictions(
            str(prediction_path),
            str(output_alias),
            str(tmp_path / 'support.zarr'),
        )


_OVERLAPPING_STORE_PATHS = [
    ('artifact.zarr', 'artifact.zarr'),
    ('artifact.zarr/0', 'artifact.zarr'),
    ('artifact.zarr', 'artifact.zarr/0'),
]


@pytest.mark.parametrize(
    ('output_relative', 'support_relative'),
    _OVERLAPPING_STORE_PATHS,
)
def test_finalize_logits_rejects_support_output_overlap_before_io(
    tmp_path,
    monkeypatch,
    output_relative: str,
    support_relative: str,
) -> None:
    def fail_if_opened(*args, **kwargs):
        raise AssertionError('overlap validation must run before opening any store')

    monkeypatch.setattr(
        'vesuvius.models.run.finalize_outputs.open_zarr',
        fail_if_opened,
    )

    with pytest.raises(
        ValueError,
        match='must not equal, contain, or be contained by support_volume_path',
    ):
        finalize_logits(
            input_path=str(tmp_path / 'missing-logits.zarr'),
            output_path=str(tmp_path / output_relative),
            support_volume_path=str(tmp_path / support_relative),
            num_workers=1,
            verbose=False,
        )


@pytest.mark.parametrize(
    ('output_relative', 'support_relative'),
    _OVERLAPPING_STORE_PATHS,
)
def test_fused_finalize_rejects_support_output_overlap_before_scanning(
    tmp_path,
    output_relative: str,
    support_relative: str,
) -> None:
    from vesuvius.models.run.blending import merge_inference_outputs

    with pytest.raises(
        ValueError,
        match='must not equal, contain, or be contained by support_volume_path',
    ):
        merge_inference_outputs(
            parent_dir=str(tmp_path / 'missing-parts'),
            output_path=str(tmp_path / output_relative),
            num_workers=1,
            verbose=False,
            finalize_config=FinalizeConfig(
                mode='binary',
                support_volume_path=str(tmp_path / support_relative),
            ),
        )


def _support_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['binary', 'multiclass'], default='binary')
    add_support_mask_arguments(parser)
    return parser


def test_support_cli_defaults_to_anonymous_public_s3() -> None:
    parser = _support_parser()
    args = parser.parse_args(['--support-volume', 's3://bucket/ct.zarr/2'])

    assert resolve_support_mask(parser, args) == (
        's3://bucket/ct.zarr/2',
        0.0,
        True,
    )


def test_support_cli_can_request_authenticated_s3_and_threshold() -> None:
    parser = _support_parser()
    args = parser.parse_args(
        [
            '--support-volume',
            's3://bucket/private.zarr/2',
            '--support-threshold',
            '5',
            '--support-authenticated',
        ]
    )

    assert resolve_support_mask(parser, args) == (
        's3://bucket/private.zarr/2',
        5.0,
        False,
    )


@pytest.mark.parametrize(
    'arguments',
    [
        ['--support-threshold', '5'],
        ['--support-authenticated'],
        ['--mode', 'multiclass', '--support-volume', 'ct.zarr'],
        ['--support-volume', 'ct.zarr', '--support-threshold', 'nan'],
    ],
)
def test_support_cli_rejects_invalid_combinations(arguments: list[str]) -> None:
    parser = _support_parser()
    args = parser.parse_args(arguments)

    with pytest.raises(SystemExit):
        resolve_support_mask(parser, args)


def test_finalize_logits_applies_local_support_zarr_and_records_metadata(
    tmp_path,
) -> None:
    import zarr

    input_path = tmp_path / 'logits.zarr'
    support_path = tmp_path / 'support.zarr'
    output_path = tmp_path / 'final.zarr'

    logits = np.zeros((2, 2, 2, 4), dtype=np.float32)
    logits[1] = np.linspace(-0.5, 0.5, 16, dtype=np.float32).reshape(2, 2, 4)
    support = np.array(
        [
            [[0, 6, 5, 7], [8, 0, 9, 5]],
            [[10, 11, 0, 0], [5, 12, 13, 0]],
        ],
        dtype=np.uint8,
    )

    logits_store = zarr.open_array(
        str(input_path),
        mode='w',
        shape=logits.shape,
        chunks=logits.shape,
        dtype=logits.dtype,
        zarr_format=2,
    )
    logits_store[:] = logits
    support_store = zarr.open_array(
        str(support_path),
        mode='w',
        shape=support.shape,
        chunks=support.shape,
        dtype=support.dtype,
        zarr_format=2,
    )
    support_store[:] = support

    finalize_logits(
        input_path=str(input_path),
        output_path=str(output_path),
        mode='binary',
        threshold=0.2,
        chunk_size=support.shape,
        num_workers=1,
        verbose=False,
        support_volume_path=str(support_path),
        support_threshold=5,
    )

    output_store = zarr.open_array(str(output_path), mode='r')
    expected = np.where(support > 5, 255, 0).astype(np.uint8)[np.newaxis]
    np.testing.assert_array_equal(output_store[:], expected)
    assert output_store.attrs['support_mask_applied'] is True
    assert output_store.attrs['support_threshold'] == 5.0
    assert output_store.attrs['support_mask_stats'] == {
        'spatial_voxels': 16,
        'supported_voxels': 8,
        'unsupported_voxels': 8,
        'nonzero_voxels_before': 16,
        'nonzero_voxels_after': 8,
        'nonzero_voxels_removed': 8,
        'phantom_fraction_before': 0.5,
        'scope': 'chunks_with_nonempty_finalized_output',
    }


def test_fused_blend_and_finalize_applies_support_mask(tmp_path) -> None:
    import zarr

    from vesuvius.models.run.blending import merge_inference_outputs

    parts_path = tmp_path / 'parts'
    parts_path.mkdir()
    logits_path = parts_path / 'logits_part_0.zarr'
    coords_path = parts_path / 'coordinates_part_0.zarr'
    support_path = tmp_path / 'support.zarr'
    output_path = tmp_path / 'fused.zarr'

    patch_size = (2, 2, 4)
    logits = np.zeros((1, 2, *patch_size), dtype=np.float32)
    logits[0, 1] = np.linspace(-0.5, 0.5, 16, dtype=np.float32).reshape(
        patch_size
    )
    support = np.array(
        [
            [[0, 6, 5, 7], [8, 0, 9, 5]],
            [[10, 11, 0, 0], [5, 12, 13, 0]],
        ],
        dtype=np.uint8,
    )

    logits_store = zarr.open_array(
        str(logits_path),
        mode='w',
        shape=logits.shape,
        chunks=logits.shape,
        dtype=logits.dtype,
        zarr_format=2,
    )
    logits_store[:] = logits
    logits_store.attrs['patch_size'] = list(patch_size)
    logits_store.attrs['original_volume_shape'] = list(patch_size)

    coords_store = zarr.open_array(
        str(coords_path),
        mode='w',
        shape=(1, 3),
        chunks=(1, 3),
        dtype=np.int32,
        zarr_format=2,
    )
    coords_store[:] = np.array([[0, 0, 0]], dtype=np.int32)

    support_store = zarr.open_array(
        str(support_path),
        mode='w',
        shape=support.shape,
        chunks=support.shape,
        dtype=support.dtype,
        zarr_format=2,
    )
    support_store[:] = support

    merge_inference_outputs(
        parent_dir=str(parts_path),
        output_path=str(output_path),
        chunk_size=patch_size,
        num_workers=1,
        verbose=False,
        finalize_config=FinalizeConfig(
            mode='binary',
            threshold=0.2,
            support_volume_path=str(support_path),
            support_threshold=5,
        ),
    )

    output_store = zarr.open_array(str(output_path), mode='r')
    expected = np.where(support > 5, 255, 0).astype(np.uint8)[np.newaxis]
    np.testing.assert_array_equal(output_store[:], expected)
    assert output_store.attrs['fused_blend_finalize'] is True
    assert output_store.attrs['support_mask_stats'] == {
        'spatial_voxels': 16,
        'supported_voxels': 8,
        'unsupported_voxels': 8,
        'nonzero_voxels_before': 16,
        'nonzero_voxels_after': 8,
        'nonzero_voxels_removed': 8,
        'phantom_fraction_before': 0.5,
        'scope': 'chunks_with_nonempty_finalized_output',
    }


def test_mask_existing_prediction_accepts_zarr_v3_support(tmp_path) -> None:
    import zarr

    prediction_path = tmp_path / 'published-prediction.zarr'
    support_path = tmp_path / 'support-v3.zarr'
    output_path = tmp_path / 'corrected.zarr'
    prediction = np.full((2, 2, 4), 255, dtype=np.uint8)
    support = np.array(
        [
            [[0, 6, 5, 7], [8, 0, 9, 5]],
            [[10, 11, 0, 0], [5, 12, 13, 0]],
        ],
        dtype=np.uint8,
    )

    prediction_store = zarr.open_array(
        str(prediction_path),
        mode='w',
        shape=prediction.shape,
        chunks=(1, 2, 2),
        dtype=prediction.dtype,
        zarr_format=2,
    )
    prediction_store[:] = prediction
    prediction_store.attrs['artifact_id'] = 'fixture'
    support_store = zarr.open_array(
        str(support_path),
        mode='w',
        shape=support.shape,
        chunks=(1, 2, 2),
        dtype=support.dtype,
        zarr_format=3,
    )
    support_store[:] = support

    summary = mask_finalized_predictions(
        str(prediction_path),
        str(output_path),
        str(support_path),
        support_threshold=5,
        num_workers=1,
        verbose=False,
    )

    expected = np.where(support > 5, prediction, 0)
    output_store = zarr.open_array(str(output_path), mode='r')
    np.testing.assert_array_equal(output_store[:], expected)
    assert output_store.attrs['artifact_id'] == 'fixture'
    assert output_store.attrs['support_alignment_validation'] == (
        'shape_only_physical_alignment_asserted_by_caller'
    )
    assert summary == {
        'nonzero_voxels_before': 16,
        'nonzero_voxels_after': 8,
        'nonzero_voxels_removed': 8,
        'phantom_fraction_before': 0.5,
        'scope': 'all_nonzero_input_predictions',
    }
    assert output_store.attrs['support_mask_stats'] == summary


@pytest.mark.parametrize(
    ("output", "support", "message"),
    [
        (
            np.zeros((2, 2, 2), dtype=np.uint8),
            np.zeros((2, 2, 2), dtype=np.float32),
            "output_np must have shape",
        ),
        (
            np.zeros((1, 2, 2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
            "support_np must have shape",
        ),
        (
            np.zeros((1, 2, 2, 2), dtype=np.uint8),
            np.zeros((2, 2, 2, 2), dtype=np.float32),
            "singleton channel dimension",
        ),
        (
            np.zeros((1, 2, 2, 2), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.float32),
            "spatial shape mismatch",
        ),
    ],
)
def test_apply_support_mask_rejects_bad_shapes(
    output: np.ndarray,
    support: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        apply_support_mask(output, support)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_apply_support_mask_rejects_nonfinite_threshold(threshold: float) -> None:
    output = np.ones((1, 1, 1, 1), dtype=np.uint8)
    support = np.ones((1, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="support threshold must be finite"):
        apply_support_mask(output, support, threshold=threshold)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((2, 3, 4), (2, 3, 4)),
        ((1, 2, 3, 4), (1, 2, 3, 4)),
    ],
)
def test_validate_support_volume_accepts_aligned_shapes(
    shape: tuple[int, ...],
    expected: tuple[int, ...],
) -> None:
    support_store = SimpleNamespace(shape=shape)

    assert validate_support_volume(support_store, (2, 3, 4)) == expected


@pytest.mark.parametrize("shape", [(2, 3), (2, 2, 3, 4), (1, 1, 2, 3, 4)])
def test_validate_support_volume_rejects_invalid_dimensionality(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="Support volume must have shape"):
        validate_support_volume(SimpleNamespace(shape=shape), (2, 3, 4))


def test_validate_support_volume_rejects_missing_array_shape() -> None:
    with pytest.raises(ValueError, match="must point to a Zarr array"):
        validate_support_volume(object(), (2, 3, 4))


def test_validate_support_volume_rejects_spatial_mismatch() -> None:
    with pytest.raises(ValueError, match="spatial shape mismatch"):
        validate_support_volume(SimpleNamespace(shape=(2, 3, 5)), (2, 3, 4))
