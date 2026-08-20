from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from vesuvius.ink_detection.data.augmentations import (
    build_augmentations,
    maybe_translate_crop_bbox,
)
from vesuvius.ink_detection.config import InkDataConfig, NormalizationConfig
from vesuvius.ink_detection.data.geometry import (
    _draw_line,
    compute_native_crop_bbox,
    project_binary_mask_along_normals,
    project_flat_patch,
    project_labels_and_supervision,
    project_surface_distance,
)
from vesuvius.ink_detection.training.metrics import BalancedAccuracy, Confusion
from vesuvius.ink_detection.data.normalization import normalize_image
from vesuvius.ink_detection.training.samplers import (
    FixedScrollPriorStratifiedBatchSampler,
    build_sampling_policy,
    hierarchical_scroll_segment_weights,
)
from vesuvius.ink_detection.types import MetricBatch, Patch, Segment


@pytest.mark.parametrize(
    ("authored", "expected"),
    [
        ("none", np.array([0, 10, 255], dtype=np.float32)),
        ({"mode": "divide", "divisor": 255}, np.array([0, 10, 255], dtype=np.float32) / 255),
        (
            {"mode": "clip_zscore", "clip_min": 10, "clip_max": 200, "mean": 50, "std": 2},
            (np.array([10, 10, 200], dtype=np.float32) - 50) / 2,
        ),
    ],
)
def test_normalization_modes_with_fixed_anchors(authored, expected):
    image = np.array([0, 10, 255], dtype=np.uint8)
    np.testing.assert_allclose(
        normalize_image(image, NormalizationConfig.from_value(authored)), expected
    )


def test_robust_span_minmax_and_mad_degenerate_anchors():
    image = np.array([0, 1, 2, 3, 100], dtype=np.float32)
    span = normalize_image(
        image.copy(),
        NormalizationConfig.from_value(
            {"mode": "robust_percentile_span", "percentile_lower": 0, "percentile_upper": 80}
        ),
    )
    clipped = np.clip(image, 0, np.percentile(image, 80))
    expected = (clipped - np.median(clipped)) / (0.5 * np.percentile(image, 80))
    np.testing.assert_allclose(span, expected, rtol=1e-6, atol=1e-6)
    minmax = normalize_image(image.copy(), NormalizationConfig.from_value("minmax"))
    np.testing.assert_allclose(minmax, (image - image.min()) / np.ptp(image))
    percentile_minmax = normalize_image(
        image.copy(),
        NormalizationConfig.from_value(
            {"mode": "percentile_minmax", "percentile_lower": 0, "percentile_upper": 80}
        ),
    )
    clipped_percentile = np.clip(image, 0, np.percentile(image, 80))
    np.testing.assert_allclose(
        percentile_minmax,
        clipped_percentile / np.percentile(image, 80),
    )
    mad = normalize_image(
        np.ones(4, dtype=np.float32) * 7,
        NormalizationConfig.from_value("robust_mad"),
    )
    np.testing.assert_array_equal(mad, np.zeros(4, dtype=np.float32))
    zero_mad_nonconstant = normalize_image(
        np.array([0, 0, 0, 0, 10], dtype=np.float32),
        NormalizationConfig.from_value(
            {
                "mode": "robust_mad",
                "percentile_lower": 0,
                "percentile_upper": 100,
            }
        ),
    )
    np.testing.assert_allclose(
        zero_mad_nonconstant,
        np.array([0, 0, 0, 0, 2.5], dtype=np.float32),
    )
    empty = normalize_image(
        np.empty((0,), dtype=np.float32),
        NormalizationConfig.from_value(
            {"mode": "percentile_minmax", "percentile_lower": 2, "percentile_upper": 98}
        ),
    )
    assert empty.shape == (0,)
    with pytest.raises(ValueError, match="percentiles"):
        NormalizationConfig.from_value(
            {"mode": "minmax", "percentile_lower": 100, "percentile_upper": 0}
        )
    minmax = NormalizationConfig.from_value(
        {"mode": "minmax", "percentile_lower": 0, "percentile_upper": 100}
    )
    np.testing.assert_allclose(
        normalize_image(image.copy(), minmax),
        (image - image.min()) / np.ptp(image),
    )


def test_native_crop_scatter_surface_and_dense_normal_projection():
    positions = np.array(
        [[[2, 1, 1], [2, 1, 4]], [[2, 4, 1], [2, 4, 4]]],
        dtype=np.float32,
    )
    valid = np.ones((2, 2), dtype=bool)
    assert compute_native_crop_bbox(positions, valid, (5, 6, 6)) == (0, 0, 0, 5, 6, 6)
    scattered = project_flat_patch(
        np.array([[1, 2], [3, 4]], dtype=np.uint8),
        positions,
        valid,
        (0, 0, 0, 5, 6, 6),
    )
    assert scattered[2, 4, 4] == 4
    surface = project_surface_distance(
        positions, valid, (0, 0, 0, 5, 6, 6), max_distance_voxels=2
    )
    assert surface.dtype == np.float32
    assert surface[2, 1, 1] == 1.0

    normals = np.zeros((2, 2, 3), dtype=np.float32)
    normals[..., 0] = 1
    labels, supervision = project_labels_and_supervision(
        positions_zyx=positions,
        valid_mask=valid,
        inklabels_flat=np.ones((2, 2), dtype=np.uint8),
        supervision_flat=np.ones((2, 2), dtype=np.uint8),
        crop_bbox_zyx=(0, 0, 0, 5, 6, 6),
        normals_zyx=normals,
        label_half_thickness=1.0,
        background_half_thickness=1.0,
    )
    expected = np.zeros((5, 6, 6), dtype=np.float32)
    expected[1:4, 1:5, 1:5] = 1
    np.testing.assert_array_equal(labels, expected)
    np.testing.assert_array_equal(supervision, expected)


def test_line_projection_preserves_reciprocal_multiply_rounding():
    """The unmarked voxel at index 5 is a parity contract, not a target.

    _draw_line divides by a reciprocal multiply whose rounding skips one
    lattice voxel on this span. The native full-3D reference pipeline
    rasterized training labels with exactly this kernel, so parity with
    checkpoints trained on those labels requires keeping the behavior.
    """
    output = np.zeros((1, 1, 8), dtype=np.uint8)
    _draw_line(
        output,
        np.array([0, 0, 0], dtype=np.float32),
        np.array([0, 0, 7], dtype=np.float32),
        0,
        0,
        0,
    )
    assert output[0, 0].tolist() == [1, 1, 1, 1, 1, 0, 1, 1]
    assert output[0, 0, 5] == 0


def _marked_voxels(output):
    return sorted(tuple(coordinate) for coordinate in np.argwhere(output).tolist())


def test_normal_projection_marks_axis_aligned_column_with_truncation():
    output = project_binary_mask_along_normals(
        np.ones((1, 1), dtype=np.uint8),
        np.array([[[2.3, 1.6, 2.4]]], dtype=np.float32),
        np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32),
        np.ones((1, 1), dtype=bool),
        (0, 0, 0, 6, 4, 4),
        half_thickness_voxels=2.0,
    )
    assert _marked_voxels(output) == [
        (0, 1, 2),
        (1, 1, 2),
        (2, 1, 2),
        (3, 1, 2),
        (4, 1, 2),
    ]


def test_normal_projection_normalizes_diagonals_and_connects_neighbors():
    output = project_binary_mask_along_normals(
        np.ones((1, 2), dtype=np.uint8),
        np.array([[[2.4, 1.3, 1.2], [2.4, 1.3, 2.8]]], dtype=np.float32),
        np.array([[[0.0, 2.0, 2.0], [0.0, 2.0, 2.0]]], dtype=np.float32),
        np.ones((1, 2), dtype=bool),
        (0, 0, 0, 5, 4, 6),
        half_thickness_voxels=1.25,
    )
    assert _marked_voxels(output) == [
        (2, 0, 0),
        (2, 0, 1),
        (2, 0, 2),
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, 1),
        (2, 2, 2),
        (2, 2, 3),
    ]


def test_normal_projection_fills_slab_between_offset_sheets_trilinearly():
    """Fanning normals separate consecutive offset sheets laterally.

    The top row and left column of the 4x4 middle slice fall strictly
    between the step-0 and step-+1 sheets: corner lines and the bilinear
    sheet fills miss them, and only _draw_trilinear's slab filling between
    consecutive offset sheets marks those seven voxels.
    """
    output = project_binary_mask_along_normals(
        np.ones((2, 2), dtype=np.uint8),
        np.array(
            [[[2.5, 1.2, 1.2], [2.5, 1.2, 3.2]], [[2.5, 3.2, 1.2], [2.5, 3.2, 3.2]]],
            dtype=np.float32,
        ),
        np.array(
            [[[1.0, -0.8, -0.8], [1.0, -0.8, 0.8]], [[1.0, 0.8, -0.8], [1.0, 0.8, 0.8]]],
            dtype=np.float32,
        ),
        np.ones((2, 2), dtype=bool),
        (0, 0, 0, 5, 5, 5),
        half_thickness_voxels=1.0,
    )
    expected = np.zeros((5, 5, 5), dtype=output.dtype)
    expected[1, 1:3, 1:3] = 1
    expected[2:4, 0:4, 0:4] = 1
    np.testing.assert_array_equal(output, expected)


def test_normal_projection_skips_nan_normals_and_invalid_cells():
    normals = np.zeros((2, 2, 3), dtype=np.float32)
    normals[..., 0] = 1.0
    normals[0, 1] = np.nan
    output = project_binary_mask_along_normals(
        np.ones((2, 2), dtype=np.uint8),
        np.array(
            [
                [[1.4, 1.3, 1.2], [1.4, 1.3, 6.7]],
                [[1.4, 2.6, 1.2], [1.4, 2.6, 6.7]],
            ],
            dtype=np.float32,
        ),
        normals,
        np.array([[True, True], [True, False]]),
        (0, 0, 0, 4, 4, 9),
        half_thickness_voxels=1.0,
    )
    assert _marked_voxels(output) == [
        (0, 1, 1),
        (0, 2, 1),
        (1, 1, 1),
        (1, 2, 1),
        (2, 1, 1),
        (2, 2, 1),
    ]


def _augmentation_nodes(transform):
    yield transform
    for attribute in ("transforms", "list_of_transforms"):
        children = getattr(transform, attribute, None)
        if children is not None:
            for child in children:
                yield from _augmentation_nodes(child)
    inner = getattr(transform, "transform", None)
    if inner is not None:
        yield from _augmentation_nodes(inner)


def test_default_augmentation_graph_and_seeded_values():
    transforms = build_augmentations(
        "default", (17, 128, 128), rotation_axes=(0,)
    )
    nodes = list(_augmentation_nodes(transforms))
    names = [type(node).__name__ for node in nodes]
    assert "BlankRectangleTransform" in names
    assert "LocalGammaTransform" not in names
    blur = next(node for node in nodes if type(node).__name__ == "GaussianBlurTransform")
    assert blur.benchmark is False
    restricted = build_augmentations(
        "spatial_intensity_no_clip", (17, 128, 128), rotation_axes=(0,)
    )
    restricted_names = [type(node).__name__ for node in _augmentation_nodes(restricted)]
    assert "GaussianNoiseTransform" in restricted_names
    assert "ContrastTransform" not in restricted_names

    def run_seeded():
        image = torch.linspace(-1.0, 1.0, 192, dtype=torch.float32).reshape(1, 3, 8, 8)
        inklabels = (torch.arange(192).reshape(1, 3, 8, 8) % 3 == 0).float()
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        return transforms(
            image=image,
            inklabels=inklabels,
            supervision_mask=torch.ones_like(inklabels),
        )

    augmented = run_seeded()
    repeated = run_seeded()
    assert torch.equal(repeated["image"], augmented["image"])
    assert torch.equal(repeated["inklabels"], augmented["inklabels"])

    image = augmented["image"]
    assert image.shape == (1, 3, 8, 8)
    assert image.dtype == torch.float32
    assert image.mean().item() == pytest.approx(-0.0236479, abs=1e-4)
    assert image.std().item() == pytest.approx(0.4906498, abs=1e-4)
    assert image.min().item() == pytest.approx(-1.0584830, abs=1e-4)
    assert image.max().item() == pytest.approx(0.6304119, abs=1e-4)
    torch.testing.assert_close(
        image[0, 1, 4, :4],
        torch.tensor([-0.274156, -0.186243, -0.099831, 0.084368]),
        rtol=0.0,
        atol=1e-4,
    )
    # The seeded label path resolves to a pure spatial rearrangement; on this
    # symmetric lattice it equals a width flip of the input.
    expected_labels = torch.flip(
        (torch.arange(192).reshape(1, 3, 8, 8) % 3 == 0).float(), dims=(3,)
    )
    assert torch.equal(augmented["inklabels"], expected_labels)


def test_native_crop_translation_preserves_supervised_fraction():
    class StubRng:
        def __init__(self):
            self.random_values = iter((0.0, 0.0, 0.0, 0.0))

        def random(self):
            return next(self.random_values)

        @staticmethod
        def randint(low, high):
            assert (low, high) == (10, 41)
            return 10

        @staticmethod
        def shuffle(values):
            values[:] = np.array((1, 2), dtype=values.dtype)

    positions = np.array(
        [[[110, 208, 305], [111, 209, 315], [110, 208, 325]]],
        dtype=np.float32,
    )
    translated = maybe_translate_crop_bbox(
        (100, 200, 300, 120, 220, 340),
        positions,
        np.ones((1, 3), dtype=bool),
        np.ones((1, 3), dtype=np.uint8),
        rng=StubRng(),
    )
    assert translated == (100, 190, 290, 120, 210, 330)


def test_confusion_and_balanced_accuracy_match_binary_counts():
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    valid = torch.tensor([[[[True, True], [True, False]]]])
    batch = MetricBatch(logits=logits, targets=targets, valid_mask=valid)
    counts = Confusion().compute_batch(batch)
    count_values = tuple(
        value.item() for value in (counts.tp, counts.fp, counts.fn, counts.tn)
    )
    assert count_values == (1, 1, 1, 0)
    assert BalancedAccuracy.from_counts(counts).item() == pytest.approx(0.25)


def _sampling_config(tmp_path: Path, strategy: str) -> InkDataConfig:
    return InkDataConfig.from_mapping(
        {
            "mode": "flat",
            "patch_size": [1, 2, 2],
            "patch_overlap": 0.25,
            "patch_min_labeled_coverage": 0.0,
            "datasets": [
                {
                    "segments_path": str(tmp_path / "a"),
                    "volume_scale": 0,
                    "sampling_scroll": "A",
                    "sampling_physical_segment_keys": {"a1": "A:1", "a2": "A:2"},
                    "sampling_representation_keys": {"a1": "ra1", "a2": "ra2"},
                },
                {
                    "segments_path": str(tmp_path / "b"),
                    "volume_scale": 0,
                    "sampling_scroll": "B",
                    "sampling_physical_segment_keys": {"b1": "B:1"},
                    "sampling_representation_keys": {"b1": "rb1"},
                },
            ],
            "seed": 42,
            "sampling_strategy": strategy,
            "fixed_scroll_prior": {
                "seed": 42,
                "target_batch_counts": {"A": 2, "B": 2},
            },
        }
    )


def _sampling_patches(config: InkDataConfig, tmp_path: Path) -> list[Patch]:
    patches = []
    for dataset_idx, relpath, count in ((0, "a1", 2), (0, "a2", 2), (1, "b1", 4)):
        source = config.datasets[dataset_idx]
        segment = Segment(
            data_config=config,
            source=source,
            dataset_idx=dataset_idx,
            segment_relpath=relpath,
            segment_dir=tmp_path / relpath,
            segment_name=relpath,
            image_volume="unused",
        )
        patches.extend(
            Patch(segment=segment, bbox=(0, 0, 0, 1, 2, 2)) for _ in range(count)
        )
    return patches


def test_all_three_sampling_policies(tmp_path):
    uniform_config = _sampling_config(tmp_path, "uniform")
    patches = _sampling_patches(uniform_config, tmp_path)
    uniform = build_sampling_policy(patches, uniform_config, batch_size=4)
    assert uniform.audit["strategy"] == uniform_config.sampling.strategy
    assert uniform.generator.initial_seed() == 42

    balanced_config = _sampling_config(tmp_path, "scroll_segment_balanced")
    balanced_patches = _sampling_patches(balanced_config, tmp_path)
    weights, audit = hierarchical_scroll_segment_weights(
        balanced_patches, balanced_config
    )
    torch.testing.assert_close(weights, torch.full((8,), 0.125, dtype=torch.double))
    assert audit["segments_per_scroll"] == {"A": 2, "B": 1}
    balanced = build_sampling_policy(
        balanced_patches, balanced_config, batch_size=4
    )
    assert balanced.sampler.replacement is True
    assert balanced.sampler.num_samples == len(balanced_patches)

    fixed_config = _sampling_config(tmp_path, "fixed_scroll_prior_stratified")
    fixed_patches = _sampling_patches(fixed_config, tmp_path)
    fixed = FixedScrollPriorStratifiedBatchSampler(
        fixed_patches, fixed_config, batch_size=4
    )
    assert fixed.definition_audit() == {
        "strategy": "fixed_scroll_prior_stratified",
        "seed": 42,
        "batch_size": 4,
        "batches_per_loader_epoch": 2,
        "target_per_batch": {"A": 2, "B": 2},
        "target_fraction": {"A": 0.5, "B": 0.5},
        "source_patches": 8,
        "source_patches_by_scroll": {"A": 4, "B": 4},
        "source_patches_by_physical_segment": {"A:1": 2, "A:2": 2, "B:1": 4},
        "source_patches_by_representation": {"ra1": 2, "ra2": 2, "rb1": 4},
        "physical_segments_by_scroll": {"A": ["A:1", "A:2"], "B": ["B:1"]},
        "representations_by_physical_segment": {
            "A:1": ["ra1"],
            "A:2": ["ra2"],
            "B:1": ["rb1"],
        },
    }
    batches = list(fixed)
    assert batches == [[6, 4, 2, 1], [3, 0, 7, 5]]
    for batch in batches:
        scrolls = Counter(
            fixed_patches[index].segment.source.sampling_scroll for index in batch
        )
        assert scrolls == {"A": 2, "B": 2}
    assert fixed.observed_audit() == {
        "strategy": "fixed_scroll_prior_stratified",
        "seed": 42,
        "batches_yielded_to_dataloader": 2,
        "samples_yielded_to_dataloader": 8,
        "observed_by_scroll": {"A": 4, "B": 4},
        "observed_by_physical_segment": {"A:1": 2, "A:2": 2, "B:1": 4},
        "observed_by_representation": {"ra1": 2, "ra2": 2, "rb1": 4},
        "patch_queue_recycles": {"ra1": 0, "ra2": 0, "rb1": 0},
    }
    second = FixedScrollPriorStratifiedBatchSampler(
        fixed_patches, fixed_config, batch_size=4
    )
    assert batches == list(second)
