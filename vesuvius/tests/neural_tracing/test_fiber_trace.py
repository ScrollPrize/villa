from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace.dataset import FiberTraceBatchBuilder
from vesuvius.neural_tracing.fiber_trace.fiber_json import parse_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace.geometry import (
    classify_voxels,
    construct_up_vector,
    tangent_at_point,
)
from vesuvius.neural_tracing.fiber_trace.labels import (
    IGNORE_INDEX,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
)
from vesuvius.neural_tracing.fiber_trace.losses import (
    compute_fiber_trace_loss,
    sign_ambiguous_up_loss,
    supervised_contrastive_loss,
)
from vesuvius.neural_tracing.fiber_trace.model import (
    DirectionConditionedFiberTraceModel,
)


def _write_fiber(path: Path) -> None:
    payload = {
        "type": "vc3d_fiber",
        "version": 1,
        "line_points": [[float(x), 8.0, 8.0] for x in range(2, 14)],
        "control_points": [[8.0, 8.0, 8.0], [10.0, 8.0, 8.0]],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _synthetic_config(
    tmp_path: Path, *, batch_size: int = 4, crop_size=(16, 16, 16)
) -> dict:
    volume = np.arange(16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16)
    mask = np.ones((16, 16, 16), dtype=np.uint8)
    fiber_path = tmp_path / "fiber.json"
    _write_fiber(fiber_path)
    return {
        "crop_size": list(crop_size),
        "batch_size": batch_size,
        "seed": 123,
        "image_normalization": "unit",
        "positive_direction_probability": 1.0,
        "positive_direction_jitter_degrees": 0.0,
        "positive_radius": 1.25,
        "ignore_radius": 2.5,
        "_array_records": [
            {
                "volume": volume,
                "mask": mask,
                "fiber_path": str(fiber_path),
            }
        ],
    }


def test_parse_vc3d_fiber_validates_and_exposes_zyx_order():
    fiber = parse_vc3d_fiber(
        {
            "type": "vc3d_fiber",
            "version": 1,
            "line_points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "control_points": [[1.0, 2.0, 3.0]],
            "generation": 7,
        }
    )
    assert fiber.version == 1
    assert fiber.generation == 7
    np.testing.assert_array_equal(
        fiber.line_points_zyx[0], np.array([3.0, 2.0, 1.0], dtype=np.float32)
    )

    with pytest.raises(ValueError, match="line_points"):
        parse_vc3d_fiber(
            {
                "type": "vc3d_fiber",
                "version": 1,
                "line_points": [[1, 2, 3]],
                "control_points": [[1, 2, 3]],
            }
        )
    with pytest.raises(ValueError, match="version"):
        parse_vc3d_fiber(
            {
                "type": "vc3d_fiber",
                "version": 2,
                "line_points": [[1, 2, 3], [2, 3, 4]],
                "control_points": [[1, 2, 3]],
            }
        )


def test_tangent_uses_line_points_and_control_point_query():
    fiber = parse_vc3d_fiber(
        {
            "type": "vc3d_fiber",
            "version": 1,
            "line_points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "control_points": [[1.0, 0.0, 0.0]],
        }
    )
    tangent = tangent_at_point(fiber.line_points_zyx, fiber.control_points_zyx[0])
    np.testing.assert_allclose(tangent, np.array([0.0, 0.0, 1.0], dtype=np.float32))


def test_up_vector_construction_and_sign_ambiguous_loss():
    up = construct_up_vector(
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        up, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6
    )

    target = torch.tensor([[[[[0.0]]], [[[1.0]]], [[[0.0]]]]])
    pred = -target
    mask = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    assert float(sign_ambiguous_up_loss(pred, target, mask)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_batch_builder_samples_one_fiber_with_half_gt_and_half_random_crops(
    tmp_path: Path,
):
    builder = FiberTraceBatchBuilder(
        _synthetic_config(tmp_path), rng=np.random.default_rng(5)
    )
    batch = builder.sample_batch(record_index=0)

    assert batch.volume.shape == (4, 1, 16, 16, 16)
    assert batch.labels.shape == (4, 16, 16, 16)
    assert batch.crop_kinds.count("gt_control") == 2
    assert batch.crop_kinds.count("random_valid") == 2
    assert len(set(batch.fiber_paths)) == 1
    assert bool((batch.labels == POSITIVE_LABEL).any())
    assert bool((batch.labels == NEGATIVE_LABEL).any())


def test_missing_mask_fails_loudly(tmp_path: Path):
    fiber_path = tmp_path / "fiber.json"
    _write_fiber(fiber_path)
    config = {
        "crop_size": [8, 8, 8],
        "batch_size": 2,
        "datasets": [
            {
                "volume_path": str(tmp_path / "missing.zarr"),
                "fiber_paths": [str(fiber_path)],
            }
        ],
    }
    with pytest.raises(ValueError, match="mask_path or grad_mag_path"):
        FiberTraceBatchBuilder(config)


def test_voxel_classification_positive_negative_and_ignore():
    line = np.array([[4.0, 4.0, 2.0], [4.0, 4.0, 6.0]], dtype=np.float32)
    mask = np.ones((9, 9, 9), dtype=bool)
    mask[1, 1, 1] = False
    result = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_zyx=line,
        cond_fw_zyx=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        valid_mask=mask,
        positive_radius=1.0,
        ignore_radius=2.0,
    )
    assert int(result["labels"][4, 4, 4]) == POSITIVE_LABEL
    assert int(result["labels"][0, 0, 0]) == NEGATIVE_LABEL
    assert int(result["labels"][1, 1, 1]) == IGNORE_INDEX

    wrong_dir = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_zyx=line,
        cond_fw_zyx=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        positive_radius=1.0,
        ignore_radius=2.0,
    )
    assert int(wrong_dir["labels"][4, 4, 4]) == NEGATIVE_LABEL


def test_supervised_contrastive_loss_is_finite_on_synthetic_embeddings():
    embeddings = torch.zeros((1, 2, 1, 1, 4), dtype=torch.float32)
    embeddings[:, :, 0, 0, 0] = torch.tensor([1.0, 0.0])
    embeddings[:, :, 0, 0, 1] = torch.tensor([1.0, 0.0])
    embeddings[:, :, 0, 0, 2] = torch.tensor([0.0, 1.0])
    embeddings[:, :, 0, 0, 3] = torch.tensor([0.0, 1.0])
    labels = torch.tensor(
        [[[[POSITIVE_LABEL, POSITIVE_LABEL, NEGATIVE_LABEL, NEGATIVE_LABEL]]]]
    )
    loss = supervised_contrastive_loss(embeddings, labels, temperature=0.1)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_model_forward_loss_and_backward_smoke(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=2, crop_size=(8, 8, 8))
    builder = FiberTraceBatchBuilder(config, rng=np.random.default_rng(9))
    batch = builder.sample_batch(record_index=0)
    model = DirectionConditionedFiberTraceModel(
        backbone_channels=2,
        embedding_dim=6,
        features_per_stage=(2,),
        head_channels=4,
    )
    outputs = model(batch.volume, batch.cond_fw, batch.cond_up)
    assert outputs["embedding"].shape == (2, 6, 8, 8, 8)
    assert outputs["fw"].shape == (2, 3, 8, 8, 8)

    losses = compute_fiber_trace_loss(outputs, batch, max_contrastive_samples=512)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert any(param.grad is not None for param in model.parameters())
