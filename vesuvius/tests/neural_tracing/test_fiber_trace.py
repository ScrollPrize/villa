from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import vesuvius.neural_tracing.fiber_trace.dataset as fiber_dataset
import vesuvius.neural_tracing.fiber_trace.train as fiber_train
from vesuvius.neural_tracing.fiber_trace.dataset import FiberTraceBatchBuilder
from vesuvius.neural_tracing.fiber_trace.fiber_json import parse_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace.geometry import (
    classify_voxels,
    construct_up_vector,
    decode_lasagna_normals_xyz,
    folded_frame_error_degrees,
    tangent_at_point,
)
from vesuvius.neural_tracing.fiber_trace.labels import (
    IGNORE_ID,
    IGNORE_INDEX,
    NEGATIVE_LABEL,
    NEGATIVE_ONLY_ID,
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
    nx = np.full((16, 16, 16), 128, dtype=np.uint8)
    ny = np.full((16, 16, 16), 128, dtype=np.uint8)
    fiber_path = tmp_path / "fiber.json"
    _write_fiber(fiber_path)
    return {
        "crop_size": list(crop_size),
        "batch_size": batch_size,
        "seed": 123,
        "image_normalization": "unit",
        "positive_direction_probability": 1.0,
        "positive_direction_jitter_degrees": 30.0,
        "negative_direction_min_degrees": 60.0,
        "negative_direction_max_degrees": 90.0,
        "normal_plane_jitter_voxels": 40.0,
        "normal_perpendicular_jitter_voxels": 10.0,
        "negative_cone_distance_voxels": 30.0,
        "positive_radius": 1.25,
        "ignore_radius": 2.5,
        "positive_cosine": float(np.cos(np.deg2rad(30.0))),
        "negative_cosine": float(np.cos(np.deg2rad(60.0))),
        "_array_records": [
            {
                "volume": volume,
                "mask": mask,
                "nx": nx,
                "ny": ny,
                "fiber_path": str(fiber_path),
            }
        ],
    }


def test_parse_vc3d_fiber_validates_and_preserves_xyz_order():
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
        fiber.line_points_xyz[0], np.array([1.0, 2.0, 3.0], dtype=np.float32)
    )
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
            "line_points": [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [4.0, 2.0, 0.0]],
            "control_points": [[2.0, 1.0, 0.0]],
        }
    )
    tangent = tangent_at_point(fiber.line_points_xyz, fiber.control_points_xyz[0])
    np.testing.assert_allclose(
        tangent,
        np.array([2.0, 1.0, 0.0], dtype=np.float32) / np.sqrt(5.0),
        atol=1e-6,
    )


def test_lasagna_normal_decoding_and_up_projection():
    normals, valid = decode_lasagna_normals_xyz(
        np.array([128, 128, 255, 0], dtype=np.uint8),
        np.array([128, 255, 128, 0], dtype=np.uint8),
    )
    np.testing.assert_allclose(normals[0], np.array([0.0, 0.0, 1.0]), atol=1e-6)
    np.testing.assert_allclose(normals[1], np.array([0.0, 1.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(normals[2], np.array([1.0, 0.0, 0.0]), atol=1e-6)
    assert bool(valid[:3].all())
    assert not bool(valid[3])

    up = construct_up_vector(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        up, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6
    )
    with pytest.raises(ValueError, match="normal_xyz is required"):
        construct_up_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="degenerate"):
        construct_up_vector(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )


def test_sign_ambiguous_loss():

    target = torch.tensor([[[[[0.0]]], [[[1.0]]], [[[0.0]]]]])
    pred = -target
    mask = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    assert float(sign_ambiguous_up_loss(pred, target, mask)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_folded_frame_error_boundaries():
    target_fw = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    target_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for raw_degrees, expected_degrees in [
        (0.0, 0.0),
        (30.0, 30.0),
        (60.0, 60.0),
        (90.0, 90.0),
        (120.0, 60.0),
        (150.0, 30.0),
        (180.0, 0.0),
    ]:
        radians = np.deg2rad(raw_degrees)
        cond_fw = np.array([np.cos(radians), np.sin(radians), 0.0], dtype=np.float32)
        error = folded_frame_error_degrees(cond_fw, target_up, target_fw, target_up)
        assert float(error) == pytest.approx(expected_degrees, abs=1e-4)

    coupled_error = folded_frame_error_degrees(
        -target_fw,
        -target_up,
        target_fw,
        target_up,
    )
    assert float(coupled_error) == pytest.approx(0.0, abs=1e-4)


def test_batch_builder_samples_one_fiber_with_paired_conditioning_variants(
    tmp_path: Path,
):
    builder = FiberTraceBatchBuilder(
        _synthetic_config(tmp_path), rng=np.random.default_rng(5)
    )
    batch = builder.sample_batch(record_index=0)

    assert batch.volume.shape == (4, 1, 16, 16, 16)
    assert batch.labels.shape == (4, 16, 16, 16)
    assert batch.target_id.shape == (4, 16, 16, 16)
    assert batch.crop_kinds.count("gt_control") == 2
    assert batch.crop_kinds.count("random_valid") == 2
    assert batch.direction_kinds == ("positive", "negative", "positive", "negative")
    assert torch.equal(batch.crop_origin_zyx[0], batch.crop_origin_zyx[1])
    assert torch.equal(batch.crop_origin_zyx[2], batch.crop_origin_zyx[3])
    assert len(set(batch.fiber_paths)) == 1
    assert bool((batch.labels == POSITIVE_LABEL).any())
    assert bool((batch.labels == NEGATIVE_LABEL).any())
    assert bool((batch.target_id[batch.labels == POSITIVE_LABEL] == 0).all())
    assert bool(
        (batch.target_id[batch.labels == NEGATIVE_LABEL] == NEGATIVE_ONLY_ID).all()
    )
    assert bool((batch.target_up_valid & (batch.labels == POSITIVE_LABEL)).any())
    target_fw = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    target_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    folded_errors = folded_frame_error_degrees(
        batch.cond_fw_xyz.numpy(),
        batch.cond_up_xyz.numpy(),
        target_fw,
        target_up,
    )
    positive_errors = folded_errors[np.array(batch.direction_kinds) == "positive"]
    negative_errors = folded_errors[np.array(batch.direction_kinds) == "negative"]
    assert bool((positive_errors >= -1e-4).all())
    assert bool((positive_errors <= 30.0 + 1e-4).all())
    assert bool((negative_errors >= 60.0 - 1e-4).all())
    assert bool((negative_errors <= 90.0 + 1e-4).all())


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


def test_valid_mask_threshold_key_is_removed(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=2, crop_size=(8, 8, 8))
    config["_array_records"][0]["valid_mask_threshold"] = 0.0
    with pytest.raises(ValueError, match="valid_mask_threshold was removed"):
        FiberTraceBatchBuilder(config)


def test_negative_direction_max_is_folded_frame_degree_bound(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=2, crop_size=(8, 8, 8))
    config["negative_direction_max_degrees"] = 120.0
    with pytest.raises(ValueError, match="negative_direction_max_degrees"):
        FiberTraceBatchBuilder(config)


def test_missing_normals_fail_loudly(tmp_path: Path):
    fiber_path = tmp_path / "fiber.json"
    _write_fiber(fiber_path)
    config = {
        "crop_size": [8, 8, 8],
        "batch_size": 2,
        "datasets": [
            {
                "volume_path": str(tmp_path / "volume.zarr"),
                "mask_path": str(tmp_path / "mask.zarr"),
                "fiber_paths": [str(fiber_path)],
            }
        ],
    }
    with pytest.raises(ValueError, match="nx_path and ny_path"):
        FiberTraceBatchBuilder(config)


def test_array_record_normal_shape_mismatch_fails(tmp_path: Path):
    fiber_path = tmp_path / "fiber.json"
    _write_fiber(fiber_path)
    config = _synthetic_config(tmp_path, batch_size=2, crop_size=(8, 8, 8))
    config["_array_records"][0]["ny"] = np.full((15, 16, 16), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="normal channel shapes"):
        FiberTraceBatchBuilder(config)


def test_remote_zarr_requires_cache_dir_before_network(tmp_path: Path):
    fiber_path = tmp_path / "fiber.json"
    _write_fiber(fiber_path)
    config = {
        "crop_size": [8, 8, 8],
        "batch_size": 2,
        "datasets": [
            {
                "volume_path": "https://example.com/volume.zarr",
                "mask_path": "https://example.com/mask.zarr",
                "nx_path": "https://example.com/nx.zarr",
                "ny_path": "https://example.com/ny.zarr",
                "fiber_paths": [str(fiber_path)],
            }
        ],
    }
    with pytest.raises(ValueError, match="volume_cache_dir"):
        FiberTraceBatchBuilder(config)


def test_zarr_access_uses_common_helpers_without_direct_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fiber_path = tmp_path / "fiber.json"
    _write_fiber(fiber_path)
    volume = np.zeros((16, 16, 16), dtype=np.uint8)
    mask = np.ones((16, 16, 16), dtype=np.uint8)
    nx = np.full((16, 16, 16), 128, dtype=np.uint8)
    ny = np.full((16, 16, 16), 128, dtype=np.uint8)
    calls: list[tuple[str, str, int | None]] = []

    def fake_open_group(path, *, auth_json_path=None, config=None):
        calls.append(("group", str(path), None))
        return {"0": volume}

    def fake_open_zarr(path, *, scale=None, auth_json_path=None, config=None):
        calls.append(("array", str(path), int(scale)))
        return {
            str((Path.cwd() / "volume").resolve()): volume,
            str((Path.cwd() / "mask").resolve()): mask,
            str((Path.cwd() / "nx").resolve()): nx,
            str((Path.cwd() / "ny").resolve()): ny,
        }[str(path)]

    monkeypatch.setattr(fiber_dataset, "_common_open_zarr_group", fake_open_group)
    monkeypatch.setattr(fiber_dataset, "_common_open_zarr", fake_open_zarr)

    FiberTraceBatchBuilder(
        {
            "crop_size": [8, 8, 8],
            "batch_size": 2,
            "datasets": [
                {
                    "volume_path": "volume",
                    "mask_path": "mask",
                    "nx_path": "nx",
                    "ny_path": "ny",
                    "fiber_paths": [str(fiber_path)],
                }
            ],
        }
    )
    assert calls == [
        ("array", str((Path.cwd() / "volume").resolve()), 0),
        ("array", str((Path.cwd() / "mask").resolve()), 0),
        ("array", str((Path.cwd() / "nx").resolve()), 0),
        ("array", str((Path.cwd() / "ny").resolve()), 0),
    ]


def test_lasagna_manifest_drives_channels_and_scaled_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fiber_path = tmp_path / "fiber.json"
    _write_fiber(fiber_path)
    (tmp_path / "umbilicus.json").write_text(
        json.dumps({"control_points": [{"x": 8, "y": 8, "z": 8}]}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "pred.lasagna.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "source_to_base": 1.0,
                "base_shape_zyx": [16, 16, 16],
                "umbilicus_json": "umbilicus.json",
                "groups": {
                    "grad_mag": {
                        "zarr": "grad_mag.ome.zarr/2",
                        "scaledown": 2,
                        "channels": ["grad_mag"],
                    },
                    "nx": {
                        "zarr": "normal.ome.zarr/2",
                        "scaledown": 2,
                        "channels": ["nx", "ny"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    base_root = tmp_path / "base.ome.zarr"
    grad_root = tmp_path / "grad_mag.ome.zarr"
    normal_root = tmp_path / "normal.ome.zarr"
    base0 = np.zeros((16, 16, 16), dtype=np.uint8)
    base1 = np.zeros((8, 8, 8), dtype=np.uint8)
    grad = np.ones((4, 4, 4), dtype=np.uint8)
    normals = np.stack(
        [
            np.full((4, 4, 4), 128, dtype=np.uint8),
            np.full((4, 4, 4), 128, dtype=np.uint8),
        ],
        axis=0,
    )
    calls: list[tuple[str, int]] = []

    def fake_open_zarr(path, *, scale=None, auth_json_path=None, config=None):
        calls.append((str(path), int(scale)))
        if str(path) == str(base_root.resolve()) and int(scale) == 0:
            return base0
        if str(path) == str(base_root.resolve()) and int(scale) == 1:
            return base1
        if str(path) == str(grad_root.resolve()) and int(scale) == 2:
            return grad
        if str(path) == str(normal_root.resolve()) and int(scale) == 2:
            return normals
        raise AssertionError((path, scale))

    monkeypatch.setattr(fiber_dataset, "_common_open_zarr", fake_open_zarr)
    builder = FiberTraceBatchBuilder(
        {
            "crop_size": [4, 4, 4],
            "batch_size": 2,
            "seed": 1,
            "image_normalization": "unit",
            "positive_direction_probability": 1.0,
            "positive_direction_jitter_degrees": 0.0,
            "datasets": [
                {
                    "base_volume_path": str(base_root),
                    "base_volume_scale": 1,
                    "lasagna_manifest_path": str(manifest_path),
                    "fiber_paths": [str(fiber_path)],
                }
            ],
        },
        rng=np.random.default_rng(3),
    )
    batch = builder.sample_batch(record_index=0)

    assert batch.volume.shape == (2, 1, 4, 4, 4)
    assert bool((batch.labels == POSITIVE_LABEL).any())
    assert bool((batch.target_up_valid & (batch.labels == POSITIVE_LABEL)).any())
    assert calls[:5] == [
        (str(base_root.resolve()), 1),
        (str(base_root.resolve()), 0),
        (str(grad_root.resolve()), 2),
        (str(normal_root.resolve()), 2),
        (str(normal_root.resolve()), 2),
    ]


def test_voxel_classification_uses_normal_plane_positive_zone():
    line = np.array([[2.0, 3.0, 5.0], [6.0, 3.0, 5.0]], dtype=np.float32)
    mask = np.ones((9, 9, 9), dtype=bool)
    mask[1, 1, 1] = False
    normal_xyz = np.zeros((9, 9, 9, 3), dtype=np.float32)
    normal_xyz[..., 2] = 1.0
    result = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        cond_up_xyz=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        valid_mask=mask,
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=1.0,
        normal_perpendicular_jitter_voxels=1.0,
        negative_cone_distance_voxels=3.0,
        positive_cosine=float(np.cos(np.deg2rad(30.0))),
        negative_cosine=float(np.cos(np.deg2rad(60.0))),
        positive_target_id=7,
    )
    assert int(result["labels"][5, 3, 4]) == POSITIVE_LABEL
    assert int(result["target_id"][5, 3, 4]) == 7
    assert int(result["labels"][5, 5, 4]) == IGNORE_INDEX
    assert int(result["labels"][7, 5, 4]) == IGNORE_INDEX
    assert int(result["labels"][1, 1, 1]) == IGNORE_INDEX
    assert int(result["target_id"][1, 1, 1]) == IGNORE_ID
    np.testing.assert_allclose(
        result["target_fw_xyz"][:, 5, 3, 4],
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["target_up_xyz"][:, 5, 3, 4],
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )
    assert bool(result["target_up_valid"][5, 3, 4])


def test_voxel_classification_cone_negatives_and_positive_precedence():
    line = np.array([[2.0, 3.0, 5.0], [6.0, 3.0, 5.0]], dtype=np.float32)
    normal_xyz = np.zeros((9, 9, 9, 3), dtype=np.float32)
    normal_xyz[..., 2] = 1.0
    result = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        cond_up_xyz=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=1.0,
        normal_perpendicular_jitter_voxels=1.0,
        negative_cone_distance_voxels=3.0,
        positive_target_id=11,
    )
    assert int(result["labels"][8, 5, 4]) == NEGATIVE_LABEL
    assert int(result["target_id"][8, 5, 4]) == NEGATIVE_ONLY_ID
    assert int(result["labels"][7, 5, 4]) == IGNORE_INDEX
    assert int(result["labels"][8, 3, 8]) == IGNORE_INDEX

    overlap = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        cond_up_xyz=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=3.0,
        normal_perpendicular_jitter_voxels=3.0,
        negative_cone_distance_voxels=2.0,
        positive_target_id=12,
    )
    assert int(overlap["labels"][7, 5, 4]) == POSITIVE_LABEL
    assert int(overlap["target_id"][7, 5, 4]) == 12


def test_voxel_classification_direction_pairs_label_near_fiber_zone():
    line = np.array([[2.0, 3.0, 5.0], [6.0, 3.0, 5.0]], dtype=np.float32)
    normal_xyz = np.zeros((9, 9, 9, 3), dtype=np.float32)
    normal_xyz[..., 2] = 1.0
    positive = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        cond_up_xyz=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=1.0,
        normal_perpendicular_jitter_voxels=1.0,
        negative_cone_distance_voxels=3.0,
        positive_cosine=float(np.cos(np.deg2rad(30.0))),
        negative_cosine=float(np.cos(np.deg2rad(60.0))),
    )
    assert int(positive["labels"][5, 3, 4]) == POSITIVE_LABEL

    negative = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array([0.5, np.sqrt(3.0) / 2.0, 0.0], dtype=np.float32),
        cond_up_xyz=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=1.0,
        normal_perpendicular_jitter_voxels=1.0,
        negative_cone_distance_voxels=3.0,
        positive_cosine=float(np.cos(np.deg2rad(30.0))),
        negative_cosine=float(np.cos(np.deg2rad(60.0))),
    )
    assert int(negative["labels"][5, 3, 4]) == NEGATIVE_LABEL

    transition = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array(
            [np.cos(np.deg2rad(50.0)), np.sin(np.deg2rad(50.0)), 0.0],
            dtype=np.float32,
        ),
        cond_up_xyz=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=1.0,
        normal_perpendicular_jitter_voxels=1.0,
        negative_cone_distance_voxels=3.0,
        positive_cosine=float(np.cos(np.deg2rad(30.0))),
        negative_cosine=float(np.cos(np.deg2rad(60.0))),
    )
    assert int(transition["labels"][5, 3, 4]) == IGNORE_INDEX

    positive_equivalent = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array(
            [np.cos(np.deg2rad(150.0)), np.sin(np.deg2rad(150.0)), 0.0],
            dtype=np.float32,
        ),
        cond_up_xyz=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=1.0,
        normal_perpendicular_jitter_voxels=1.0,
        negative_cone_distance_voxels=3.0,
        positive_cosine=float(np.cos(np.deg2rad(30.0))),
        negative_cosine=float(np.cos(np.deg2rad(60.0))),
    )
    assert int(positive_equivalent["labels"][5, 3, 4]) == POSITIVE_LABEL

    negative_equivalent = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array(
            [np.cos(np.deg2rad(120.0)), np.sin(np.deg2rad(120.0)), 0.0],
            dtype=np.float32,
        ),
        cond_up_xyz=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=1.0,
        normal_perpendicular_jitter_voxels=1.0,
        negative_cone_distance_voxels=3.0,
        positive_cosine=float(np.cos(np.deg2rad(30.0))),
        negative_cosine=float(np.cos(np.deg2rad(60.0))),
    )
    assert int(negative_equivalent["labels"][5, 3, 4]) == NEGATIVE_LABEL


def test_degenerate_up_vectors_are_invalid_or_raise():
    line = np.array([[2.0, 3.0, 5.0], [6.0, 3.0, 5.0]], dtype=np.float32)
    normal_xyz = np.zeros((9, 9, 9, 3), dtype=np.float32)
    normal_xyz[..., 0] = 1.0
    result = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        cond_up_xyz=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        positive_radius=0.25,
        ignore_radius=2.0,
    )
    assert int(result["labels"][5, 3, 4]) == POSITIVE_LABEL
    assert not bool(result["target_up_valid"][5, 3, 4])

    with pytest.raises(ValueError, match="degenerate"):
        classify_voxels(
            crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
            crop_shape=(9, 9, 9),
            line_points_xyz=line,
            cond_fw_xyz=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            cond_up_xyz=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            valid_mask=np.ones((9, 9, 9), dtype=bool),
            normal_xyz=normal_xyz,
            normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
            degenerate_up_policy="raise",
            positive_radius=0.25,
            ignore_radius=2.0,
        )


def test_supervised_contrastive_loss_is_finite_on_synthetic_embeddings():
    embeddings = torch.zeros((1, 2, 1, 1, 4), dtype=torch.float32)
    embeddings[:, :, 0, 0, 0] = torch.tensor([1.0, 0.0])
    embeddings[:, :, 0, 0, 1] = torch.tensor([1.0, 0.0])
    embeddings[:, :, 0, 0, 2] = torch.tensor([0.0, 1.0])
    embeddings[:, :, 0, 0, 3] = torch.tensor([0.0, 1.0])
    labels = torch.tensor(
        [[[[POSITIVE_LABEL, POSITIVE_LABEL, NEGATIVE_LABEL, NEGATIVE_LABEL]]]]
    )
    target_id = torch.tensor([[[[3, 3, NEGATIVE_ONLY_ID, NEGATIVE_ONLY_ID]]]])
    loss = supervised_contrastive_loss(embeddings, labels, target_id, temperature=0.1)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_supervised_contrastive_negatives_are_denominator_only():
    labels = torch.tensor(
        [[[[POSITIVE_LABEL, POSITIVE_LABEL, NEGATIVE_LABEL, NEGATIVE_LABEL]]]]
    )
    target_id = torch.tensor([[[[5, 5, NEGATIVE_ONLY_ID, NEGATIVE_ONLY_ID]]]])

    embeddings_same_neg = torch.zeros((1, 2, 1, 1, 4), dtype=torch.float32)
    embeddings_same_neg[:, :, 0, 0, 0] = torch.tensor([1.0, 0.0])
    embeddings_same_neg[:, :, 0, 0, 1] = torch.tensor([1.0, 0.0])
    embeddings_same_neg[:, :, 0, 0, 2] = torch.tensor([0.0, 1.0])
    embeddings_same_neg[:, :, 0, 0, 3] = torch.tensor([0.0, 1.0])

    embeddings_opposed_neg = embeddings_same_neg.clone()
    embeddings_opposed_neg[:, :, 0, 0, 3] = torch.tensor([0.0, -1.0])

    same_loss = supervised_contrastive_loss(
        embeddings_same_neg, labels, target_id, temperature=0.1
    )
    opposed_loss = supervised_contrastive_loss(
        embeddings_opposed_neg, labels, target_id, temperature=0.1
    )
    assert float(same_loss) == pytest.approx(float(opposed_loss), abs=1e-6)


def test_supervised_contrastive_same_line_positives_attract_across_crops():
    labels = torch.full((2, 1, 1, 2), IGNORE_INDEX, dtype=torch.long)
    labels[0, 0, 0, 0] = POSITIVE_LABEL
    labels[1, 0, 0, 0] = POSITIVE_LABEL
    labels[0, 0, 0, 1] = NEGATIVE_LABEL
    target_id = torch.full_like(labels, IGNORE_ID)
    target_id[0, 0, 0, 0] = 9
    target_id[1, 0, 0, 0] = 9
    target_id[0, 0, 0, 1] = NEGATIVE_ONLY_ID

    aligned = torch.zeros((2, 2, 1, 1, 2), dtype=torch.float32)
    aligned[0, :, 0, 0, 0] = torch.tensor([1.0, 0.0])
    aligned[1, :, 0, 0, 0] = torch.tensor([1.0, 0.0])
    aligned[0, :, 0, 0, 1] = torch.tensor([0.0, 1.0])
    opposed = aligned.clone()
    opposed[1, :, 0, 0, 0] = torch.tensor([-1.0, 0.0])

    aligned_loss = supervised_contrastive_loss(
        aligned, labels, target_id, temperature=0.1
    )
    opposed_loss = supervised_contrastive_loss(
        opposed, labels, target_id, temperature=0.1
    )
    assert float(aligned_loss) < float(opposed_loss)


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
    outputs = model(batch.volume, batch.cond_fw_xyz, batch.cond_up_xyz)
    assert outputs["embedding"].shape == (2, 6, 8, 8, 8)
    assert outputs["fw"].shape == (2, 3, 8, 8, 8)

    losses = compute_fiber_trace_loss(outputs, batch, max_contrastive_samples=512)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert any(param.grad is not None for param in model.parameters())


def test_test_fiber_glob_builds_separate_test_config():
    config = {
        "datasets": [
            {
                "base_volume_path": "/base.zarr",
                "lasagna_manifest_path": "/pred.lasagna.json",
                "fiber_glob": "/train/*.json",
                "test_fiber_glob": "/test/*.json",
            }
        ]
    }

    test_config = fiber_train._make_test_config(config)

    assert test_config is not None
    assert test_config["datasets"][0]["fiber_glob"] == "/test/*.json"
    assert "test_fiber_glob" in test_config["datasets"][0]


def test_training_writes_tensorboard_text_and_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class FakeWriter:
        def __init__(self, log_dir: Path) -> None:
            self.log_dir = Path(log_dir)
            self.scalars: list[tuple[str, float, int]] = []
            self.texts: list[tuple[str, str, int]] = []
            self.closed = False

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars.append((tag, float(value), int(step)))

        def add_text(self, tag: str, text: str, step: int) -> None:
            self.texts.append((tag, str(text), int(step)))

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    writers: list[FakeWriter] = []

    def fake_make_summary_writer(log_dir: Path, *, enabled: bool):
        assert enabled
        writer = FakeWriter(log_dir)
        writers.append(writer)
        return writer

    monkeypatch.setattr(fiber_train, "_make_summary_writer", fake_make_summary_writer)

    config = _synthetic_config(tmp_path, batch_size=2, crop_size=(8, 8, 8))
    config.update(
        {
            "device": "cpu",
            "num_steps": 1,
            "log_every": 100,
            "run_path": str(tmp_path / "runs"),
            "run_name": "unit run",
            "run_datestr": "20260102_030405",
            "_test_array_records": config["_array_records"],
            "model": {
                "input_channels": 1,
                "backbone_channels": 2,
                "embedding_dim": 4,
                "features_per_stage": [2],
                "head_channels": 4,
            },
            "loss": {"max_contrastive_samples": 256},
        }
    )

    result = fiber_train.run_training(config)

    run_dir = tmp_path / "runs" / "unit_run_20260102_030405"
    assert Path(result["run_dir"]) == run_dir
    assert (run_dir / "snapshots" / "current.pt").is_file()
    assert (run_dir / "snapshots" / "best.pt").is_file()
    assert writers and writers[0].closed
    assert writers[0].log_dir == run_dir
    assert any(
        tag == "config/json" and '"run_name": "unit run"' in text
        for tag, text, _ in writers[0].texts
    )
    scalar_tags = {tag for tag, _, _ in writers[0].scalars}
    assert {
        "train/total",
        "train/contrastive",
        "train/fw",
        "train/up",
    } <= scalar_tags
    assert {"test/total", "test/contrastive", "test/fw", "test/up"} <= scalar_tags


def test_training_rejects_legacy_checkpoint_path(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=2, crop_size=(8, 8, 8))
    config["checkpoint_path"] = str(tmp_path / "model.pt")
    with pytest.raises(ValueError, match="checkpoint_path was replaced"):
        fiber_train.run_training(config)
