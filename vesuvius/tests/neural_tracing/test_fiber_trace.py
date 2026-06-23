from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

import vesuvius.neural_tracing.fiber_trace.dataset as fiber_dataset
import vesuvius.neural_tracing.fiber_trace.train as fiber_train
from vesuvius.neural_tracing.fiber_trace.dataset import (
    FiberTraceBatch,
    FiberTraceBatchBuilder,
)
from vesuvius.neural_tracing.fiber_trace.fiber_json import parse_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace.geometry import (
    classify_voxels,
    decode_lasagna_normals_xyz,
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
    sample_contrastive_pair_indices,
    supervised_contrastive_loss,
)
from vesuvius.neural_tracing.fiber_trace.model import (
    DirectionConditionedFiberTraceModel,
    build_fiber_trace_model,
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
        "positive_direction_jitter_degrees": 30.0,
        "normal_plane_jitter_voxels": 40.0,
        "normal_perpendicular_jitter_voxels": 10.0,
        "positive_along_fiber_limit_voxels": 40.0,
        "negative_cone_distance_voxels": 30.0,
        "random_negative_pool_size": 8,
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


def test_lasagna_normal_decoding():
    normals, valid = decode_lasagna_normals_xyz(
        np.array([128, 128, 255, 0], dtype=np.uint8),
        np.array([128, 255, 128, 0], dtype=np.uint8),
    )
    np.testing.assert_allclose(normals[0], np.array([0.0, 0.0, 1.0]), atol=1e-6)
    np.testing.assert_allclose(normals[1], np.array([0.0, 1.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(normals[2], np.array([1.0, 0.0, 0.0]), atol=1e-6)
    assert bool(valid.all())
    assert np.isfinite(normals[3]).all()
    assert float(np.linalg.norm(normals[3])) == pytest.approx(1.0, abs=1e-6)


def test_batch_builder_samples_one_fiber_with_mixed_and_random_negative_crops(
    tmp_path: Path,
):
    builder = FiberTraceBatchBuilder(_synthetic_config(tmp_path))
    batch = fiber_train.classify_batch_on_device(
        builder.sample_batch(record_index=0), builder.config
    )

    assert batch.volume.shape == (4, 1, 16, 16, 16)
    assert batch.labels.shape == (4, 16, 16, 16)
    assert batch.target_id.shape == (4, 16, 16, 16)
    assert batch.crop_kinds.count("gt_control") == 3
    assert batch.crop_kinds.count("random_negative") == 1
    assert batch.direction_kinds == ("gt_jitter", "gt_jitter", "gt_jitter", "random")
    assert len(set(batch.fiber_paths)) == 1
    assert bool((batch.labels[:3] == POSITIVE_LABEL).any())
    assert bool((batch.labels[3:][batch.valid_mask[3:]] == NEGATIVE_LABEL).all())
    assert not bool((batch.labels[3:] == POSITIVE_LABEL).any())
    assert bool((batch.target_id[batch.labels == POSITIVE_LABEL] == 0).all())
    assert bool(
        (batch.target_id[batch.labels == NEGATIVE_LABEL] == NEGATIVE_ONLY_ID).all()
    )
    target_fw = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    gt_cond = torch.nn.functional.normalize(batch.cond_fw_xyz[:3], dim=1).numpy()
    gt_errors = np.degrees(np.arccos(np.clip(gt_cond @ target_fw, -1.0, 1.0)))
    assert bool((gt_errors >= -1e-4).all())
    assert bool((gt_errors <= 30.0 + 1e-4).all())


def test_visualization_sample_selection_uses_first_two_gt_crops_only(tmp_path: Path):
    builder = FiberTraceBatchBuilder(
        _synthetic_config(tmp_path, batch_size=8, crop_size=(16, 16, 16))
    )
    batch = builder.sample_batch(record_index=0, iteration=0)

    assert batch.crop_kinds.count("gt_control") == 6
    assert batch.crop_kinds.count("random_negative") == 2
    assert fiber_train._visualization_positive_sample_indices(
        batch, fallback_index=0
    ) == [0, 1]


def test_debug_cache_logs_one_table_row_per_batch(tmp_path: Path, capsys):
    config = _synthetic_config(tmp_path, batch_size=4, crop_size=(8, 8, 8))
    config["debug_cache"] = True
    builder = FiberTraceBatchBuilder(config)

    builder.sample_batch(record_index=0, iteration=7)

    output = capsys.readouterr().out
    lines = output.splitlines()
    assert lines[0].startswith("fiber_trace batch columns:")
    assert "data=batch data-loading ms" in lines[0]
    assert "throughput" in lines[0]
    assert "it" in lines[1]
    assert "data" in lines[1]
    assert "rec" not in lines[1]
    assert "k" not in lines[1]
    assert "ctl" not in lines[1]
    assert "hMiB/s" in lines[1]
    assert "dMiB/s" in lines[1]
    assert {"hit", "dl", "mis", "hms", "dms"} <= set(lines[1].split())
    data_lines = [line for line in lines if line.split()[:1] == ["7"]]
    assert len(data_lines) == 1
    assert data_lines[0].split()[0] == "7"
    assert "=" not in data_lines[0]
    assert "oz" not in lines[1]
    assert "bz" not in lines[1]
    assert "[fiber_trace:patch]" not in output
    assert "[fiber_trace:sample]" not in output
    assert "_DiskCacheStore" not in output


def test_iteration_sampling_is_deterministic_and_stateless(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=4, crop_size=(8, 8, 8))
    first_builder = FiberTraceBatchBuilder(config)
    first = first_builder.sample_batch(record_index=0, iteration=5)
    _ = first_builder.sample_batch(record_index=0, iteration=2)
    repeated = first_builder.sample_batch(record_index=0, iteration=5)

    second_builder = FiberTraceBatchBuilder(config)
    second = second_builder.sample_batch(record_index=0, iteration=5)

    for other in (repeated, second):
        torch.testing.assert_close(other.volume, first.volume)
        torch.testing.assert_close(other.mask_values, first.mask_values)
        torch.testing.assert_close(other.cond_fw_xyz, first.cond_fw_xyz)
        torch.testing.assert_close(other.crop_origin_zyx, first.crop_origin_zyx)
        torch.testing.assert_close(other.sample_local_zyx, first.sample_local_zyx)
        assert other.crop_kinds == first.crop_kinds
        assert other.direction_kinds == first.direction_kinds


def test_sample_limit_reuses_initial_deterministic_samples(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=4, crop_size=(8, 8, 8))
    config["sample_limit"] = 2
    builder = FiberTraceBatchBuilder(config)

    first = builder.sample_batch(record_index=0, iteration=1)
    second = builder.sample_batch(record_index=0, iteration=2)
    third = builder.sample_batch(record_index=0, iteration=3)
    fourth = builder.sample_batch(record_index=0, iteration=4)

    torch.testing.assert_close(third.volume, first.volume)
    torch.testing.assert_close(third.cond_fw_xyz, first.cond_fw_xyz)
    torch.testing.assert_close(third.crop_origin_zyx, first.crop_origin_zyx)
    torch.testing.assert_close(third.sample_local_zyx, first.sample_local_zyx)
    torch.testing.assert_close(fourth.volume, second.volume)
    torch.testing.assert_close(fourth.cond_fw_xyz, second.cond_fw_xyz)
    torch.testing.assert_close(fourth.crop_origin_zyx, second.crop_origin_zyx)
    torch.testing.assert_close(fourth.sample_local_zyx, second.sample_local_zyx)


def test_sample_limit_rejects_non_positive_values(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=4, crop_size=(8, 8, 8))
    config["sample_limit"] = 0
    with pytest.raises(ValueError, match="sample_limit"):
        FiberTraceBatchBuilder(config)


def test_control_point_crop_offset_keeps_configured_margin(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=2, crop_size=(16, 16, 16))
    config["control_point_margin_voxels"] = 3
    builder = FiberTraceBatchBuilder(config)

    batch = builder.sample_batch(record_index=0, iteration=4)
    origin = batch.crop_origin_zyx[0].numpy()
    controls_zyx = np.array([[8, 8, 8], [8, 8, 10]], dtype=np.int64)
    local_options = controls_zyx - origin[None, :]

    assert any(
        bool(np.all((local >= 3) & (local <= 12))) for local in local_options
    )
    sample_local = batch.sample_local_zyx[0].numpy()
    assert bool(np.all((sample_local >= 3) & (sample_local <= 12)))

    locals_by_iteration = np.stack(
        [
            builder.sample_batch(record_index=0, iteration=iteration)
            .sample_local_zyx[0]
            .numpy()
            for iteration in range(1, 9)
        ],
        axis=0,
    )
    assert bool(np.all((locals_by_iteration >= 3) & (locals_by_iteration <= 12)))
    assert bool(((locals_by_iteration > 3) & (locals_by_iteration < 12)).any())


def test_control_point_margin_rejects_impossible_crop(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=2, crop_size=(16, 16, 16))
    config["control_point_margin_voxels"] = 8
    with pytest.raises(ValueError, match="control_point_margin_voxels"):
        FiberTraceBatchBuilder(config)


def test_augmentation_crop_size_is_center_trimmed_to_crop_size(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=4, crop_size=(8, 8, 8))
    config["augmentation_crop_size"] = [12, 12, 12]
    builder = FiberTraceBatchBuilder(config)

    crop_base = builder._sample_crop_base(
        builder.records[0], crop_kind="gt_control", control_index=0
    )

    assert crop_base["volume"].shape == (8, 8, 8)
    assert crop_base["mask_values"].shape == (8, 8, 8)
    assert crop_base["nx_values"].shape == (8, 8, 8)
    assert crop_base["ny_values"].shape == (8, 8, 8)
    np.testing.assert_array_equal(
        crop_base["origin"], np.array([4, 4, 4], dtype=np.int64)
    )


def test_sampled_control_outside_normals_reports_mapping(tmp_path: Path):
    fiber_path = tmp_path / "outside_fiber.json"
    fiber_payload = {
        "type": "vc3d_fiber",
        "version": 1,
        "line_points": [[float(x), 8.0, 8.0] for x in range(20, 30)],
        "control_points": [[1000.0, 8.0, 8.0]],
    }
    fiber_path.write_text(json.dumps(fiber_payload), encoding="utf-8")
    config = _synthetic_config(tmp_path, batch_size=4, crop_size=(8, 8, 8))
    config["_array_records"][0]["fiber_path"] = str(fiber_path)

    builder = FiberTraceBatchBuilder(config)
    with pytest.raises(ValueError, match="nx_zyx=.*reason='mapped outside nx/ny volume'"):
        builder.sample_batch(record_index=0)


def test_missing_lasagna_manifest_fails_loudly(tmp_path: Path):
    fiber_path = tmp_path / "fiber.json"
    _write_fiber(fiber_path)
    config = {
        "crop_size": [8, 8, 8],
        "batch_size": 2,
        "datasets": [
            {
                "base_volume_path": str(tmp_path / "missing.zarr"),
                "fiber_paths": [str(fiber_path)],
            }
        ],
    }
    with pytest.raises(ValueError, match="lasagna_manifest_path"):
        FiberTraceBatchBuilder(config)


def test_valid_mask_threshold_key_is_removed(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=2, crop_size=(8, 8, 8))
    config["_array_records"][0]["valid_mask_threshold"] = 0.0
    with pytest.raises(ValueError, match="valid_mask_threshold was removed"):
        FiberTraceBatchBuilder(config)


def test_removed_direction_label_threshold_keys_are_rejected(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=2, crop_size=(8, 8, 8))
    config["negative_direction_max_degrees"] = 90.0
    with pytest.raises(ValueError, match="negative_direction_max_degrees was removed"):
        FiberTraceBatchBuilder(config)


def test_manifest_missing_normals_fails_loudly(
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
                        "zarr": "grad_mag.ome.zarr/0",
                        "scaledown": 0,
                        "channels": ["grad_mag"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    base = np.zeros((16, 16, 16), dtype=np.uint8)
    grad = np.ones((16, 16, 16), dtype=np.uint8)
    base_root = tmp_path / "base.ome.zarr"
    grad_root = tmp_path / "grad_mag.ome.zarr"

    def fake_open_zarr(path, *, scale=None, auth_json_path=None, config=None):
        if str(path) == str(base_root.resolve()) and int(scale) == 0:
            return base
        if str(path) == str(grad_root.resolve()) and int(scale) == 0:
            return grad
        raise AssertionError((path, scale))

    monkeypatch.setattr(fiber_dataset, "_common_open_zarr", fake_open_zarr)
    config = {
        "crop_size": [8, 8, 8],
        "batch_size": 2,
        "datasets": [
            {
                "base_volume_path": str(base_root),
                "lasagna_manifest_path": str(manifest_path),
                "fiber_paths": [str(fiber_path)],
            }
        ],
    }
    with pytest.raises(ValueError, match="missing required nx/ny"):
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
                "base_volume_path": "https://example.com/volume.zarr",
                "lasagna_manifest_path": str(tmp_path / "pred.lasagna.json"),
                "fiber_paths": [str(fiber_path)],
            }
        ],
    }
    with pytest.raises(ValueError, match="volume_cache_dir"):
        FiberTraceBatchBuilder(config)


def test_manifest_zarr_access_uses_common_helpers_without_direct_fallback(
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
                        "zarr": "mask.ome.zarr/0",
                        "scaledown": 0,
                        "channels": ["grad_mag"],
                    },
                    "normal": {
                        "zarr": "normal.ome.zarr/0",
                        "scaledown": 0,
                        "channels": ["nx", "ny"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    volume = np.zeros((16, 16, 16), dtype=np.uint8)
    mask = np.ones((16, 16, 16), dtype=np.uint8)
    normals = np.stack(
        [
            np.full((16, 16, 16), 128, dtype=np.uint8),
            np.full((16, 16, 16), 128, dtype=np.uint8),
        ],
        axis=0,
    )
    calls: list[tuple[str, str, int | None]] = []

    def fake_open_zarr(path, *, scale=None, auth_json_path=None, config=None):
        calls.append(("array", str(path), int(scale)))
        return {
            str((tmp_path / "volume.ome.zarr").resolve()): volume,
            str((tmp_path / "mask.ome.zarr").resolve()): mask,
            str((tmp_path / "normal.ome.zarr").resolve()): normals,
        }[str(path)]

    monkeypatch.setattr(fiber_dataset, "_common_open_zarr", fake_open_zarr)

    FiberTraceBatchBuilder(
        {
            "crop_size": [8, 8, 8],
            "batch_size": 2,
            "datasets": [
                {
                    "base_volume_path": str(tmp_path / "volume.ome.zarr"),
                    "lasagna_manifest_path": str(manifest_path),
                    "fiber_paths": [str(fiber_path)],
                }
            ],
        }
    )
    assert calls == [
        ("array", str((tmp_path / "volume.ome.zarr").resolve()), 0),
        ("array", str((tmp_path / "mask.ome.zarr").resolve()), 0),
        ("array", str((tmp_path / "normal.ome.zarr").resolve()), 0),
        ("array", str((tmp_path / "normal.ome.zarr").resolve()), 0),
    ]


def test_manifestless_channel_paths_are_rejected(tmp_path: Path):
    fiber_path = tmp_path / "fiber.json"
    _write_fiber(fiber_path)
    config = {
        "crop_size": [8, 8, 8],
        "batch_size": 2,
        "datasets": [
            {
                "base_volume_path": str(tmp_path / "volume.ome.zarr"),
                "lasagna_manifest_path": str(tmp_path / "pred.lasagna.json"),
                "grad_mag_path": str(tmp_path / "mask.ome.zarr"),
                "nx_path": str(tmp_path / "nx.ome.zarr"),
                "ny_path": str(tmp_path / "ny.ome.zarr"),
                "fiber_paths": [str(fiber_path)],
            }
        ],
    }
    with pytest.raises(ValueError, match="manifest-less"):
        FiberTraceBatchBuilder(config)


def test_lasagna_manifest_source_to_base_contributes_channel_spacing(
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
                "source_to_base": 2.0,
                "base_shape_zyx": [16, 16, 16],
                "umbilicus_json": "umbilicus.json",
                "groups": {
                    "grad_mag": {
                        "zarr": "grad_mag.ome.zarr/1",
                        "scaledown": 1,
                        "channels": ["grad_mag"],
                    },
                    "normal": {
                        "zarr": "normal.ome.zarr/1",
                        "scaledown": 1,
                        "channels": ["nx", "ny"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    base = np.zeros((16, 16, 16), dtype=np.uint8)
    grad = np.ones((8, 8, 8), dtype=np.uint8)
    normals = np.stack(
        [
            np.full((8, 8, 8), 128, dtype=np.uint8),
            np.full((8, 8, 8), 128, dtype=np.uint8),
        ],
        axis=0,
    )
    base_root = tmp_path / "base.ome.zarr"
    grad_root = tmp_path / "grad_mag.ome.zarr"
    normal_root = tmp_path / "normal.ome.zarr"

    def fake_open_zarr(path, *, scale=None, auth_json_path=None, config=None):
        if str(path) == str(base_root.resolve()) and int(scale) == 0:
            return base
        if str(path) == str(grad_root.resolve()) and int(scale) == 1:
            return grad
        if str(path) == str(normal_root.resolve()) and int(scale) == 1:
            return normals
        raise AssertionError((path, scale))

    monkeypatch.setattr(fiber_dataset, "_common_open_zarr", fake_open_zarr)
    builder = FiberTraceBatchBuilder(
        {
            "crop_size": [8, 8, 8],
            "batch_size": 2,
            "datasets": [
                {
                    "base_volume_path": str(base_root),
                    "lasagna_manifest_path": str(manifest_path),
                    "fiber_paths": [str(fiber_path)],
                }
            ],
        }
    )
    record = builder.records[0]
    assert record.mask_spacing_base == pytest.approx(4.0)
    assert record.nx_spacing_base == pytest.approx(4.0)
    assert record.ny_spacing_base == pytest.approx(4.0)


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
            "positive_direction_jitter_degrees": 0.0,
            "normal_plane_jitter_voxels": 40.0,
            "normal_perpendicular_jitter_voxels": 10.0,
            "positive_along_fiber_limit_voxels": 40.0,
            "negative_cone_distance_voxels": 30.0,
            "datasets": [
                {
                    "base_volume_path": str(base_root),
                    "base_volume_scale": 1,
                    "lasagna_manifest_path": str(manifest_path),
                    "fiber_paths": [str(fiber_path)],
                }
            ],
        },
    )
    batch = fiber_train.classify_batch_on_device(
        builder.sample_batch(record_index=0), builder.config
    )

    assert batch.volume.shape == (2, 1, 4, 4, 4)
    assert bool((batch.labels == POSITIVE_LABEL).any())
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
        valid_mask=mask,
        positive_center_zyx=np.array([5, 3, 4], dtype=np.int64),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=1.0,
        normal_perpendicular_jitter_voxels=1.0,
        positive_along_fiber_limit_voxels=1.0,
        negative_cone_distance_voxels=3.0,
        positive_target_id=7,
    )
    assert int(result["labels"][5, 3, 4]) == POSITIVE_LABEL
    assert int(result["target_id"][5, 3, 4]) == 7
    assert int(result["labels"][5, 3, 6]) == IGNORE_INDEX
    assert int(result["labels"][5, 5, 4]) == IGNORE_INDEX
    assert int(result["labels"][7, 5, 4]) == IGNORE_INDEX
    assert int(result["labels"][1, 1, 1]) == IGNORE_INDEX
    assert int(result["target_id"][1, 1, 1]) == IGNORE_ID
    np.testing.assert_allclose(
        result["target_fw_xyz"][:, 5, 3, 4],
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )


def test_voxel_classification_cone_negatives_and_positive_precedence():
    line = np.array([[2.0, 3.0, 5.0], [6.0, 3.0, 5.0]], dtype=np.float32)
    normal_xyz = np.zeros((9, 9, 9, 3), dtype=np.float32)
    normal_xyz[..., 2] = 1.0
    result = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        positive_center_zyx=np.array([5, 3, 4], dtype=np.int64),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=1.0,
        normal_perpendicular_jitter_voxels=1.0,
        positive_along_fiber_limit_voxels=1.0,
        negative_cone_distance_voxels=3.0,
        positive_target_id=11,
    )
    assert int(result["labels"][8, 3, 4]) == NEGATIVE_LABEL
    assert int(result["target_id"][8, 3, 4]) == NEGATIVE_ONLY_ID
    assert int(result["labels"][2, 3, 4]) == NEGATIVE_LABEL
    assert int(result["labels"][8, 4, 4]) == IGNORE_INDEX
    assert int(result["labels"][8, 6, 4]) == IGNORE_INDEX
    assert int(result["labels"][8, 3, 8]) == IGNORE_INDEX

    overlap = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        positive_center_zyx=np.array([5, 3, 4], dtype=np.int64),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=3.0,
        normal_perpendicular_jitter_voxels=3.0,
        positive_along_fiber_limit_voxels=3.0,
        negative_cone_distance_voxels=2.0,
        positive_target_id=12,
    )
    assert int(overlap["labels"][7, 5, 4]) == POSITIVE_LABEL
    assert int(overlap["target_id"][7, 5, 4]) == 12


def test_device_classification_uses_normal_axis_cone_negatives():
    shape = (9, 9, 9)
    batch = FiberTraceBatch(
        volume=torch.zeros((1, 1) + shape, dtype=torch.float32),
        mask_values=torch.ones((1,) + shape, dtype=torch.uint8),
        nx_values=torch.full((1,) + shape, 255, dtype=torch.uint8),
        ny_values=torch.full((1,) + shape, 128, dtype=torch.uint8),
        valid_mask=torch.zeros((1,) + shape, dtype=torch.bool),
        labels=torch.full((1,) + shape, IGNORE_INDEX, dtype=torch.long),
        target_id=torch.full((1,) + shape, IGNORE_ID, dtype=torch.long),
        target_fw_xyz=torch.zeros((1, 3) + shape, dtype=torch.float32),
        center_normal_xyz=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        cond_fw_xyz=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        crop_origin_zyx=torch.zeros((1, 3), dtype=torch.long),
        sample_local_zyx=torch.tensor([[4, 4, 4]], dtype=torch.long),
        line_points_xyz=torch.tensor(
            [[[2.0, 3.0, 5.0], [6.0, 3.0, 5.0]]], dtype=torch.float32
        ),
        positive_target_id=torch.tensor([4], dtype=torch.long),
        crop_kinds=("unit",),
        fiber_paths=("",),
        direction_kinds=("gt_jitter",),
    )

    classified = fiber_train.classify_batch_on_device(
        batch,
        {
            "normal_plane_jitter_voxels": 1.0,
            "normal_perpendicular_jitter_voxels": 1.0,
            "positive_along_fiber_limit_voxels": 1.0,
            "negative_cone_distance_voxels": 3.0,
        },
    )

    assert int(classified.labels[0, 8, 3, 4]) == NEGATIVE_LABEL
    assert int(classified.labels[0, 2, 3, 4]) == NEGATIVE_LABEL
    assert int(classified.labels[0, 8, 4, 4]) == IGNORE_INDEX


def test_voxel_classification_ignores_conditioning_direction_for_labels():
    line = np.array([[2.0, 3.0, 5.0], [6.0, 3.0, 5.0]], dtype=np.float32)
    normal_xyz = np.zeros((9, 9, 9, 3), dtype=np.float32)
    normal_xyz[..., 2] = 1.0
    positive = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        positive_center_zyx=np.array([5, 3, 4], dtype=np.int64),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=1.0,
        normal_perpendicular_jitter_voxels=1.0,
        positive_along_fiber_limit_voxels=1.0,
        negative_cone_distance_voxels=3.0,
    )
    assert int(positive["labels"][5, 3, 4]) == POSITIVE_LABEL

    rotated = classify_voxels(
        crop_origin_zyx=np.array([0, 0, 0], dtype=np.int64),
        crop_shape=(9, 9, 9),
        line_points_xyz=line,
        cond_fw_xyz=np.array([0.5, np.sqrt(3.0) / 2.0, 0.0], dtype=np.float32),
        valid_mask=np.ones((9, 9, 9), dtype=bool),
        positive_center_zyx=np.array([5, 3, 4], dtype=np.int64),
        normal_xyz=normal_xyz,
        normal_valid_mask=np.ones((9, 9, 9), dtype=bool),
        normal_plane_jitter_voxels=1.0,
        normal_perpendicular_jitter_voxels=1.0,
        positive_along_fiber_limit_voxels=1.0,
        negative_cone_distance_voxels=3.0,
    )
    assert int(rotated["labels"][5, 3, 4]) == POSITIVE_LABEL
    np.testing.assert_array_equal(rotated["labels"], positive["labels"])


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
    builder = FiberTraceBatchBuilder(config)
    batch = fiber_train.classify_batch_on_device(
        builder.sample_batch(record_index=0), builder.config
    )
    model = DirectionConditionedFiberTraceModel(
        backbone_channels=2,
        embedding_dim=6,
        features_per_stage=(2,),
        head_channels=4,
    )
    outputs = model(batch.volume, batch.cond_fw_xyz)
    assert outputs["embedding"].shape == (2, 6, 8, 8, 8)
    assert outputs["fw"].shape == (2, 3, 8, 8, 8)
    assert "up" not in outputs

    samples = sample_contrastive_pair_indices(
        batch.labels, batch.target_id, max_samples=512
    )
    assert samples.flat_indices.numel() > 0
    sampled_outputs = model(
        batch.volume, batch.cond_fw_xyz, sample_indices=samples.flat_indices
    )
    assert sampled_outputs["embedding"].shape == (
        samples.flat_indices.numel(),
        6,
    )
    dense_embedding = outputs["embedding"].permute(0, 2, 3, 4, 1).reshape(-1, 6)
    torch.testing.assert_close(
        sampled_outputs["embedding"], dense_embedding[samples.flat_indices]
    )

    losses = compute_fiber_trace_loss(
        sampled_outputs,
        batch,
        max_contrastive_samples=512,
        contrastive_samples=samples,
    )
    assert torch.isfinite(losses.total)
    assert float(losses.total.detach()) == pytest.approx(
        float(losses.contrastive.detach()), abs=1e-6
    )
    losses.total.backward()
    assert any(param.grad is not None for param in model.parameters())


def test_contrastive_pair_sampler_shuffles_full_batch_lists():
    labels = torch.full((2, 1, 1, 8), IGNORE_INDEX, dtype=torch.long)
    target_id = torch.full_like(labels, IGNORE_ID)
    labels[0, 0, 0, 0] = POSITIVE_LABEL
    labels[1, 0, 0, 0] = POSITIVE_LABEL
    target_id[labels == POSITIVE_LABEL] = 7
    labels[0, 0, 0, 1:] = NEGATIVE_LABEL
    labels[1, 0, 0, 1:] = NEGATIVE_LABEL
    target_id[labels == NEGATIVE_LABEL] = NEGATIVE_ONLY_ID

    first = sample_contrastive_pair_indices(
        labels, target_id, max_samples=6, seed=11
    )
    again = sample_contrastive_pair_indices(
        labels, target_id, max_samples=6, seed=11
    )
    changed = sample_contrastive_pair_indices(
        labels, target_id, max_samples=6, seed=12
    )

    first_flat = first.flat_indices
    pos_left = first_flat[first.pos_a]
    pos_right = first_flat[first.pos_b]
    neg_left = first_flat[first.neg_a]
    neg_right = first_flat[first.neg_b]

    assert first.pos_a.numel() == 3
    assert first.neg_a.numel() == 3
    assert set(pos_left.tolist()) <= {0, 8}
    assert set(pos_right.tolist()) <= {0, 8}
    assert bool(torch.all(pos_left != pos_right))
    assert set(neg_left.tolist()) <= {0, 8}
    assert all(int(item) not in {0, 8} for item in neg_right.tolist())

    torch.testing.assert_close(first.flat_indices, again.flat_indices)
    torch.testing.assert_close(first.pos_a, again.pos_a)
    torch.testing.assert_close(first.pos_b, again.pos_b)
    torch.testing.assert_close(first.neg_a, again.neg_a)
    torch.testing.assert_close(first.neg_b, again.neg_b)

    changed_neg_right = changed.flat_indices[changed.neg_b]
    assert not torch.equal(neg_right, changed_neg_right)


def test_model_builder_derives_requested_unet_depth_and_condition_width():
    model = build_fiber_trace_model(
        {
            "model": {
                "input_channels": 1,
                "unet_base_channels": 16,
                "unet_depth": 7,
                "conditioned_feature_channels": 64,
                "embedding_dim": 16,
                "head_channels": 64,
            }
        }
    )

    assert model.backbone.features_per_stage == [16, 32, 64, 128, 256, 512, 1024]
    assert model.backbone.strides == [
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ]
    assert model.conditioned_feature_channels == 64
    assert model.embedding_dim == 16
    assert model.head[0].in_channels == 67
    assert model.head[0].out_channels == 64
    assert model.head[2].out_channels == 19


def test_sample_plane_visualization_fuses_labels_to_uint8_values():
    volume = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
    labels = torch.full((3, 3, 3), IGNORE_INDEX, dtype=torch.long)
    labels[1, 0, 0] = NEGATIVE_LABEL
    labels[1, 1, 1] = POSITIVE_LABEL
    embedding_similarity = torch.zeros((3, 3, 3), dtype=torch.float32)
    embedding_similarity[1, 0, 0] = -1.0
    embedding_similarity[1, 1, 1] = 1.0
    grid = fiber_train._slice_grid(
        shape_zyx=(3, 3, 3),
        center_xyz=torch.tensor([1.0, 1.0, 1.0]),
        axis_u_xyz=torch.tensor([1.0, 0.0, 0.0]),
        axis_v_xyz=torch.tensor([0.0, 1.0, 0.0]),
        size=5,
        device=torch.device("cpu"),
    )

    images = fiber_train._sample_plane_image(
        volume, labels, grid, embedding_similarity=embedding_similarity
    )

    label_image = images["labels"]
    assert label_image.dtype == torch.uint8
    assert int(label_image[1, 1]) == 0
    assert int(label_image[2, 2]) == 255
    assert int(label_image[0, 0]) == 63
    cos_image = images["cos_emb_cp"]
    assert float(cos_image[1, 1]) == pytest.approx(0.0, abs=1e-6)
    assert float(cos_image[2, 2]) == pytest.approx(1.0, abs=1e-6)
    assert float(cos_image[0, 0]) == pytest.approx(63.0 / 255.0, abs=1e-6)


def test_sample_plane_grid_even_size_has_reference_pixel():
    grid = fiber_train._slice_grid(
        shape_zyx=(8, 8, 8),
        center_xyz=torch.tensor([4.0, 4.0, 4.0]),
        axis_u_xyz=torch.tensor([1.0, 0.0, 0.0]),
        axis_v_xyz=torch.tensor([0.0, 1.0, 0.0]),
        size=8,
        device=torch.device("cpu"),
    )

    center_grid = grid[0, 0, 4, 4]

    torch.testing.assert_close(
        center_grid,
        torch.tensor([1.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0], dtype=torch.float32),
    )


def test_sample_plane_visualization_marks_reference_cross_without_overwriting_center():
    volume = torch.zeros((1, 1, 5, 5, 5), dtype=torch.float32)
    labels = torch.full((5, 5, 5), IGNORE_INDEX, dtype=torch.long)
    labels[2, 2, 2] = POSITIVE_LABEL
    embedding_similarity = torch.zeros((5, 5, 5), dtype=torch.float32)
    embedding_similarity[2, 2, 2] = 1.0
    grid = fiber_train._slice_grid(
        shape_zyx=(5, 5, 5),
        center_xyz=torch.tensor([2.0, 2.0, 2.0]),
        axis_u_xyz=torch.tensor([1.0, 0.0, 0.0]),
        axis_v_xyz=torch.tensor([0.0, 1.0, 0.0]),
        size=5,
        device=torch.device("cpu"),
    )

    images = fiber_train._sample_plane_image(
        volume,
        labels,
        grid,
        embedding_similarity=embedding_similarity,
        center_marker_yx=(2, 2),
    )

    assert float(images["cos_emb_cp"][2, 2]) == pytest.approx(1.0, abs=1e-6)
    assert int(images["labels"][2, 2]) == 255
    assert float(images["image"][2, 1]) == pytest.approx(0.0, abs=1e-6)
    assert float(images["image"][2, 3]) == pytest.approx(0.0, abs=1e-6)
    assert float(images["image"][2, 0]) == pytest.approx(1.0, abs=1e-6)
    assert float(images["image"][2, 4]) == pytest.approx(1.0, abs=1e-6)
    assert float(images["cos_emb_cp"][2, 1]) == pytest.approx(0.5, abs=1e-6)
    assert float(images["cos_emb_cp"][2, 3]) == pytest.approx(0.5, abs=1e-6)
    assert int(images["labels"][2, 1]) == 127
    assert int(images["labels"][2, 3]) == 127
    assert int(images["labels"][0, 2]) == 0
    assert int(images["labels"][4, 2]) == 0


def test_training_sample_visualization_uses_cp_local_reference_not_crop_center():
    class FakeWriter:
        def __init__(self) -> None:
            self.images: dict[str, torch.Tensor] = {}

        def add_image(self, tag: str, image: torch.Tensor, step: int) -> None:
            del step
            self.images[tag] = image.detach().clone()

    class FakeModel(torch.nn.Module):
        def forward(
            self, volume: torch.Tensor, cond_fw_xyz: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            del cond_fw_xyz
            batch_size = int(volume.shape[0])
            embedding = torch.zeros((batch_size, 2, 8, 8, 8), dtype=volume.dtype)
            embedding[0, 1] = 1.0
            embedding[1, 0] = 1.0
            embedding[0, :, 1, 2, 3] = torch.tensor([1.0, 0.0])
            embedding[1, :, 5, 4, 2] = torch.tensor([0.0, 1.0])
            return {"embedding": embedding}

    batch = FiberTraceBatch(
        volume=torch.zeros((2, 1, 8, 8, 8), dtype=torch.float32),
        mask_values=torch.ones((2, 8, 8, 8), dtype=torch.float32),
        nx_values=None,
        ny_values=None,
        valid_mask=torch.ones((2, 8, 8, 8), dtype=torch.bool),
        labels=torch.full((2, 8, 8, 8), IGNORE_INDEX, dtype=torch.long),
        target_id=torch.full((2, 8, 8, 8), IGNORE_ID, dtype=torch.long),
        target_fw_xyz=torch.zeros((2, 3, 8, 8, 8), dtype=torch.float32),
        center_normal_xyz=torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
        cond_fw_xyz=torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        ),
        crop_origin_zyx=torch.zeros((2, 3), dtype=torch.long),
        sample_local_zyx=torch.tensor([[1, 2, 3], [5, 4, 2]], dtype=torch.long),
        line_points_xyz=torch.zeros((2, 2, 3), dtype=torch.float32),
        positive_target_id=torch.tensor([0, 0], dtype=torch.long),
        crop_kinds=("gt_control", "gt_control"),
        fiber_paths=("fiber.json", "fiber.json"),
        direction_kinds=("gt_jitter", "gt_jitter"),
    )
    batch.target_fw_xyz[:, 0] = 1.0

    writer = FakeWriter()
    fiber_train._log_training_sample_visualization(
        writer, FakeModel(), batch, step=1
    )

    cos = writer.images["train_sample/cross/cos_emb_cp"][0]
    assert tuple(cos.shape) == (8, 16)
    assert float(cos[4, 4]) == pytest.approx(1.0, abs=1e-6)
    assert float(cos[4, 12]) == pytest.approx(1.0, abs=1e-6)

    other = writer.images["train_sample/cross/cos_emb_other_cp"][0]
    assert float(other[4, 4]) == pytest.approx(0.5, abs=1e-6)
    assert float(other[4, 12]) == pytest.approx(0.5, abs=1e-6)

    label_image = writer.images["train_sample/cross/labels"]
    assert label_image.dtype == torch.uint8
    assert bool((label_image == 127).any())

    principal_yx = writer.images["train_sample/principal_yx/cos_emb_cp"][0]
    assert tuple(principal_yx.shape) == (8, 16)
    assert float(principal_yx[2, 3]) == pytest.approx(1.0, abs=1e-6)
    assert float(principal_yx[4, 10]) == pytest.approx(1.0, abs=1e-6)

    principal_zx = writer.images["train_sample/principal_zx/cos_emb_cp"][0]
    assert float(principal_zx[1, 3]) == pytest.approx(1.0, abs=1e-6)
    assert float(principal_zx[5, 10]) == pytest.approx(1.0, abs=1e-6)

    principal_zy = writer.images["train_sample/principal_zy/cos_emb_cp"][0]
    assert float(principal_zy[1, 2]) == pytest.approx(1.0, abs=1e-6)
    assert float(principal_zy[5, 12]) == pytest.approx(1.0, abs=1e-6)


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


def _prefetch_zarr_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    fiber_path = tmp_path / "fiber.json"
    _write_fiber(fiber_path)
    manifest_path = tmp_path / "pred.lasagna.json"
    manifest_path.write_text("{}", encoding="utf-8")

    base_root = tmp_path / "base.ome.zarr"
    grad_root = tmp_path / "grad_mag.ome.zarr"
    normal_root = tmp_path / "normal.ome.zarr"
    base = zarr.open(
        str(base_root),
        mode="w",
        path="0",
        shape=(16, 16, 16),
        chunks=(8, 8, 8),
        dtype="u1",
        dimension_separator="/",
    )
    base[:] = np.arange(16 * 16 * 16, dtype=np.uint16).reshape(16, 16, 16) % 255
    setattr(base.store, "_url", "s3://example/base.ome.zarr")
    setattr(base.store, "_cache_dir", str(tmp_path / "cache" / "base"))
    grad = zarr.open(
        str(grad_root),
        mode="w",
        path="0",
        shape=(16, 16, 16),
        chunks=(8, 8, 8),
        dtype="u1",
        dimension_separator="/",
    )
    grad[:] = 1
    normals = zarr.open(
        str(normal_root),
        mode="w",
        path="0",
        shape=(2, 16, 16, 16),
        chunks=(1, 8, 8, 8),
        dtype="u1",
        dimension_separator="/",
    )
    normals[:] = 128

    class FakeLasagnaVolume:
        path = manifest_path
        base_shape_zyx = (16, 16, 16)
        source_to_base = 1.0

        def channel_group(self, channel_name: str):
            if channel_name == "grad_mag":
                return (
                    SimpleNamespace(
                        zarr_path="grad_mag.ome.zarr/0",
                        scaledown=0,
                        sd_fac=1.0,
                    ),
                    0,
                )
            if channel_name == "nx":
                return (
                    SimpleNamespace(
                        zarr_path="normal.ome.zarr/0",
                        scaledown=0,
                        sd_fac=1.0,
                    ),
                    0,
                )
            if channel_name == "ny":
                return (
                    SimpleNamespace(
                        zarr_path="normal.ome.zarr/0",
                        scaledown=0,
                        sd_fac=1.0,
                    ),
                    1,
                )
            raise KeyError(channel_name)

    monkeypatch.setattr(
        fiber_dataset,
        "_load_lasagna_volume",
        lambda path, config: FakeLasagnaVolume(),
    )

    def fake_open_zarr(path, *, scale=None, auth_json_path=None, config=None):
        if str(path) == "s3://example/base.ome.zarr" and int(scale) == 0:
            return base
        if str(path) == str(grad_root.resolve()) and int(scale) == 0:
            return grad
        if str(path) == str(normal_root.resolve()) and int(scale) == 0:
            return normals
        raise AssertionError((path, scale))

    monkeypatch.setattr(fiber_dataset, "_common_open_zarr", fake_open_zarr)
    return {
        "crop_size": [8, 8, 8],
        "batch_size": 2,
        "seed": 3,
        "image_normalization": "unit",
        "positive_direction_jitter_degrees": 0.0,
        "control_point_margin_voxels": 3,
        "random_negative_pool_size": 2,
        "random_valid_max_attempts": 64,
        "num_steps": 1,
        "log_every": 1,
        "tensorboard_enabled": False,
        "datasets": [
            {
                "base_volume_path": "s3://example/base.ome.zarr",
                "lasagna_manifest_path": str(manifest_path),
                "fiber_paths": [str(fiber_path)],
            }
        ],
    }


def test_prefetch_chunk_requests_use_zarr_chunk_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _prefetch_zarr_config(tmp_path, monkeypatch)
    builder = FiberTraceBatchBuilder(config)

    requests = builder.prefetch_chunk_requests_for_iteration(iteration=1)
    labels = {request.label for request in requests}

    assert labels == {"base"}
    assert requests
    assert all(request.key.startswith("0/") for request in requests)
    assert any(request.key.count("/") == 3 for request in requests)


def test_prefetch_mode_fetches_chunks_with_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    config = _prefetch_zarr_config(tmp_path, monkeypatch)

    result = fiber_train.run_prefetch(config, max_workers=2)
    output = capsys.readouterr().out

    assert result["unique_chunks"] > 0
    assert result["pending_chunks"] > 0
    assert result["cached_chunks"] == 0
    assert result["bytes"] > 0
    assert result["errors"] == 0
    assert "prefetch chunks:" in output
    assert "pending=" in output
    assert "eta=" in output
    assert "MiB/s" in output


def test_prefetch_mode_skips_already_cached_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    config = _prefetch_zarr_config(tmp_path, monkeypatch)
    builder = FiberTraceBatchBuilder(config)
    requests = fiber_train._dedupe_chunk_requests(
        builder.prefetch_chunk_requests_for_iteration(iteration=1)
    )
    assert requests
    for request in requests:
        cache_dir = Path(getattr(request.store, "_cache_dir"))
        cache_path = cache_dir / request.key
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"cached")

    result = fiber_train.run_prefetch(config, max_workers=2)
    output = capsys.readouterr().out

    assert result["unique_chunks"] == len(requests)
    assert result["cached_chunks"] == len(requests)
    assert result["pending_chunks"] == 0
    assert result["bytes"] == 0
    assert "pending=0" in output


def test_prefetch_mode_adds_test_batch_once(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, int]] = []

    class FakeStore:
        def __getitem__(self, key: str) -> bytes:
            return b"x"

    class FakeBuilder:
        def __init__(self, config: dict, **kwargs) -> None:
            self.name = "test" if config.get("is_test") else "train"

        def prefetch_chunk_requests_for_iteration(self, *, iteration: int):
            calls.append((self.name, int(iteration)))
            return [
                fiber_dataset.ZarrChunkRequest(
                    store=FakeStore(),
                    store_identity=self.name,
                    key=f"{self.name}/{iteration}",
                    label="base",
                )
            ]

    monkeypatch.setattr(fiber_train, "FiberTraceBatchBuilder", FakeBuilder)
    monkeypatch.setattr(
        fiber_train,
        "_make_test_config",
        lambda config: {"is_test": True},
    )

    result = fiber_train.run_prefetch({"num_steps": 5}, max_workers=1)

    assert calls == [
        ("train", 1),
        ("train", 2),
        ("train", 3),
        ("train", 4),
        ("train", 5),
        ("test", 1),
    ]
    assert result["unique_chunks"] == 6


def test_prefetch_mode_respects_sample_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, int]] = []

    class FakeStore:
        def __getitem__(self, key: str) -> bytes:
            return b"x"

    class FakeBuilder:
        def __init__(self, config: dict, **kwargs) -> None:
            self.name = "test" if config.get("is_test") else "train"
            self.sample_limit = config.get("sample_limit")

        def prefetch_chunk_requests_for_iteration(self, *, iteration: int):
            calls.append((self.name, int(iteration)))
            return [
                fiber_dataset.ZarrChunkRequest(
                    store=FakeStore(),
                    store_identity=self.name,
                    key=f"{self.name}/{iteration}",
                    label="base",
                )
            ]

    monkeypatch.setattr(fiber_train, "FiberTraceBatchBuilder", FakeBuilder)
    monkeypatch.setattr(
        fiber_train,
        "_make_test_config",
        lambda config: {"is_test": True, "sample_limit": config.get("sample_limit")},
    )

    result = fiber_train.run_prefetch({"num_steps": 5, "sample_limit": 2}, max_workers=1)

    assert calls == [
        ("train", 1),
        ("train", 2),
        ("test", 1),
    ]
    assert result["unique_chunks"] == 3


def test_training_writes_tensorboard_text_and_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    class FakeWriter:
        def __init__(self, log_dir: Path) -> None:
            self.log_dir = Path(log_dir)
            self.scalars: list[tuple[str, float, int]] = []
            self.texts: list[tuple[str, str, int]] = []
            self.images: list[tuple[str, tuple[int, ...], int]] = []
            self.closed = False

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars.append((tag, float(value), int(step)))

        def add_text(self, tag: str, text: str, step: int) -> None:
            self.texts.append((tag, str(text), int(step)))

        def add_image(self, tag: str, image: torch.Tensor, step: int) -> None:
            self.images.append((tag, tuple(image.shape), int(step)))

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

    config = _synthetic_config(tmp_path, batch_size=4, crop_size=(8, 8, 8))
    config.update(
        {
            "device": "cpu",
            "num_steps": 1,
            "log_every": 100,
            "sample_visualization_every": 1,
            "run_path": str(tmp_path / "runs"),
            "run_name": "unit run",
            "run_datestr": "20260102_030405",
            "debug_cache": True,
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
    output = capsys.readouterr().out

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
    } <= scalar_tags
    assert output.count("fiber_trace batch columns:") == 0
    assert output.count("fiber_trace step columns:") == 1
    assert output.count("   it      data     train") == 1
    assert "trn_tot" in output
    assert "tst_tot" in output
    assert "step=1 " not in output
    step_rows = [
        line
        for line in output.splitlines()
        if line.split()[:1] == ["1"] and len(line.split()) == 14
    ]
    assert len(step_rows) == 1
    assert {"test/total", "test/contrastive"} <= scalar_tags
    image_tags = {tag for tag, _, _ in writers[0].images}
    expected_image_tags = {
        f"{prefix}/{view}/{name}"
        for prefix in ("train_sample", "test_sample")
        for view in ("side", "top", "cross")
        for name in ("image", "labels", "cos_emb_cp", "cos_emb_other_cp")
    }
    expected_image_tags.update(
        {
            f"{prefix}/{view}/cos_emb_cp"
            for prefix in ("train_sample", "test_sample")
            for view in ("principal_yx", "principal_zx", "principal_zy")
        }
    )
    assert expected_image_tags <= image_tags
    assert all(
        shape == (1, 8, 16) and step == 1 for _, shape, step in writers[0].images
    )


def test_training_uses_fixed_test_samples_and_snapshots_on_test_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    original_make_test_config = fiber_train._make_test_config

    def marked_test_config(config: dict):
        test_config = original_make_test_config(config)
        assert test_config is not None
        test_config["is_test"] = True
        return test_config

    sample_calls: list[tuple[str, int | None]] = []
    original_sample_batch = FiberTraceBatchBuilder.sample_batch

    def recording_sample_batch(self, *args, **kwargs):
        sample_calls.append(
            (
                "test" if self.config.get("is_test") else "train",
                kwargs.get("iteration"),
            )
        )
        return original_sample_batch(self, *args, **kwargs)

    snapshot_calls: list[tuple[str, int, str]] = []
    original_save_snapshot = fiber_train._save_snapshot

    def recording_save_snapshot(path: Path, **kwargs) -> None:
        snapshot_calls.append(
            (Path(path).name, int(kwargs["step"]), str(kwargs["metric_name"]))
        )
        original_save_snapshot(path, **kwargs)

    monkeypatch.setattr(fiber_train, "_make_test_config", marked_test_config)
    monkeypatch.setattr(FiberTraceBatchBuilder, "sample_batch", recording_sample_batch)
    monkeypatch.setattr(fiber_train, "_save_snapshot", recording_save_snapshot)

    config = _synthetic_config(tmp_path, batch_size=4, crop_size=(8, 8, 8))
    config.update(
        {
            "device": "cpu",
            "num_steps": 2,
            "log_every": 99,
            "test_every": 1,
            "test_sample_count": 2,
            "test_start_iteration": 50,
            "sample_visualization_every": 0,
            "test_visualization_every": 0,
            "tensorboard_enabled": False,
            "run_path": str(tmp_path / "runs"),
            "run_name": "fixed test",
            "run_datestr": "20260102_030405",
            "_test_array_records": config["_array_records"],
            "model": {
                "input_channels": 1,
                "backbone_channels": 2,
                "embedding_dim": 4,
                "features_per_stage": [2],
                "head_channels": 4,
            },
            "loss": {"max_contrastive_samples": 128},
        }
    )

    fiber_train.run_training(config)

    assert sample_calls == [
        ("train", 1),
        ("test", 50),
        ("test", 51),
        ("train", 2),
        ("test", 50),
        ("test", 51),
    ]
    assert ("current.pt", 1, "test/total") in snapshot_calls
    assert ("current.pt", 2, "test/total") in snapshot_calls


def test_training_rejects_legacy_checkpoint_path(tmp_path: Path):
    config = _synthetic_config(tmp_path, batch_size=2, crop_size=(8, 8, 8))
    config["checkpoint_path"] = str(tmp_path / "model.pt")
    with pytest.raises(ValueError, match="checkpoint_path was replaced"):
        fiber_train.run_training(config)
