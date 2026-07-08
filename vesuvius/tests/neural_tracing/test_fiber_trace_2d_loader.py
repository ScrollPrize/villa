from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace_2d.loader import (
    FiberStrip2DLoader,
    ZarrChunkRequest,
    load_config,
    strip_z_offsets_from_count_step,
)
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import (
    build_planar_side_strip_patch_grid,
    build_side_strip_patch_grid,
)


def _write_fiber(path: Path, points: list[list[float]] | None = None) -> Path:
    points = points or [[0.0, 20.0, 20.0], [10.0, 20.0, 20.0], [20.0, 20.0, 20.0]]
    obj = {
        "type": "vc3d_fiber",
        "version": 1,
        "line_points": points,
        "control_points": points,
        "generation": 1,
    }
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _write_zarr(path: Path) -> Path:
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_dataset(
        "0",
        shape=(48, 48, 48),
        chunks=(16, 16, 16),
        dtype="float32",
        compressor=None,
    )
    z, y, x = np.indices(arr.shape, dtype=np.float32)
    arr[:] = z * 10000.0 + y * 100.0 + x
    return path


def _write_lasagna_manifest(tmp_path: Path) -> Path:
    normals_path = tmp_path / "normals.zarr"
    root = zarr.open_group(str(normals_path), mode="w")
    arr = root.create_dataset(
        "0",
        shape=(3, 48, 48, 48),
        chunks=(1, 16, 16, 16),
        dtype="uint8",
        compressor=None,
    )
    arr[0, :, :, :] = 1
    arr[1, :, :, :] = 128
    arr[2, :, :, :] = 255
    manifest = {
        "version": 2,
        "source_to_base": 1.0,
        "base_shape_zyx": [48, 48, 48],
        "grad_mag_encode_scale": 1000.0,
        "grad_mag_factor": 1.0,
        "umbilicus_json": "umbilicus.json",
        "groups": {
            "normals": {
                "zarr": "normals.zarr",
                "scaledown": 0,
                "channels": ["grad_mag", "nx", "ny"],
            }
        },
    }
    (tmp_path / "umbilicus.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "test.lasagna.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_config(tmp_path: Path, *, batch_size: int = 2) -> Path:
    fiber = _write_fiber(tmp_path / "fiber.json")
    volume = _write_zarr(tmp_path / "vol.zarr")
    manifest = _write_lasagna_manifest(tmp_path)
    config = {
        "batch_size": batch_size,
        "patch_shape_hw": [3, 3],
        "strip_z_offset_count": 3,
        "strip_z_offset_step": 1.0,
        "seed": 123,
        "datasets": [
            {
                "fiber_paths": [str(fiber)],
                "base_volume_path": str(volume),
                "base_volume_scale": 0,
                "lasagna_manifest_path": str(manifest),
            }
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_fiber_parser_reuse(tmp_path: Path) -> None:
    path = _write_fiber(tmp_path / "fiber.json")
    fiber = load_vc3d_fiber(path)
    assert fiber.control_points_xyz.shape == (3, 3)
    assert np.allclose(fiber.control_points_zyx[0], [20.0, 20.0, 0.0])


def test_side_strip_grid_matches_vc3d_simple_line_boundaries(tmp_path: Path) -> None:
    fiber = load_vc3d_fiber(_write_fiber(tmp_path / "fiber.json"))
    grid = build_side_strip_patch_grid(
        fiber,
        control_point_index=1,
        patch_shape_hw=(3, 3),
        strip_z_offset=0.0,
        sampled_normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )

    assert np.allclose(grid.coords_xyz[0, 1], [10.0, 20.0, 19.0])
    assert np.allclose(grid.coords_xyz[1, 1], [10.0, 20.0, 20.0])
    assert np.allclose(grid.coords_xyz[2, 1], [10.0, 20.0, 21.0])
    assert np.all(grid.valid_mask)


def test_strip_and_planar_pixel_spacing_is_in_base_voxels(tmp_path: Path) -> None:
    fiber = load_vc3d_fiber(_write_fiber(tmp_path / "fiber.json"))
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    strip = build_side_strip_patch_grid(
        fiber,
        control_point_index=1,
        patch_shape_hw=(3, 3),
        strip_z_offset=0.0,
        sampled_normal=normal,
        pixel_spacing_base=2.0,
    )
    planar = build_planar_side_strip_patch_grid(
        fiber,
        control_point_index=1,
        patch_shape_hw=(3, 3),
        strip_z_offset=0.0,
        sampled_normal=normal,
        pixel_spacing_base=2.0,
    )

    assert np.allclose(strip.coords_xyz[1, :, 0], [8.0, 10.0, 12.0])
    assert np.allclose(strip.coords_xyz[:, 1, 2], [18.0, 20.0, 22.0])
    assert np.allclose(planar.coords_xyz[1, :, 0], [8.0, 10.0, 12.0])
    assert np.allclose(planar.coords_xyz[:, 1, 2], [18.0, 20.0, 22.0])


def test_strip_z_offsets_generated_from_count_and_step() -> None:
    assert strip_z_offsets_from_count_step(16, 1.0) == tuple(float(v) for v in range(-7, 9))
    assert strip_z_offsets_from_count_step(3, 2.0) == (-2.0, 0.0, 2.0)


def test_config_rejects_literal_strip_z_offsets(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["strip_z_offsets"] = [-1, 0, 1]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="strip_z_offsets was removed"):
        load_config(config_path)


def test_config_and_loader_batch_shape(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, batch_size=2))
    loader = FiberStrip2DLoader(config)
    batch = loader.load_batch(0)

    assert batch.images.shape == (2, 3, 1, 3, 3)
    assert batch.coords_zyx.shape == (2, 3, 3, 3, 3)
    assert batch.valid_mask.shape == (2, 3, 3, 3)
    assert batch.strip_z_offsets.tolist() == [-1.0, 0.0, 1.0]
    assert batch.control_point_indices.shape == (2,)


def test_deterministic_sample_index_independent_of_batch_size(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    config_a = load_config(config_path)
    config_b = load_config(config_path)
    loader_a = FiberStrip2DLoader(config_a)
    loader_b = FiberStrip2DLoader(config_b)

    a = [loader_a.descriptor_for_sample_index(i)[2] for i in range(8)]
    b = [loader_b.descriptor_for_sample_index(i)[2] for i in range(8)]
    assert a == b


class _FakeStore(dict):
    _url = "s3://bucket/vol.zarr"
    _cache_dir = "/tmp/fake-cache"
    _NEGATIVE_MARKER_SUFFIX = ".__notfound__"


def test_prefetch_deduplicates_and_skips_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, batch_size=1))
    loader = FiberStrip2DLoader(config)
    store = _FakeStore({"0.0.0": b"abc", "0.0.1": b"def"})
    seen = [
        ZarrChunkRequest(store=store, store_identity="same", key="0.0.0"),
        ZarrChunkRequest(store=store, store_identity="same", key="0.0.0"),
        ZarrChunkRequest(store=store, store_identity="same", key="0.0.1"),
    ]
    monkeypatch.setattr(loader, "chunk_requests_for_samples", lambda start, count: seen)
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(store, "_cache_dir", str(cache_dir), raising=False)
    (cache_dir / "0.0.0").parent.mkdir(parents=True, exist_ok=True)
    (cache_dir / "0.0.0").write_bytes(b"cached")

    summary = loader.prefetch(0, 1, workers=1)

    assert summary["generated"] == 2
    assert summary["missing"] == 1
    assert summary["downloaded"] == 1
    assert summary["errors"] == 0
