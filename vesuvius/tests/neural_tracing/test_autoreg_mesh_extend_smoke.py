from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz import (
    _crop_min_corner_for_points,
    _extract_prompt_window,
    _open_zarr_volume,
    _render_projection,
    _surface_grid_zyx,
    _window_ranges,
    build_extension_batch,
    choose_source_tifxyz,
    choose_growth_direction,
    extend_tifxyz_mesh,
    finalize_iteration_extension,
    grid_to_colored_mesh,
    merge_window_prediction,
    write_colored_ply,
)
from vesuvius.tifxyz import Tifxyz


def _make_surface_grid(rows: int = 10, cols: int = 12) -> np.ndarray:
    row_axis = np.arange(rows, dtype=np.float32)[:, None]
    col_axis = np.arange(cols, dtype=np.float32)[None, :]
    row_grid = np.broadcast_to(row_axis, (rows, cols))
    col_grid = np.broadcast_to(col_axis, (rows, cols))
    z = 16.0 + 0.1 * row_grid
    y = 32.0 + 2.0 * row_grid
    x = 48.0 + 2.0 * col_grid
    return np.stack([z, y, x], axis=-1).astype(np.float32)


def _make_surface(rows: int = 10, cols: int = 12) -> Tifxyz:
    grid = _make_surface_grid(rows, cols)
    return Tifxyz(
        _x=grid[..., 2].copy(),
        _y=grid[..., 1].copy(),
        _z=grid[..., 0].copy(),
        uuid="synthetic_surface",
        _scale=(1.0, 1.0),
        _mask=np.ones((rows, cols), dtype=bool),
        resolution="stored",
    )


def test_window_ranges_are_deterministic() -> None:
    windows = _window_ranges(20, 8, 2)
    assert [(window.start, window.end) for window in windows] == [(0, 8), (6, 14), (12, 20)]


def test_crop_min_corner_rejects_oversize_envelope() -> None:
    points = np.array([[0.0, 0.0, 0.0], [256.0, 32.0, 32.0]], dtype=np.float32)
    assert _crop_min_corner_for_points(points, (128, 128, 128)) is None


def test_choose_growth_direction_prefers_valid_boundary() -> None:
    grid = _make_surface_grid()
    grid[:, :3, :] = np.nan
    direction = choose_growth_direction(grid, prompt_strips=3, predict_strips=2, crop_size=(128, 128, 128))
    assert direction == "left"


def test_choose_source_tifxyz_picks_best_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("a", "b", "c"):
        path = tmp_path / name
        path.mkdir()
        (path / "meta.json").write_text("{}")

    surface_map = {
        str(tmp_path / "a"): _make_surface(8, 8),
        str(tmp_path / "b"): _make_surface(20, 20),
        str(tmp_path / "c"): _make_surface(6, 6),
    }

    def _fake_read_tifxyz(path, load_mask=True, validate=True):
        del load_mask, validate
        return surface_map[str(path)]

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz.read_tifxyz", _fake_read_tifxyz)
    selected = choose_source_tifxyz(tmp_path, prompt_strips=3, predict_strips=2, crop_size=(128, 128, 128), limit=8)
    assert selected == tmp_path / "b"


def test_merge_window_prediction_averages_overlaps() -> None:
    sums = np.zeros((4, 2, 3), dtype=np.float64)
    counts = np.zeros((4, 2), dtype=np.int32)
    pred_a = np.ones((4, 2, 3), dtype=np.float32)
    pred_b = np.full((4, 2, 3), 3.0, dtype=np.float32)
    window = SimpleNamespace(start=0, end=4)

    merge_window_prediction(sums=sums, counts=counts, pred_grid_world=pred_a, direction="left", window=window)
    merge_window_prediction(sums=sums, counts=counts, pred_grid_world=pred_b, direction="left", window=window)
    extension, provenance = finalize_iteration_extension(sums=sums, counts=counts, direction="left")

    assert np.allclose(extension, 2.0)
    assert np.all(provenance[:, 0] == 2)
    assert np.all(provenance[:, 1] == 1)


def test_build_extension_batch_has_required_fields() -> None:
    grid = _make_surface_grid(6, 6)
    prompt_grid = grid[:, -3:, :]
    batch = build_extension_batch(
        prompt_grid_world=prompt_grid,
        direction="left",
        min_corner=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        crop_size=(128, 128, 128),
        patch_size=(8, 8, 8),
        offset_num_bins=(16, 16, 16),
        frontier_band_width=4,
        predict_strips=4,
        volume_crop=np.zeros((128, 128, 128), dtype=np.float32),
        wrap_metadata={"segment_uuid": "synthetic"},
    )

    assert batch["volume"].shape == (1, 1, 128, 128, 128)
    assert set(batch["prompt_tokens"].keys()) == {"coarse_ids", "offset_bins", "xyz", "strip_positions", "strip_coords", "mask", "valid_mask"}
    assert batch["prompt_tokens"]["coarse_ids"].ndim == 2
    assert batch["target_grid_shape"].shape == (1, 2)
    assert batch["direction"] == ["left"]


def test_surface_grid_conversion_marks_invalid_as_nan() -> None:
    surface = _make_surface()
    surface._mask[0, 0] = False
    grid = _surface_grid_zyx(surface)
    assert np.isnan(grid[0, 0]).all()
    assert np.isfinite(grid[1, 1]).all()


def test_grid_to_colored_mesh_and_ply_export(tmp_path: Path) -> None:
    grid = _make_surface_grid(3, 4)
    provenance = np.zeros((3, 4), dtype=np.uint8)
    provenance[:, -1] = 1
    provenance[:, -2] = 2
    vertices, faces, colors = grid_to_colored_mesh(grid, provenance)
    ply_path = write_colored_ply(tmp_path / "mesh.ply", vertices, faces, colors)

    assert vertices.shape[0] == 12
    assert faces.shape[0] == 12
    assert any((colors == np.array([255, 140, 0], dtype=np.uint8)).all(axis=1))
    assert any((colors == np.array([255, 0, 255], dtype=np.uint8)).all(axis=1))
    assert ply_path.exists()


def test_render_projection_returns_nonempty_canvas() -> None:
    grid = _make_surface_grid(4, 5)
    provenance = np.zeros((4, 5), dtype=np.uint8)
    provenance[:, -2:] = 1
    image = _render_projection(grid, provenance, plane="xy", size=256)
    assert image.shape == (256, 256, 3)
    assert int(image.sum()) > 0


def test_open_zarr_volume_routes_s3_via_s3fs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class _FakeS3Map:
        def __init__(self, root, s3, check):
            calls["root"] = root
            calls["check"] = check
            self.root = root

    class _FakeS3FS:
        def __init__(self, anon):
            calls["anon"] = anon

    class _FakeGroup(dict):
        pass

    def _fake_open(store=None, mode=None):
        calls["mode"] = mode
        return _FakeGroup({"0": np.zeros((8, 8, 8), dtype=np.float32)})

    fake_s3fs = SimpleNamespace(S3FileSystem=_FakeS3FS, S3Map=_FakeS3Map)
    monkeypatch.setitem(sys.modules, "s3fs", fake_s3fs)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz.zarr.open", _fake_open)

    volume = _open_zarr_volume("s3://bucket/path/to/volume.zarr")

    assert calls["anon"] is True
    assert calls["root"] == "bucket/path/to/volume.zarr"
    assert calls["check"] is False
    assert calls["mode"] == "r"
    assert isinstance(volume, np.ndarray)


def test_extend_tifxyz_mesh_synthetic_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    surface = _make_surface(8, 10)
    volume = np.zeros((256, 256, 256), dtype=np.float32)

    def _fake_read_tifxyz(path, load_mask=True, validate=True):
        del path, load_mask, validate
        return surface

    def _fake_open_volume(uri):
        del uri
        return volume

    class _FakeModel:
        def eval(self):
            return self

    def _fake_load_model(*, dino_backbone, autoreg_checkpoint, device):
        del dino_backbone, autoreg_checkpoint, device
        return _FakeModel(), {"patch_size": [8, 8, 8], "offset_num_bins": [16, 16, 16], "frontier_band_width": 4}

    def _fake_infer(model, batch, *, max_steps=None, stop_probability_threshold=None, greedy=True):
        del model, max_steps, stop_probability_threshold, greedy
        direction = batch["direction"][0]
        cond = batch["conditioning_grid_local"][0].numpy()
        if direction == "left":
            boundary = cond[:, -1, :]
            previous = cond[:, -2, :]
            step = boundary - previous
            continuation = np.stack([boundary + float(i + 1) * step for i in range(8)], axis=1)
        elif direction == "right":
            boundary = cond[:, 0, :]
            previous = cond[:, 1, :]
            step = boundary - previous
            continuation = np.stack([boundary + float(i + 1) * step for i in range(8)], axis=1)
        elif direction == "up":
            boundary = cond[-1, :, :]
            previous = cond[-2, :, :]
            step = boundary - previous
            continuation = np.stack([boundary + float(i + 1) * step for i in range(8)], axis=0)
        else:
            boundary = cond[0, :, :]
            previous = cond[1, :, :]
            step = boundary - previous
            continuation = np.stack([boundary + float(i + 1) * step for i in range(8)], axis=0)
        min_corner = batch["min_corner"][0].numpy()
        continuation_world = continuation + min_corner.reshape(1, 1, 3)
        return {"continuation_grid_world": continuation_world}

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz.read_tifxyz", _fake_read_tifxyz)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz._open_zarr_volume", _fake_open_volume)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz._load_autoreg_model", _fake_load_model)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz.infer_autoreg_mesh", _fake_infer)

    result = extend_tifxyz_mesh(
        tifxyz_path="dummy",
        volume_uri="s3://dummy",
        dino_backbone="backbone",
        autoreg_checkpoint="ckpt",
        out_dir=tmp_path,
        device="cpu",
        prompt_strips=3,
        predict_strips_per_iter=8,
        window_strip_length=4,
        window_overlap=2,
        max_extension_iters=1,
    )

    assert result["predicted_vertex_count"] > 0
    assert Path(result["mesh_path"]).exists()
    assert Path(result["tifxyz_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert result["iteration_stats"][0]["fitted_window_count"] >= 1
    assert result["iteration_stats"][0]["skipped_window_count"] >= 0
    for preview in result["preview_paths"]:
        assert Path(preview).exists()
