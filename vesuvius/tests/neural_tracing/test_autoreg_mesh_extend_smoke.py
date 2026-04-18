from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz import (
    _crop_min_corner_for_points,
    _extract_prompt_window,
    _open_zarr_volume,
    _render_projection,
    _surface_grid_zyx,
    _window_ranges,
    build_extension_sample,
    choose_source_tifxyz,
    choose_growth_direction,
    extend_tifxyz_mesh,
    finalize_iteration_extension,
    grid_to_colored_mesh,
    infer_extension_windows_batched,
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


def test_build_extension_sample_has_required_fields() -> None:
    grid = _make_surface_grid(6, 6)
    prompt_grid = grid[:, -3:, :]
    sample = build_extension_sample(
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

    assert sample["volume"].shape == (1, 128, 128, 128)
    assert set(sample["prompt_tokens"].keys()) == {"coarse_ids", "offset_bins", "xyz", "strip_positions", "strip_coords", "valid_mask"}
    assert sample["prompt_tokens"]["coarse_ids"].ndim == 1
    assert sample["target_grid_shape"].shape == (2,)
    assert sample["direction"] == "left"


class _FakeBatchModel:
    def __init__(self) -> None:
        self.offset_num_bins = (4, 4, 4)
        self.coarse_prediction_mode = "joint_pointer"
        self.coarse_grid_shape = (2, 2, 2)
        self._param = np.zeros((), dtype=np.float32)

    def encode_conditioning(self, volume, vol_tokens=None):
        del vol_tokens
        batch_size = int(volume.shape[0])
        device = volume.device
        return {
            "memory_tokens": torch.zeros((batch_size, 8, 4), device=device, dtype=torch.float32),
            "memory_patch_centers": torch.zeros((batch_size, 8, 3), device=device, dtype=torch.float32),
        }

    def forward_from_encoded(self, batch, *, memory_tokens, memory_patch_centers):
        del memory_tokens, memory_patch_centers
        device = batch["target_coarse_ids"].device
        batch_size, target_len = batch["target_coarse_ids"].shape
        coarse_logits = torch.full((batch_size, target_len, 8), -8.0, device=device)
        coarse_logits[:, :, 3] = 8.0
        offset_logits = torch.full((batch_size, target_len, 3, 4), -6.0, device=device)
        offset_logits[:, :, :, 1] = 6.0
        pred_coarse_ids = torch.full((batch_size, target_len), 3, dtype=torch.long, device=device)
        pred_offset_bins = torch.ones((batch_size, target_len, 3), dtype=torch.long, device=device)
        pred_xyz = self.decode_local_xyz(pred_coarse_ids, pred_offset_bins)
        pred_refine_residual = torch.zeros((batch_size, target_len, 3), dtype=torch.float32, device=device)
        return {
            "coarse_logits": coarse_logits,
            "coarse_axis_logits": None,
            "offset_logits": offset_logits,
            "stop_logits": torch.full((batch_size, target_len), -8.0, dtype=torch.float32, device=device),
            "pred_coarse_ids": pred_coarse_ids,
            "pred_coarse_axis_ids": {
                "z": torch.zeros((batch_size, target_len), dtype=torch.long, device=device),
                "y": torch.ones((batch_size, target_len), dtype=torch.long, device=device),
                "x": torch.ones((batch_size, target_len), dtype=torch.long, device=device),
            },
            "pred_offset_bins": pred_offset_bins,
            "pred_refine_residual": pred_refine_residual,
            "pred_xyz": pred_xyz,
            "pred_xyz_soft": pred_xyz,
            "pred_xyz_refined": pred_xyz,
            "coarse_grid_shape": self.coarse_grid_shape,
            "coarse_prediction_mode": self.coarse_prediction_mode,
        }

    def decode_local_xyz(self, coarse_ids: torch.Tensor, offset_bins: torch.Tensor) -> torch.Tensor:
        coarse = coarse_ids.to(torch.long)
        gyx = self.coarse_grid_shape[1] * self.coarse_grid_shape[2]
        z = coarse // gyx
        rem = coarse % gyx
        y = rem // self.coarse_grid_shape[2]
        x = rem % self.coarse_grid_shape[2]
        starts = torch.stack([z, y, x], dim=-1).to(torch.float32) * 8.0
        widths = torch.tensor([2.0, 2.0, 2.0], device=coarse.device, dtype=torch.float32)
        return starts + (offset_bins.to(torch.float32) + 0.5) * widths.view(1, 1, 3)


def _make_window_payloads() -> list:
    grid = _make_surface_grid(8, 10)
    prompt_a = grid[:4, -3:, :]
    prompt_b = grid[2:6, -3:, :]
    sample_a = build_extension_sample(
        prompt_grid_world=prompt_a,
        direction="left",
        min_corner=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        crop_size=(128, 128, 128),
        patch_size=(8, 8, 8),
        offset_num_bins=(4, 4, 4),
        frontier_band_width=3,
        predict_strips=2,
        volume_crop=np.zeros((128, 128, 128), dtype=np.float32),
        wrap_metadata={"segment_uuid": "synthetic_a"},
    )
    sample_b = build_extension_sample(
        prompt_grid_world=prompt_b,
        direction="left",
        min_corner=np.array([4.0, 4.0, 4.0], dtype=np.float32),
        crop_size=(128, 128, 128),
        patch_size=(8, 8, 8),
        offset_num_bins=(4, 4, 4),
        frontier_band_width=3,
        predict_strips=2,
        volume_crop=np.zeros((128, 128, 128), dtype=np.float32),
        wrap_metadata={"segment_uuid": "synthetic_b"},
    )
    from vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz import ExtensionWindow, ExtensionWindowPayload
    return [
        ExtensionWindowPayload(ExtensionWindow(0, 4), sample_a, "left", (4, 2), 4, 2, 3, 2),
        ExtensionWindowPayload(ExtensionWindow(2, 6), sample_b, "left", (4, 2), 4, 2, 3, 2),
    ]


def test_batched_extension_inference_matches_serial() -> None:
    model = _FakeBatchModel()
    payloads = _make_window_payloads()

    serial_results, _, _ = infer_extension_windows_batched(model, payloads, window_batch_size=1, device=torch.device("cpu"))
    batched_results, _, peak_batch_size = infer_extension_windows_batched(model, payloads, window_batch_size=2, device=torch.device("cpu"))

    assert peak_batch_size == 2
    assert len(serial_results) == len(batched_results) == 2
    for serial, batched in zip(serial_results, batched_results, strict=True):
        np.testing.assert_allclose(serial["continuation_grid_world"], batched["continuation_grid_world"])
        assert serial["window"] == batched["window"]


def test_two_iteration_rollout_appends_geometry_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        return _FakeBatchModel(), {"patch_size": [8, 8, 8], "offset_num_bins": [4, 4, 4], "frontier_band_width": 4}

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz.read_tifxyz", _fake_read_tifxyz)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz._open_zarr_volume", _fake_open_volume)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz._load_autoreg_model", _fake_load_model)

    result = extend_tifxyz_mesh(
        tifxyz_path="dummy",
        volume_uri="s3://dummy",
        dino_backbone="backbone",
        autoreg_checkpoint="ckpt",
        out_dir=tmp_path,
        device="cpu",
        prompt_strips=3,
        predict_strips_per_iter=2,
        window_strip_length=4,
        window_overlap=2,
        window_batch_size=2,
        max_extension_iters=2,
    )

    assert len(result["iteration_stats"]) == 2
    assert result["iteration_stats"][0]["valid_new_vertices"] > 0
    assert result["iteration_stats"][1]["valid_new_vertices"] > 0
    assert result["cumulative_predicted_vertex_count"] > result["iteration_stats"][0]["valid_new_vertices"]
    assert result["iteration_stats"][1]["peak_batch_size_used"] == 2


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

    def _fake_load_model(*, dino_backbone, autoreg_checkpoint, device):
        del dino_backbone, autoreg_checkpoint, device
        return _FakeBatchModel(), {"patch_size": [8, 8, 8], "offset_num_bins": [4, 4, 4], "frontier_band_width": 4}

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz.read_tifxyz", _fake_read_tifxyz)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz._open_zarr_volume", _fake_open_volume)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz._load_autoreg_model", _fake_load_model)

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
