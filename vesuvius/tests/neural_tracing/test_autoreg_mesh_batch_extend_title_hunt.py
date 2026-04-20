from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt import (
    SYNC_EXCLUDE_PATTERNS,
    TITLE_HUNT_PREDICT_STRIPS_PER_ITER,
    TITLE_HUNT_PREFLIGHT_MIN_FITTED_PLANS,
    TITLE_HUNT_PROMPT_STRIPS,
    TITLE_HUNT_WINDOW_OVERLAP,
    TITLE_HUNT_WINDOW_STRIP_LENGTH,
    _aws_sync_exclude_args,
    _compact_extension_summary,
    _list_title_hunt_prefixes,
    _new_surface_state,
    _process_surface,
    _run_direction_to_exhaustion,
    _select_dry_run_surfaces,
    _surface_paths,
    _timestamp_suffix,
    _title_hunt_extension_preset,
    _write_json,
    run_batch_extend_title_hunt,
    run_title_hunt_planner_preflight,
    _scale_stored_tifxyz,
)
from vesuvius.tifxyz import Tifxyz, read_tifxyz, write_tifxyz


def _make_surface(rows: int = 4, cols: int = 5) -> Tifxyz:
    row_axis = np.arange(rows, dtype=np.float32)[:, None]
    col_axis = np.arange(cols, dtype=np.float32)[None, :]
    row_grid = np.broadcast_to(row_axis, (rows, cols))
    col_grid = np.broadcast_to(col_axis, (rows, cols))
    z = 10.0 + row_grid
    y = 20.0 + 2.0 * row_grid
    x = 30.0 + 3.0 * col_grid
    mask = np.ones((rows, cols), dtype=bool)
    mask[0, 0] = False
    return Tifxyz(
        _x=x.copy(),
        _y=y.copy(),
        _z=z.copy(),
        uuid="synthetic_surface",
        _scale=(0.5, 0.5),
        bbox=(30.0, 20.0, 10.0, 42.0, 26.0, 13.0),
        _mask=mask,
        resolution="stored",
    )


def test_timestamp_suffix_format() -> None:
    suffix = _timestamp_suffix()
    assert suffix.endswith("Z")
    assert len(suffix) == 16


def test_surface_paths_include_timestamp_suffix() -> None:
    paths = _surface_paths(
        relative_surface_id="title_hunt_rd1/surface_a",
        timestamp_suffix="20260419T120000Z",
        local_output_root=Path("/tmp/out"),
        state_root=Path("/tmp/state"),
        output_s3_uri="s3://bucket/root",
    )

    assert paths.output_dir == Path("/tmp/out/title_hunt_rd1/surface_a__20260419T120000Z")
    assert paths.s3_surface_prefix == "s3://bucket/root/title_hunt_rd1/surface_a__20260419T120000Z"
    assert paths.s3_final_tifxyz_uri.endswith("/final_tifxyz")


def test_scale_stored_tifxyz_multiplies_coordinates_and_bbox(tmp_path: Path) -> None:
    source_surface = _make_surface()
    source_dir = write_tifxyz(tmp_path / "source", source_surface, overwrite=True)

    scaled_dir, resample_factor = _scale_stored_tifxyz(source_dir, tmp_path / "scaled", coordinate_scale_factor=4.0)
    scaled_surface = read_tifxyz(scaled_dir, load_mask=True, validate=True).use_stored_resolution()

    valid_mask = source_surface._mask
    np.testing.assert_allclose(scaled_surface._x[valid_mask], source_surface._x[valid_mask] * 4.0)
    np.testing.assert_allclose(scaled_surface._y[valid_mask], source_surface._y[valid_mask] * 4.0)
    np.testing.assert_allclose(scaled_surface._z[valid_mask], source_surface._z[valid_mask] * 4.0)
    assert resample_factor == 1
    assert scaled_surface.get_scale_tuple() == source_surface.get_scale_tuple()
    assert scaled_surface.bbox == tuple(value * 4.0 for value in source_surface.bbox)


def test_select_dry_run_surfaces_uses_distinct_prefixes() -> None:
    surfaces = [
        {"prefix_name": "title_hunt_rd1", "relative_surface_id": "title_hunt_rd1/a", "local_path": "/tmp/a", "uuid": "a"},
        {"prefix_name": "title_hunt_rd1", "relative_surface_id": "title_hunt_rd1/b", "local_path": "/tmp/b", "uuid": "b"},
        {"prefix_name": "title_hunt_rd2", "relative_surface_id": "title_hunt_rd2/c", "local_path": "/tmp/c", "uuid": "c"},
    ]

    selected = _select_dry_run_surfaces(surfaces, count=2)

    assert [item["relative_surface_id"] for item in selected] == ["title_hunt_rd1/a", "title_hunt_rd2/c"]


def test_list_title_hunt_prefixes_parses_aws_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "z")

    class _Result:
        stdout = "                           PRE title_hunt_rd2/\n                           PRE title_hunt_rd1/\n                           PRE other/\n"

    monkeypatch.setattr(
        "vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt._run_cmd",
        lambda *args, **kwargs: _Result(),
    )

    prefixes = _list_title_hunt_prefixes("s3://bucket/root")

    assert prefixes == ["title_hunt_rd1", "title_hunt_rd2"]


def test_aws_sync_exclude_args_cover_layers_and_zarr() -> None:
    args = _aws_sync_exclude_args()

    for pattern in SYNC_EXCLUDE_PATTERNS:
        assert ["--exclude", pattern] == args[args.index("--exclude"):args.index("--exclude") + 2] or pattern in args
    assert "layers/*" in args
    assert "*.zarr/*" in args


def test_title_hunt_extension_preset_is_conservative() -> None:
    preset = _title_hunt_extension_preset(
        window_batch_size=256,
        show_progress=False,
        distributed_infer=True,
        device="cpu",
        attention_scaling_mode="legacy_double_scaled",
    )

    assert preset["prompt_strips"] == TITLE_HUNT_PROMPT_STRIPS == 4
    assert preset["predict_strips_per_iter"] == TITLE_HUNT_PREDICT_STRIPS_PER_ITER == 1
    assert preset["window_strip_length"] == TITLE_HUNT_WINDOW_STRIP_LENGTH == 16
    assert preset["window_overlap"] == TITLE_HUNT_WINDOW_OVERLAP == 8
    assert preset["window_batch_size"] == 256
    assert preset["distributed_infer"] is True
    assert preset["device"] == "cpu"
    assert preset["attention_scaling_mode"] == "legacy_double_scaled"


def test_run_direction_to_exhaustion_loops_until_stop_reason_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    surface = _make_surface()
    initial_input = write_tifxyz(tmp_path / "initial", surface, overwrite=True)
    call_counter = {"count": 0}

    def _fake_extend(**kwargs):
        call_idx = call_counter["count"]
        call_counter["count"] += 1
        tifxyz_dir = write_tifxyz(Path(kwargs["out_dir"]) / "merged", surface, overwrite=True)
        assert kwargs["prompt_strips"] == 4
        assert kwargs["predict_strips_per_iter"] == 1
        assert kwargs["window_strip_length"] == 16
        assert kwargs["window_overlap"] == 8
        stop_reason = "max_extension_iters" if call_idx == 0 else "zero_growth_iteration"
        return {
            "tifxyz_path": str(tifxyz_dir),
            "summary_path": str(Path(kwargs["out_dir"]) / "summary.json"),
            "direction": kwargs["grow_direction"],
            "predicted_vertex_count": 10,
            "cumulative_predicted_vertex_count": 10,
            "final_predicted_nonseam_vertex_count": 8,
            "final_seam_vertex_count": 2,
            "new_band_frontier_coverage_fraction": 1.0,
            "new_band_cell_coverage_fraction": 1.0,
            "new_band_max_gap": 0,
            "new_band_gap_spans": [],
            "first_uncovered_frontier_index": None,
            "iterations_completed": 1,
            "stop_reason": stop_reason,
            "window_batch_size": kwargs["window_batch_size"],
            "encode_decode_ms_per_fitted_window": 1.0,
            "crop_cache_hits": 0,
            "crop_cache_misses": 1,
            "total_wall_ms": 1.0,
            "fast_infer_enabled": True,
            "compile_infer_requested": False,
            "compile_infer_actual": False,
            "amp_dtype": "bf16",
        }

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt.extend_tifxyz_mesh", _fake_extend)

    result = _run_direction_to_exhaustion(
        input_tifxyz_path=Path(initial_input),
        output_dir=tmp_path / "stage",
        direction="up",
        volume_uri="s3://volume",
        dino_backbone="backbone",
        autoreg_checkpoint="ckpt",
        window_batch_size=256,
        max_extension_iters_per_call=16,
        show_progress=False,
    )

    assert call_counter["count"] == 2
    assert Path(result.final_tifxyz_path).exists()
    assert result.stage_summary["call_count"] == 2
    assert result.stage_summary["final"]["stop_reason"] == "zero_growth_iteration"
    assert result.stage_summary["extension_preset"]["prompt_strips"] == 4


def test_run_direction_to_exhaustion_stops_at_dry_run_call_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    surface = _make_surface()
    initial_input = write_tifxyz(tmp_path / "initial_cap", surface, overwrite=True)
    call_counter = {"count": 0}

    def _fake_extend(**kwargs):
        call_counter["count"] += 1
        tifxyz_dir = write_tifxyz(Path(kwargs["out_dir"]) / "merged", surface, overwrite=True)
        return {
            "tifxyz_path": str(tifxyz_dir),
            "summary_path": str(Path(kwargs["out_dir"]) / "summary.json"),
            "direction": kwargs["grow_direction"],
            "predicted_vertex_count": 10,
            "cumulative_predicted_vertex_count": 10,
            "final_predicted_nonseam_vertex_count": 8,
            "final_seam_vertex_count": 2,
            "new_band_frontier_coverage_fraction": 1.0,
            "new_band_cell_coverage_fraction": 1.0,
            "new_band_max_gap": 0,
            "new_band_gap_spans": [],
            "first_uncovered_frontier_index": None,
            "iterations_completed": 1,
            "stop_reason": "max_extension_iters",
            "window_batch_size": kwargs["window_batch_size"],
            "encode_decode_ms_per_fitted_window": 1.0,
            "crop_cache_hits": 0,
            "crop_cache_misses": 1,
            "total_wall_ms": 1.0,
            "fast_infer_enabled": True,
            "compile_infer_requested": False,
            "compile_infer_actual": False,
            "amp_dtype": "bf16",
        }

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt.extend_tifxyz_mesh", _fake_extend)

    result = _run_direction_to_exhaustion(
        input_tifxyz_path=Path(initial_input),
        output_dir=tmp_path / "stage_cap",
        direction="up",
        volume_uri="s3://volume",
        dino_backbone="backbone",
        autoreg_checkpoint="ckpt",
        window_batch_size=256,
        max_extension_iters_per_call=16,
        show_progress=False,
        max_calls=2,
    )

    assert call_counter["count"] == 2
    assert result.stage_summary["stage_stop_reason"] == "max_calls_reached"


def test_run_direction_to_exhaustion_rejects_bad_dry_run_geometry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    surface = _make_surface()
    initial_input = write_tifxyz(tmp_path / "initial_bad", surface, overwrite=True)

    def _fake_extend(**kwargs):
        tifxyz_dir = write_tifxyz(Path(kwargs["out_dir"]) / "merged", surface, overwrite=True)
        return {
            "tifxyz_path": str(tifxyz_dir),
            "summary_path": str(Path(kwargs["out_dir"]) / "summary.json"),
            "direction": kwargs["grow_direction"],
            "predicted_vertex_count": 12,
            "cumulative_predicted_vertex_count": 12,
            "final_predicted_nonseam_vertex_count": 10,
            "final_seam_vertex_count": 2,
            "new_band_frontier_coverage_fraction": 0.25,
            "new_band_cell_coverage_fraction": 0.25,
            "new_band_max_gap": 120,
            "new_band_gap_spans": [(0, 120)],
            "first_uncovered_frontier_index": 0,
            "iterations_completed": 3,
            "stop_reason": "max_extension_iters",
            "window_batch_size": kwargs["window_batch_size"],
            "encode_decode_ms_per_fitted_window": 1.0,
            "crop_cache_hits": 0,
            "crop_cache_misses": 1,
            "total_wall_ms": 1.0,
            "fast_infer_enabled": True,
            "compile_infer_requested": False,
            "compile_infer_actual": False,
            "amp_dtype": "bf16",
        }

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt.extend_tifxyz_mesh", _fake_extend)

    with pytest.raises(RuntimeError, match="low frontier coverage|excessive gap"):
        _run_direction_to_exhaustion(
            input_tifxyz_path=Path(initial_input),
            output_dir=tmp_path / "stage_bad",
            direction="up",
            volume_uri="s3://volume",
            dino_backbone="backbone",
            autoreg_checkpoint="ckpt",
            window_batch_size=256,
            max_extension_iters_per_call=3,
            show_progress=False,
            max_calls=1,
            enforce_dry_run_quality_gate=True,
        )


def test_process_surface_resumes_and_uploads_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_surface = _make_surface()
    source_dir = write_tifxyz(tmp_path / "source", source_surface, overwrite=True)
    state_root = tmp_path / "state"
    manifest_jsonl_path = state_root / "manifest.jsonl"
    uploads: list[str] = []

    def _fake_run_direction(**kwargs):
        direction = kwargs["direction"]
        out_dir = Path(kwargs["output_dir"]) / "merged"
        write_tifxyz(out_dir, source_surface, overwrite=True)
        summary = {
            "direction": direction,
            "call_count": 1,
            "final_tifxyz_path": str(out_dir),
            "calls": [{"stop_reason": "zero_growth_iteration"}],
            "final": {
                "direction": direction,
                "predicted_vertex_count": 12,
                "cumulative_predicted_vertex_count": 12,
                "final_predicted_nonseam_vertex_count": 10,
                "final_seam_vertex_count": 2,
                "new_band_frontier_coverage_fraction": 1.0,
                "new_band_cell_coverage_fraction": 1.0,
                "new_band_max_gap": 0,
                "new_band_gap_spans": [],
                "first_uncovered_frontier_index": None,
                "iterations_completed": 1,
                "stop_reason": "zero_growth_iteration",
                "window_batch_size": 256,
                "encode_decode_ms_per_fitted_window": 1.0,
                "crop_cache_hits": 0,
                "crop_cache_misses": 1,
                "total_wall_ms": 1.0,
                "fast_infer_enabled": True,
                "compile_infer_requested": False,
                "compile_infer_actual": False,
                "amp_dtype": "bf16",
            },
        }
        return type("FakeDirectionRun", (), {"final_tifxyz_path": out_dir, "stage_summary": summary})()

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt._run_direction_to_exhaustion", _fake_run_direction)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt._upload_surface_output", lambda paths: uploads.append(paths.s3_surface_prefix))

    state = _process_surface(
        surface={
            "prefix_name": "title_hunt_rd1",
            "relative_surface_id": "title_hunt_rd1/surface_a",
            "local_path": str(source_dir),
            "uuid": "surface_a",
        },
        source_s3_uri="s3://bucket/source",
        output_s3_uri="s3://bucket/out",
        volume_uri="s3://volume",
        dino_backbone="backbone",
        autoreg_checkpoint="ckpt",
        local_output_root=tmp_path / "outputs",
        state_root=state_root,
        manifest_jsonl_path=manifest_jsonl_path,
        coordinate_scale_factor=4.0,
        max_extension_iters_per_call=16,
        window_batch_size=256,
        show_progress=False,
        dry_run_mode=False,
        dry_run_max_calls_per_direction=2,
        distributed_infer=False,
    )

    assert state["status"] == "uploaded"
    assert uploads
    manifest_path = Path(state["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["coordinate_scale_factor"] == pytest.approx(4.0)
    assert manifest["direction_summaries"]["up"]["predicted_vertex_count"] == 12
    assert manifest["direction_summaries"]["down"]["predicted_vertex_count"] == 12
    assert manifest["cumulative_predicted_vertex_count"] == 0
    assert manifest["extension_preset"]["prompt_strips"] == 4


def test_process_surface_preserves_direction_order_and_cumulative_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_surface = _make_surface(rows=4, cols=5)
    grown_surface = _make_surface(rows=4, cols=6)
    source_dir = write_tifxyz(tmp_path / "source_order", source_surface, overwrite=True)
    state_root = tmp_path / "state_order"
    manifest_jsonl_path = state_root / "manifest.jsonl"
    call_order: list[str] = []

    def _fake_run_direction(**kwargs):
        direction = kwargs["direction"]
        call_order.append(direction)
        out_dir = Path(kwargs["output_dir"]) / "merged"
        surface = source_surface if direction == "down" else grown_surface
        write_tifxyz(out_dir, surface, overwrite=True)
        summary = {
            "direction": direction,
            "call_count": 1,
            "final_tifxyz_path": str(out_dir),
            "calls": [{"stop_reason": "zero_growth_iteration"}],
            "final": {
                "direction": direction,
                "predicted_vertex_count": 3 if direction == "down" else 7,
                "cumulative_predicted_vertex_count": 3 if direction == "down" else 10,
                "final_predicted_nonseam_vertex_count": 6,
                "final_seam_vertex_count": 1,
                "new_band_frontier_coverage_fraction": 1.0,
                "new_band_cell_coverage_fraction": 1.0,
                "new_band_max_gap": 0,
                "new_band_gap_spans": [],
                "first_uncovered_frontier_index": None,
                "iterations_completed": 1,
                "stop_reason": "zero_growth_iteration",
                "window_batch_size": 256,
                "encode_decode_ms_per_fitted_window": 1.0,
                "crop_cache_hits": 0,
                "crop_cache_misses": 1,
                "total_wall_ms": 1.0,
                "fast_infer_enabled": True,
                "compile_infer_requested": False,
                "compile_infer_actual": False,
                "amp_dtype": "bf16",
            },
        }
        return type("FakeDirectionRun", (), {"final_tifxyz_path": out_dir, "stage_summary": summary})()

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt._run_direction_to_exhaustion", _fake_run_direction)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt._upload_surface_output", lambda paths: None)

    state = _process_surface(
        surface={
            "prefix_name": "title_hunt_rd1",
            "relative_surface_id": "title_hunt_rd1/surface_b",
            "local_path": str(source_dir),
            "uuid": "surface_b",
        },
        source_s3_uri="s3://bucket/source",
        output_s3_uri="s3://bucket/out",
        volume_uri="s3://volume",
        dino_backbone="backbone",
        autoreg_checkpoint="ckpt",
        local_output_root=tmp_path / "outputs_order",
        state_root=state_root,
        manifest_jsonl_path=manifest_jsonl_path,
        coordinate_scale_factor=1.0,
        max_extension_iters_per_call=16,
        window_batch_size=256,
        show_progress=False,
        dry_run_mode=False,
        dry_run_max_calls_per_direction=2,
        distributed_infer=False,
        extend_directions=["down", "up"],
    )

    manifest = json.loads(Path(state["manifest_path"]).read_text())
    assert call_order == ["down", "up"]
    assert manifest["directions"] == ["down", "up"]
    assert manifest["cumulative_predicted_vertex_count"] == 4
    assert manifest["final_predicted_vertex_count"] == 4


def test_title_hunt_planner_preflight_picks_best_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_surface = _make_surface()
    source_dir = write_tifxyz(tmp_path / "source_preflight", source_surface, overwrite=True)

    def _fake_scale(source_dir, output_dir, *, coordinate_scale_factor, target_vertex_spacing=0.0):
        del source_dir, coordinate_scale_factor
        out_dir = Path(output_dir)
        write_tifxyz(out_dir, source_surface, overwrite=True)
        if target_vertex_spacing > 0:
            return out_dir, 2
        return out_dir, 1

    def _fake_evaluate(**kwargs):
        window = int(kwargs["window_strip_length"])
        if window == 24:
            fitted = 19
            covered = 180
        elif window == 32:
            fitted = 18
            covered = 170
        else:
            fitted = 9
            covered = 90
        return {
            "direction": kwargs["grow_direction"],
            "requested_direction": kwargs["grow_direction"],
            "frontier_length": 200,
            "fitted_plans": fitted,
            "covered_frontier": covered,
            "covered_frontier_fraction": covered / 200.0,
            "planning_stats": {"crop_fit_failed_count": 10},
            "fitted_plan_spans": [(0, covered)],
            "covered_frontier_spans": [(0, covered)],
            "trimmed_bbox_rc": [0, 4, 0, 5],
        }

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt._scale_stored_tifxyz", _fake_scale)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt.evaluate_extension_planner", _fake_evaluate)

    summary = run_title_hunt_planner_preflight(
        source_tifxyz_path=source_dir,
        output_dir=tmp_path / "preflight",
    )

    assert summary["best_candidate"]["name"] == "expanded_context"
    assert summary["best_candidate"]["fitted_plans"] >= TITLE_HUNT_PREFLIGHT_MIN_FITTED_PLANS
    assert Path(summary["planner_preflight_path"]).exists()


def test_run_batch_extend_title_hunt_stops_after_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "z")
    monkeypatch.setattr(
        "vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt._list_title_hunt_prefixes",
        lambda source_s3_uri: ["title_hunt_rd1", "title_hunt_rd2"],
    )
    monkeypatch.setattr(
        "vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt._sync_title_hunt_prefix",
        lambda source_s3_uri, prefix_name, local_source_root: local_source_root / prefix_name,
    )
    monkeypatch.setattr(
        "vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt._discover_local_surfaces",
        lambda local_source_root, prefixes: [
            {"prefix_name": "title_hunt_rd1", "relative_surface_id": "title_hunt_rd1/a", "local_path": "/tmp/a", "uuid": "a"},
            {"prefix_name": "title_hunt_rd2", "relative_surface_id": "title_hunt_rd2/b", "local_path": "/tmp/b", "uuid": "b"},
            {"prefix_name": "title_hunt_rd2", "relative_surface_id": "title_hunt_rd2/c", "local_path": "/tmp/c", "uuid": "c"},
        ],
    )
    processed_ids: list[str] = []

    def _fake_process_surface(**kwargs):
        processed_ids.append(kwargs["surface"]["relative_surface_id"])
        return {
            "relative_surface_id": kwargs["surface"]["relative_surface_id"],
            "timestamp_suffix": "20260419T120000Z",
            "status": "uploaded",
        }

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt._process_surface", _fake_process_surface)

    result = run_batch_extend_title_hunt(
        source_s3_uri="s3://bucket/source",
        output_s3_uri="s3://bucket/out",
        volume_uri="s3://volume",
        dino_backbone="backbone",
        autoreg_checkpoint="ckpt",
        local_source_root=tmp_path / "source_root",
        local_output_root=tmp_path / "output_root",
        state_root=tmp_path / "state_root",
        coordinate_scale_factor=4.0,
        max_extension_iters_per_call=16,
        window_batch_size=256,
        dry_run_surface_count=2,
        dry_run_max_calls_per_direction=2,
        auto_continue=False,
        show_progress=False,
        distributed_infer=False,
    )

    assert processed_ids == ["title_hunt_rd1/a", "title_hunt_rd2/b"]
    assert result["dry_run_surface_count"] == 2
    assert len(result["processed"]) == 2
    assert result["distributed_infer"] is False


def test_sync_title_hunt_prefix_passes_exclude_patterns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "z")
    calls = {}

    class _Result:
        stdout = ""

    def _fake_run_cmd(args, **kwargs):
        calls["args"] = list(args)
        return _Result()

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt._run_cmd", _fake_run_cmd)

    from vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt import _sync_title_hunt_prefix

    out = _sync_title_hunt_prefix(
        source_s3_uri="s3://bucket/root",
        prefix_name="title_hunt_rd1",
        local_source_root=tmp_path / "sources",
    )

    assert out == tmp_path / "sources" / "title_hunt_rd1"
    assert calls["args"][:4] == ["aws", "s3", "sync", "s3://bucket/root/title_hunt_rd1/"]
    assert "--only-show-errors" in calls["args"]
    assert "layers/*" in calls["args"]
    assert "*.zarr/*" in calls["args"]
