from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz import (
    ExtensionInferenceRuntime,
    PlannerAtlasEntry,
    PlannerCandidateSpec,
    PlannerSelectionRecord,
    PLANNER_MODE_COVERAGE_FIRST,
    _DistributedInferRuntime,
    _apply_geometric_gap_fill,
    _build_admissibility_atlas,
    _flatten_gathered_window_results,
    _initialize_distributed_infer_runtime,
    _plan_extension_windows_coverage_first,
    _plan_extension_windows_for_mode,
    _select_coverage_first_entries,
    _shard_fitted_window_plans,
    _crop_min_corner_for_points,
    _extract_prompt_window,
    _fit_child_candidate_recursive,
    _fit_window_for_crop,
    _open_zarr_volume,
    _plan_extension_windows,
    _parse_int_list,
    _render_projection,
    _surface_grid_zyx,
    _window_ranges,
    ExtensionWindowCandidate,
    build_extension_sample,
    choose_source_tifxyz,
    choose_growth_direction,
    demote_previous_seam,
    extend_tifxyz_mesh,
    finalize_iteration_extension,
    grid_to_colored_mesh,
    infer_extension_windows_batched,
    infer_extension_windows_batched_cached,
    merge_window_prediction,
    resolve_growth_direction,
    evaluate_extension_planner,
    run_extension_benchmark_suite,
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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_env(rank: int, world_size: int, port: int) -> dict[str, str]:
    return {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(int(port)),
        "WORLD_SIZE": str(int(world_size)),
        "RANK": str(int(rank)),
        "LOCAL_RANK": str(int(rank)),
    }


def test_window_ranges_are_deterministic() -> None:
    windows = _window_ranges(20, 8, 2)
    assert [(window.start, window.end) for window in windows] == [(0, 8), (6, 14), (12, 20)]


def test_parse_int_list_handles_csv() -> None:
    assert _parse_int_list("1, 2,4") == [1, 2, 4]
    assert _parse_int_list("") == []
    with pytest.raises(ValueError):
        _parse_int_list("0,2")


def test_distributed_infer_runtime_defaults_to_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)

    runtime = _initialize_distributed_infer_runtime(enabled=False, device="cpu")

    assert runtime.is_distributed is False
    assert runtime.rank == 0
    assert runtime.world_size == 1
    assert runtime.is_main_process is True
    assert str(runtime.device) == "cpu"


def test_distributed_infer_runtime_resolves_env_driven_gloo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29501")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr("torch.distributed.init_process_group", lambda backend, init_method: None)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 1)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 2)

    runtime = _initialize_distributed_infer_runtime(enabled=True, device="cpu")

    assert runtime.is_distributed is True
    assert runtime.rank == 1
    assert runtime.world_size == 2
    assert runtime.backend == "gloo"
    assert str(runtime.device) == "cpu"


def test_shard_fitted_window_plans_is_strided() -> None:
    plans = [
        SimpleNamespace(window=SimpleNamespace(start=idx, end=idx + 2), prompt_strips=3, predict_strips=2)
        for idx in range(6)
    ]
    runtime = _DistributedInferRuntime(
        is_distributed=True,
        rank=1,
        local_rank=1,
        world_size=3,
        device=torch.device("cpu"),
        backend="gloo",
        initialized_process_group=False,
    )

    shard = _shard_fitted_window_plans(plans, runtime=runtime, shard_mode="strided")

    assert [global_index for global_index, _ in shard] == [1, 4]


def test_flatten_gathered_window_results_preserves_global_order() -> None:
    gathered_payloads = [
        {
            "rank": 0,
            "fitted_window_count": 2,
            "results": [
                {"global_index": 0, "window": SimpleNamespace(start=0, end=2)},
                {"global_index": 4, "window": SimpleNamespace(start=4, end=6)},
            ],
        },
        {
            "rank": 1,
            "fitted_window_count": 2,
            "results": [
                {"global_index": 1, "window": SimpleNamespace(start=1, end=3)},
                {"global_index": 3, "window": SimpleNamespace(start=3, end=5)},
            ],
        },
        {
            "rank": 2,
            "fitted_window_count": 1,
            "results": [
                {"global_index": 2, "window": SimpleNamespace(start=2, end=4)},
            ],
        },
    ]

    flattened, per_rank_counts = _flatten_gathered_window_results(gathered_payloads)

    assert per_rank_counts == [2, 2, 1]
    assert [result["global_index"] for result in flattened] == [0, 1, 2, 3, 4]


def test_crop_min_corner_rejects_oversize_envelope() -> None:
    points = np.array([[0.0, 0.0, 0.0], [256.0, 32.0, 32.0]], dtype=np.float32)
    assert _crop_min_corner_for_points(points, (128, 128, 128)) is None


def test_fit_window_for_crop_shrinks_below_sixteen_strips() -> None:
    rows, cols = 32, 12
    row_axis = np.arange(rows, dtype=np.float32)[:, None]
    col_axis = np.arange(cols, dtype=np.float32)[None, :]
    row_grid = np.broadcast_to(row_axis, (rows, cols))
    col_grid = np.broadcast_to(col_axis, (rows, cols))
    grid = np.stack(
        [
            16.0 + 24.0 * row_grid,
            32.0 + 2.0 * row_grid,
            48.0 + 2.0 * col_grid,
        ],
        axis=-1,
    ).astype(np.float32)

    fitted = _fit_window_for_crop(
        grid,
        direction="left",
        window=SimpleNamespace(start=0, end=24),
        prompt_strips=3,
        predict_strips=2,
        crop_size=(128, 128, 128),
        max_crop_fit_retries=3,
        min_window_strip_length=4,
    )

    assert fitted is not None
    assert fitted.window.end - fitted.window.start == 8
    assert fitted.prompt_strips == 3
    assert fitted.predict_strips == 2


def test_plan_extension_windows_retiles_parent_span_densely() -> None:
    rows, cols = 24, 12
    row_axis = np.arange(rows, dtype=np.float32)[:, None]
    col_axis = np.arange(cols, dtype=np.float32)[None, :]
    row_grid = np.broadcast_to(row_axis, (rows, cols))
    col_grid = np.broadcast_to(col_axis, (rows, cols))
    grid = np.stack(
        [
            16.0 + 24.0 * row_grid,
            32.0 + 2.0 * row_grid,
            48.0 + 2.0 * col_grid,
        ],
        axis=-1,
    ).astype(np.float32)

    plans, stats = _plan_extension_windows(
        grid,
        direction="left",
        window_strip_length=24,
        window_overlap=0,
        prompt_strips=3,
        predict_strips=2,
        crop_size=(128, 128, 128),
        max_crop_fit_retries=3,
    )

    spans = [(plan.window.start, plan.window.end) for plan in plans]
    assert stats["parent_window_count"] == 1
    assert stats["child_window_count"] > 1
    assert stats["deduped_child_window_count"] == len(spans)
    assert spans[0][0] == 0
    assert spans[-1][1] == 24
    assert len(spans) == 5
    assert max(spans[idx + 1][0] - spans[idx][0] for idx in range(len(spans) - 1)) <= 4


def test_plan_extension_windows_dedupes_overlapping_parent_children() -> None:
    rows, cols = 12, 12
    row_axis = np.arange(rows, dtype=np.float32)[:, None]
    col_axis = np.arange(cols, dtype=np.float32)[None, :]
    row_grid = np.broadcast_to(row_axis, (rows, cols))
    col_grid = np.broadcast_to(col_axis, (rows, cols))
    grid = np.stack(
        [
            16.0 + 24.0 * row_grid,
            32.0 + 2.0 * row_grid,
            48.0 + 2.0 * col_grid,
        ],
        axis=-1,
    ).astype(np.float32)

    plans, stats = _plan_extension_windows(
        grid,
        direction="left",
        window_strip_length=8,
        window_overlap=4,
        prompt_strips=3,
        predict_strips=2,
        crop_size=(128, 128, 128),
        max_crop_fit_retries=3,
    )

    spans = [(plan.window.start, plan.window.end) for plan in plans]
    assert stats["parent_window_count"] == 2
    assert stats["child_window_count"] >= stats["deduped_child_window_count"]
    assert spans == sorted(set(spans))
    assert spans == [(0, 8), (4, 12)]


def test_resolve_growth_direction_projects_world_z() -> None:
    grid = _make_surface_grid(8, 10)

    decreasing, _ = resolve_growth_direction(
        grid,
        prompt_strips=3,
        predict_strips=1,
        crop_size=(128, 128, 128),
        requested_direction="decreasing-z",
    )
    increasing, _ = resolve_growth_direction(
        grid,
        prompt_strips=3,
        predict_strips=1,
        crop_size=(128, 128, 128),
        requested_direction="increasing-z",
    )

    assert decreasing == "down"
    assert increasing == "up"


def test_coverage_first_admissibility_atlas_is_deterministic() -> None:
    grid = _make_surface_grid(16, 24)
    specs = [
        PlannerCandidateSpec(
            phase="atlas",
            prompt_strips=8,
            predict_strips=1,
            window_strip_length=16,
            window_overlap=8,
        )
    ]
    atlas_a, summary_a = _build_admissibility_atlas(
        grid,
        direction="down",
        candidate_specs=specs,
        crop_size=(128, 128, 128),
        fit_cache={},
        planner_surrogate="surface_frame",
    )
    atlas_b, summary_b = _build_admissibility_atlas(
        grid,
        direction="down",
        candidate_specs=specs,
        crop_size=(128, 128, 128),
        fit_cache={},
        planner_surrogate="surface_frame",
    )

    assert summary_a == summary_b
    assert [
        (entry.requested_window.start, entry.requested_window.end, entry.fitted_plan is not None)
        for entry in atlas_a
    ] == [
        (entry.requested_window.start, entry.requested_window.end, entry.fitted_plan is not None)
        for entry in atlas_b
    ]


def test_coverage_first_selector_prefers_broader_union_coverage() -> None:
    dummy_prompt = np.zeros((8, 3, 3), dtype=np.float32)
    entries = [
        PlannerAtlasEntry(
            spec=PlannerCandidateSpec("atlas", 8, 1, 16, 8),
            requested_window=SimpleNamespace(start=0, end=10),
            fitted_plan=SimpleNamespace(window=SimpleNamespace(start=0, end=10), prompt_grid=dummy_prompt, min_corner=np.zeros(3), prompt_strips=8, predict_strips=1),
        ),
        PlannerAtlasEntry(
            spec=PlannerCandidateSpec("atlas", 12, 1, 20, 8),
            requested_window=SimpleNamespace(start=0, end=20),
            fitted_plan=SimpleNamespace(window=SimpleNamespace(start=0, end=20), prompt_grid=dummy_prompt, min_corner=np.zeros(3), prompt_strips=12, predict_strips=1),
        ),
        PlannerAtlasEntry(
            spec=PlannerCandidateSpec("atlas", 10, 1, 10, 4),
            requested_window=SimpleNamespace(start=20, end=30),
            fitted_plan=SimpleNamespace(window=SimpleNamespace(start=20, end=30), prompt_grid=dummy_prompt, min_corner=np.zeros(3), prompt_strips=10, predict_strips=1),
        ),
    ]

    selected, uncovered = _select_coverage_first_entries(entries, frontier_length=30)

    assert [(record.fitted_plan.window.start, record.fitted_plan.window.end) for record in selected] == [(0, 20), (20, 30)]
    assert not bool(uncovered.any())


def test_coverage_first_gap_rescue_targets_only_uncovered_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    call_intervals: list[list[tuple[int, int]] | None] = []

    def _fake_build_admissibility_atlas(grid_zyx, *, direction, candidate_specs, crop_size, fit_cache, frontier_intervals=None, planner_surrogate="surface_frame"):
        del grid_zyx, direction, candidate_specs, crop_size, fit_cache, planner_surrogate
        call_intervals.append(None if frontier_intervals is None else list(frontier_intervals))
        return [], {
            "frontier_length": 32,
            "requested_window_count": 0,
            "admissible_window_count": 0,
            "crop_fit_failed_count": 0,
            "candidate_specs": [],
        }

    selector_outputs = iter(
        [
            ([], np.array([False] * 8 + [True] * 4 + [False] * 8 + [True] * 4 + [False] * 8, dtype=bool)),
            ([], np.array([False] * 8 + [True] * 4 + [False] * 8 + [True] * 4 + [False] * 8, dtype=bool)),
        ]
    )

    def _fake_select(entries, *, frontier_length, uncovered_mask=None):
        del entries, frontier_length, uncovered_mask
        return next(selector_outputs)

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz._build_admissibility_atlas", _fake_build_admissibility_atlas)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz._select_coverage_first_entries", _fake_select)

    _plan_extension_windows_coverage_first(
        _make_surface_grid(8, 32),
        direction="down",
        crop_size=(128, 128, 128),
    )

    assert call_intervals == [None, [(8, 12), (20, 24)]]


def test_geometric_gap_fill_covers_ragged_frontier_columns() -> None:
    grid = _make_surface_grid(6, 8)
    grid[0, 1:-1, :] = np.nan
    sums = np.zeros((1, grid.shape[1], 3), dtype=np.float64)
    counts = np.zeros((1, grid.shape[1]), dtype=np.int32)
    counts[0, :2] = 1
    sums[0, :2] = grid[1, :2]

    filled_vertices, filled_frontier = _apply_geometric_gap_fill(
        sums,
        counts,
        working_grid=grid,
        direction="down",
    )

    assert filled_vertices > 0
    assert filled_frontier == grid.shape[1] - 2
    assert np.all(counts[0] > 0)


def test_evaluate_extension_planner_coverage_first_returns_diagnostics(tmp_path: Path) -> None:
    source_surface = _make_surface(8, 10)
    from vesuvius.tifxyz import write_tifxyz

    source_dir = write_tifxyz(tmp_path / "coverage_source", source_surface, overwrite=True)
    evaluation = evaluate_extension_planner(
        tifxyz_path=source_dir,
        grow_direction="left",
        planner_mode=PLANNER_MODE_COVERAGE_FIRST,
        prompt_strips=8,
        predict_strips_per_iter=1,
        window_strip_length=24,
        window_overlap=12,
    )

    assert evaluation["planner_mode"] == PLANNER_MODE_COVERAGE_FIRST
    assert evaluation["direction"] == "left"
    assert "planner_diagnostics" in evaluation
    assert "atlas" in evaluation["planner_diagnostics"]
    assert "selection" in evaluation["planner_diagnostics"]


def test_plan_extension_windows_recovers_failing_first_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_fit(
        grid_zyx,
        *,
        direction,
        window,
        prompt_strips,
        predict_strips,
        crop_size,
        max_crop_fit_retries,
        min_window_strip_length,
    ):
        del grid_zyx, direction, crop_size, max_crop_fit_retries, min_window_strip_length
        width = window.end - window.start
        if width >= 32 and window.start == 0:
            return None
        prompt_grid = np.zeros((width, prompt_strips, 3), dtype=np.float32)
        return SimpleNamespace(
            window=window,
            prompt_grid=prompt_grid,
            min_corner=np.zeros(3, dtype=np.int64),
            prompt_strips=prompt_strips,
            predict_strips=predict_strips,
        )

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz._fit_window_for_crop", _fake_fit)

    plans, stats = _plan_extension_windows(
        np.zeros((96, 8, 3), dtype=np.float32),
        direction="left",
        window_strip_length=48,
        window_overlap=16,
        prompt_strips=3,
        predict_strips=2,
        crop_size=(128, 128, 128),
        max_crop_fit_retries=3,
    )

    spans = [(plan.window.start, plan.window.end) for plan in plans]
    covered = np.zeros(96, dtype=bool)
    for start, end in spans:
        covered[start:end] = True
    assert stats["parent_window_count"] == 3
    assert spans[0][0] == 0
    assert covered[:32].all()


def test_fit_child_candidate_recursive_splits_to_width_two(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_fit(
        grid_zyx,
        *,
        direction,
        window,
        prompt_strips,
        predict_strips,
        crop_size,
        max_crop_fit_retries,
        min_window_strip_length,
    ):
        del grid_zyx, direction, crop_size, max_crop_fit_retries, min_window_strip_length
        width = window.end - window.start
        if width == 4:
            return None
        if width == 2 and window.start in {0, 2}:
            prompt_grid = np.zeros((width, prompt_strips, 3), dtype=np.float32)
            return SimpleNamespace(
                window=window,
                prompt_grid=prompt_grid,
                min_corner=np.zeros(3, dtype=np.int64),
                prompt_strips=prompt_strips,
                predict_strips=predict_strips,
            )
        return None

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz._fit_window_for_crop", _fake_fit)

    plans, failed_leaf_count = _fit_child_candidate_recursive(
        np.zeros((8, 8, 3), dtype=np.float32),
        direction="left",
        candidate=ExtensionWindowCandidate(window=SimpleNamespace(start=0, end=4), prompt_strips=3, predict_strips=2),
        crop_size=(128, 128, 128),
        max_crop_fit_retries=3,
        min_child_window_strip_length=2,
    )

    spans = [(plan.window.start, plan.window.end) for plan in plans]
    assert spans == [(0, 2), (2, 4)]
    assert failed_leaf_count == 1


def test_extension_inference_runtime_compile_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRuntimeModel:
        def encode_conditioning(self, volume, vol_tokens=None):
            del volume, vol_tokens
            return {"ok": True}

        def forward_from_encoded(self, batch, *, memory_tokens, memory_patch_centers):
            del batch, memory_tokens, memory_patch_centers
            return {"ok": True}

    monkeypatch.setattr("torch.compile", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("compile boom")))

    runtime = ExtensionInferenceRuntime(
        _FakeRuntimeModel(),
        device=torch.device("cuda"),
        fast_infer=True,
        compile_infer=True,
        amp_dtype="bf16",
    )

    assert runtime.fast_infer_enabled is True
    assert runtime.compile_infer_requested is True
    assert runtime.compile_infer_actual is False
    assert "compile boom" in str(runtime.compile_infer_failure)
    assert runtime.amp_dtype == torch.bfloat16
    assert runtime.encode_conditioning(None) == {"ok": True}


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


def test_demote_previous_seam_converts_old_seam_to_predicted() -> None:
    provenance = np.array([[0, 2, 1], [2, 2, 0]], dtype=np.uint8)
    demote_previous_seam(provenance)
    assert np.array_equal(provenance, np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8))


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
        ExtensionWindowPayload(0, ExtensionWindow(0, 4), sample_a, "left", (4, 2), 4, 2, 3, 2),
        ExtensionWindowPayload(1, ExtensionWindow(2, 6), sample_b, "left", (4, 2), 4, 2, 3, 2),
    ]


def _distributed_extension_worker(rank: int, world_size: int, port: int, tmpdir: str) -> None:
    import vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz as ext

    old_env = {key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK")}
    os.environ.update(_ddp_env(rank, world_size, port))
    original_read = ext.read_tifxyz
    original_open = ext._open_zarr_volume
    original_load = ext._load_autoreg_model
    try:
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

        ext.read_tifxyz = _fake_read_tifxyz
        ext._open_zarr_volume = _fake_open_volume
        ext._load_autoreg_model = _fake_load_model

        result = ext.extend_tifxyz_mesh(
            tifxyz_path="dummy",
            volume_uri="s3://dummy",
            dino_backbone="backbone",
            autoreg_checkpoint="ckpt",
            out_dir=Path(tmpdir) / "distributed_extend",
            device="cpu",
            grow_direction="left",
            prompt_strips=3,
            predict_strips_per_iter=2,
            window_strip_length=4,
            window_overlap=2,
            window_batch_size=2,
            max_extension_iters=1,
            distributed_infer=True,
            show_progress=False,
            fast_infer=True,
            compile_infer=False,
        )
        payload = {
            "rank": int(rank),
            "predicted_vertex_count": int(result["predicted_vertex_count"]),
            "distributed_infer_enabled": bool(result["distributed_infer_enabled"]),
            "distributed_world_size": int(result["distributed_world_size"]),
            "mesh_path_exists": bool(Path(result["mesh_path"]).exists()),
            "summary_path_exists": bool(Path(result["summary_path"]).exists()),
        }
        (Path(tmpdir) / f"distributed_rank{rank}.json").write_text(json.dumps(payload))
    finally:
        ext.read_tifxyz = original_read
        ext._open_zarr_volume = original_open
        ext._load_autoreg_model = original_load
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_batched_extension_inference_matches_serial() -> None:
    model = _FakeBatchModel()
    payloads = _make_window_payloads()

    serial_results, _, _ = infer_extension_windows_batched(model, payloads, window_batch_size=1, device=torch.device("cpu"))
    batched_results, _, peak_batch_size = infer_extension_windows_batched(
        model,
        payloads,
        window_batch_size=2,
        device=torch.device("cpu"),
        fast_infer=True,
        compile_infer=False,
    )

    assert peak_batch_size == 2
    assert len(serial_results) == len(batched_results) == 2
    for serial, batched in zip(serial_results, batched_results, strict=True):
        np.testing.assert_allclose(serial["continuation_grid_world"], batched["continuation_grid_world"])
        assert serial["window"] == batched["window"]


def test_cached_batched_inference_falls_back_when_cache_api_is_missing() -> None:
    model = _FakeBatchModel()
    payloads = _make_window_payloads()

    uncached_results, _, _ = infer_extension_windows_batched(
        model,
        payloads,
        window_batch_size=2,
        device=torch.device("cpu"),
    )
    cached_results, _, peak_batch_size = infer_extension_windows_batched_cached(
        model,
        payloads,
        window_batch_size=2,
        device=torch.device("cpu"),
    )

    assert peak_batch_size == 2
    assert len(cached_results) == len(uncached_results)
    for uncached, cached in zip(uncached_results, cached_results, strict=True):
        np.testing.assert_allclose(uncached["continuation_grid_world"], cached["continuation_grid_world"])


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
        grow_direction="left",
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
    assert result["final_predicted_nonseam_vertex_count"] > 0
    assert result["final_seam_vertex_count"] > 0
    assert result["final_predicted_nonseam_vertex_count"] > result["final_seam_vertex_count"]
    assert result["iterations_completed"] == 2
    assert result["stop_reason"] == "max_extension_iters"


def test_zero_growth_iteration_stops_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    call_counter = {"count": 0}

    def _fake_batched(*args, **kwargs):
        del args, kwargs
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            grid = np.ones((4, 2, 3), dtype=np.float32)
            return [{"window": SimpleNamespace(start=0, end=4), "continuation_grid_world": grid, "predicted_vertex_count": 8, "stop_count": 0}], 0.0, 1
        empty = np.full((4, 2, 3), np.nan, dtype=np.float32)
        return [{"window": SimpleNamespace(start=0, end=4), "continuation_grid_world": empty, "predicted_vertex_count": 0, "stop_count": 0}], 0.0, 1

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz.read_tifxyz", _fake_read_tifxyz)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz._open_zarr_volume", _fake_open_volume)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz._load_autoreg_model", _fake_load_model)
    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz.infer_extension_windows_batched", _fake_batched)

    result = extend_tifxyz_mesh(
        tifxyz_path="dummy",
        volume_uri="s3://dummy",
        dino_backbone="backbone",
        autoreg_checkpoint="ckpt",
        out_dir=tmp_path,
        device="cpu",
        grow_direction="left",
        prompt_strips=3,
        predict_strips_per_iter=2,
        window_strip_length=4,
        window_overlap=2,
        window_batch_size=2,
        max_extension_iters=3,
    )

    assert result["iterations_completed"] == 1
    assert result["stop_reason"] == "zero_growth_iteration"


def test_distributed_extension_inference_cpu_gloo(tmp_path: Path) -> None:
    port = _find_free_port()
    mp.spawn(_distributed_extension_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)

    rank0 = json.loads((tmp_path / "distributed_rank0.json").read_text())
    rank1 = json.loads((tmp_path / "distributed_rank1.json").read_text())

    assert rank0["distributed_infer_enabled"] is True
    assert rank0["distributed_world_size"] == 2
    assert rank0["predicted_vertex_count"] == rank1["predicted_vertex_count"]
    assert rank0["mesh_path_exists"] is True
    assert rank0["summary_path_exists"] is True


def test_extension_benchmark_suite_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    call_log = []

    def _fake_extend(**kwargs):
        batch_size = int(kwargs["window_batch_size"])
        max_iters = int(kwargs["max_extension_iters"])
        call_log.append((batch_size, max_iters))
        return {
            "window_batch_size": batch_size,
            "predicted_vertex_count": 10 * batch_size,
            "windows_per_second_overall": float(batch_size),
            "summary_path": str(tmp_path / f"summary_{batch_size}_{max_iters}.json"),
            "iteration_stats": [{"valid_new_vertices": batch_size * max_iters}],
        }

    monkeypatch.setattr("vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz.extend_tifxyz_mesh", _fake_extend)

    suite = run_extension_benchmark_suite(
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
        window_batch_sizes=[1, 2, 4],
        long_rollout_iters=3,
        max_crop_fit_retries=2,
    )

    assert call_log == [(1, 1), (2, 1), (4, 1), (4, 3)]
    assert suite["best_batch_run"]["window_batch_size"] == 4
    assert suite["long_rollout"]["window_batch_size"] == 4
    assert Path(suite["benchmark_suite_path"]).exists()


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
    assert result["iteration_stats"][0]["new_band_frontier_coverage_fraction"] >= 0.9
    assert result["iteration_stats"][0]["new_band_max_gap"] <= 2
    assert result["iteration_stats"][0]["new_band_gap_spans"] == []
    assert result["iteration_stats"][0]["first_uncovered_frontier_index"] is None
    assert result["iteration_stats"][0]["child_window_count"] >= result["iteration_stats"][0]["parent_window_count"]
    assert result["fast_infer_enabled"] is True
    assert result["distributed_infer_enabled"] is False
    assert result["distributed_world_size"] == 1
    assert result["per_rank_fitted_window_counts"] == [result["total_fitted_windows"]]
    assert result["gather_ms"] == pytest.approx(0.0)
    assert result["compile_infer_requested"] is False
    assert result["compile_infer_actual"] is False
    assert result["encode_decode_ms_per_fitted_window"] is not None
    for preview in result["preview_paths"]:
        assert Path(preview).exists()


def test_extend_tifxyz_mesh_coverage_first_writes_planner_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        grow_direction="left",
        planner_mode=PLANNER_MODE_COVERAGE_FIRST,
        max_extension_iters=1,
        show_progress=False,
    )

    assert result["planner_mode"] == PLANNER_MODE_COVERAGE_FIRST
    assert Path(result["planner_diagnostics_path"]).exists()
    diagnostics = json.loads(Path(result["planner_diagnostics_path"]).read_text())
    assert diagnostics["planner_mode"] == PLANNER_MODE_COVERAGE_FIRST
    assert diagnostics["resolved_lattice_direction"] == "left"
    assert "initial_preflight" in diagnostics
    assert "iterations" in diagnostics
