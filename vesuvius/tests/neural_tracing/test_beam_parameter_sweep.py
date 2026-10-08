import json
from pathlib import Path

import pytest

from vesuvius.neural_tracing.fiber_trace_3d import beam_parameter_sweep as sweep


BASE_ARGS = ["--threads", "1", "--beam-width", "8", "--beam-lookahead-steps", "2",
             "--step-voxels", "4", "--smoothness-normal-weight", "0.1",
             "--smoothness-tangent-weight", "10", "--embedding-weight", "0"]


def test_apply_trace_args_replaces_in_place_and_appends_sorted():
    out = sweep.apply_trace_args(BASE_ARGS, {"smoothness-tangent-weight": 100,
        "beam-lookahead-steps": 4, "smoothness-free-angle-degrees": 5.0, "exhaustive-lookahead": True})
    assert out == ["--threads", "1", "--beam-width", "8", "--beam-lookahead-steps", "4",
                   "--step-voxels", "4", "--smoothness-normal-weight", "0.1",
                   "--smoothness-tangent-weight", "100", "--embedding-weight", "0",
                   "--exhaustive-lookahead", "--smoothness-free-angle-degrees", "5"]
    assert sweep.apply_trace_args(BASE_ARGS, {}) == BASE_ARGS
    assert sweep.apply_trace_args(out, {"smoothness-free-angle-degrees": 2.5})[-1] == "2.5"
    pooled = sweep.apply_trace_args(BASE_ARGS, {"sample-pool-radius-voxels": 1, "sample-pool-mode": "mean"})
    assert pooled[-4:] == ["--sample-pool-mode", "mean", "--sample-pool-radius-voxels", "1"]
    assert sweep.apply_trace_args(pooled, {"sample-pool-mode": "max"})[-3] == "max"
    with pytest.raises(ValueError, match="one of"):
        sweep.apply_trace_args(BASE_ARGS, {"sample-pool-mode": "median"})


@pytest.mark.parametrize("overrides", [{"embedding-weight": 1}, {"beam-width": "8"},
                                       {"beam-width": True}, {"exhaustive-lookahead": False}])
def test_apply_trace_args_rejects_unknown_or_mistyped_flags(overrides):
    with pytest.raises(ValueError):
        sweep.apply_trace_args(BASE_ARGS, overrides)


def test_ordered_variants_require_baseline_first_and_unique_names():
    good = {"beam_sweep": {"variants": [{"name": "baseline"},
                                        {"name": "t100", "parameters": {"smoothness-tangent-weight": 100}}]}}
    rows = sweep.ordered_variants(good)
    assert [r["name"] for r in rows] == ["baseline", "t100"] and rows[0]["parameters"] == {}
    with pytest.raises(ValueError, match="baseline"):
        sweep.ordered_variants({"beam_sweep": {"variants": [
            {"name": "t100", "parameters": {"smoothness-tangent-weight": 100}}]}})
    with pytest.raises(ValueError, match="duplicate"):
        sweep.ordered_variants({"beam_sweep": {"variants": [{"name": "a"}, {"name": "a"}]}})
    with pytest.raises(ValueError, match="unsupported beam_sweep keys"):
        sweep.ordered_variants({"beam_sweep": {"variants": [{"name": "a"}], "devices": "0"}})


def _summary(episode_arcs_by_case):
    cases = []
    for index, arcs in enumerate(episode_arcs_by_case):
        cases.append({"source_index": 0, "run_index": index, "direction": "forward",
                      "reference_length_base_voxels": 1000.0, "evaluated_through_base_voxels": 1000.0,
                      "evaluation_complete": True,
                      "failures": [{"directional_reference_arc_base_voxels": a} for a in arcs]})
    return {"sources": [{"source_id": "S", "cases": cases}]}


def test_direction_episodes_groups_short_distance_failures():
    rows = sweep.direction_episodes(_summary([[], [100.0, 120.0, 150.0], [100.0, 500.0]]))
    assert rows["S:0:0:forward"] == {"episodes": 0, "raw_failures": 0}
    assert rows["S:0:1:forward"] == {"episodes": 1, "raw_failures": 3}
    assert rows["S:0:2:forward"] == {"episodes": 2, "raw_failures": 2}


def test_compare_to_baseline_counts_episode_changes_not_raw_failures():
    baseline = {"status": "completed", "direction_episodes": sweep.direction_episodes(
        _summary([[100.0, 120.0, 150.0], [100.0, 500.0], []]))}
    # Case0 keeps one episode with fewer raw failures: no improvement. Case1
    # loses an episode: improved. Case2 gains one: worsened.
    row = {"status": "completed", "name": "v", "direction_episodes": sweep.direction_episodes(
        _summary([[100.0], [100.0], [700.0]]))}
    sweep.compare_to_baseline(row, baseline)
    assert (row["improved_cases"], row["worsened_cases"]) == (1, 1)
    assert row["improved"] == ["S:0:1:forward"] and row["worsened"] == ["S:0:2:forward"]


def test_ranking_prefers_fewer_episodes_then_longer_mean():
    rows = [{"name": "a", "status": "completed", "global_result": {"failures": 5, "mean_segment_length_mm": 9}},
            {"name": "b", "status": "completed", "global_result": {"failures": 4, "mean_segment_length_mm": 1}},
            {"name": "c", "status": "failed"},
            {"name": "d", "status": "completed", "global_result": {"failures": 5, "mean_segment_length_mm": 12}}]
    assert [r["name"] for r in sorted(rows, key=sweep.ranking_key)] == ["b", "d", "a", "c"]


def test_bind_cached_cohort_waives_only_markers(tmp_path, monkeypatch):
    previous = tmp_path / "full"
    previous.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "inference.json").write_text(json.dumps({"model": {"checkpoint_sha256": "ck"}}))
    (cache / "manifest.json").write_text("{}")
    fiber, normal = tmp_path / "fiber.json", tmp_path / "normal.json"
    fiber.write_text("{}"); normal.write_text("{}")
    config = {"sources": [{"id": "S", "fiber_manifest": str(cache / "manifest.json"),
                           "fibers": [str(fiber)], "normal_manifest": str(normal)}],
              "trace_args": list(BASE_ARGS), "live_endpoint": {"port": 1}, "render": True}
    (previous / "summary.json").write_text(json.dumps({"completed": True, "config": config,
                                                       "failures": 3, "cases": 2}))
    (previous / "inference_summary.json").write_text(json.dumps({"cache_roots": {"k": str(cache)}}))
    (previous / "incomplete.json").write_text("{}")
    monkeypatch.setattr(sweep, "sha256_file", lambda p: "h")
    workload = {"cached_evaluation": str(previous), "checkpoint_sha256": "ck"}
    with pytest.raises(ValueError, match="incomplete markers"):
        sweep.bind_cached_cohort(workload)
    binding = sweep.bind_cached_cohort(workload, accept_incomplete_markers=True)
    assert binding["incomplete_markers"] == ["incomplete.json"]
    assert binding["previous_failures"] == 3
    assert "live_endpoint" not in binding["evaluation"] and binding["evaluation"]["render"] is False
    (previous / "summary.json").write_text(json.dumps({"completed": False, "config": config}))
    with pytest.raises(ValueError, match="completed"):
        sweep.bind_cached_cohort(workload, accept_incomplete_markers=True)
    (previous / "summary.json").write_text(json.dumps({"completed": True, "config": config,
                                                       "failures": 3, "cases": 2}))
    with pytest.raises(ValueError, match="cached checkpoint mismatch"):
        sweep.bind_cached_cohort(dict(workload, checkpoint_sha256="other"), accept_incomplete_markers=True)


def test_run_variant_rewrites_trace_args_and_reports_episodes(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import cached_trace_profile as cached
    binding = {"evaluation": {"trace_args": list(BASE_ARGS), "workers": 32, "sources": []},
               "input_sha256": {}}
    monkeypatch.setattr(sweep, "sha256_file", lambda p: "h")
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: json.dumps(
        {"sources": [{"fibers": [{"runs": [1]}]}]}))
    seen = {}
    def run_cached(evaluation, binary):
        seen.update(evaluation)
        summary = _summary([[100.0, 120.0], []])
        summary.update(cases=2, wall_seconds=1.5)
        return summary
    monkeypatch.setattr(cached, "write_metric_comparison", lambda result, path: {"after": {
        "global_result": {"failures": 1, "mean_segment_length_mm": 2, "median_segment_length_mm": 1},
        "scrolls": {}, "collections": {}}})
    row = sweep.run_variant(binding, {"name": "t100", "parameters": {"smoothness-tangent-weight": 100}},
                            tmp_path / "bin", tmp_path / "attempt_001", workers=64, run_cached=run_cached)
    assert seen["trace_args"][seen["trace_args"].index("--smoothness-tangent-weight") + 1] == "100"
    assert seen["workers"] == 64 and seen["output"] == str(tmp_path / "attempt_001" / "full")
    assert row["raw_failures"] == 2 and row["failure_free_cases"] == 1
    assert row["direction_episodes"]["S:0:0:forward"]["episodes"] == 1
    assert json.loads((tmp_path / "attempt_001" / "direction_episodes.json").read_text())["S:0:1:forward"]["episodes"] == 0


def test_manager_settings_select_cached_or_demand():
    workload = {"beam_sweep": {"variants": [{"name": "baseline"}]}, "devices": "0,1",
                "evaluation": {"gpu_manager": {"enabled": True, "exclusive": True,
                                               "vram_bytes_per_gpu": 8 << 30, "priority": 100}}}
    assert sweep.manager_settings(workload) is None
    workload["beam_sweep"]["inference"] = "demand"
    settings = sweep.manager_settings(workload)
    assert settings["devices"] == "0,1" and settings["exclusive"] is True
    workload["evaluation"]["gpu_manager"]["enabled"] = False
    with pytest.raises(ValueError, match="enabled gpu_manager"):
        sweep.manager_settings(workload)
    workload["beam_sweep"]["inference"] = "other"
    with pytest.raises(ValueError, match="cached.*demand"):
        sweep.manager_settings(workload)


def test_session_replay_opens_per_variant_and_waits_for_busy_writer(tmp_path):
    events = []
    class Session:
        def run(self, evaluation):
            events.append("run")
            out = tmp_path / "full"
            out.mkdir(exist_ok=True)
            (out / "summary.json").write_text(json.dumps({"cases": 1}))
        def close(self):
            events.append("close")
    attempts = iter([RuntimeError("live inference cache already has a writer: x"), Session(), Session()])
    def open_session():
        item = next(attempts)
        if isinstance(item, Exception):
            events.append("busy")
            raise item
        events.append("open")
        return item
    clock = iter(range(0, 1000, 10))
    replay = sweep._session_replay(open_session, lock_wait_seconds=100,
                                   sleep=lambda s: events.append("sleep"), clock=lambda: next(clock))
    assert replay({"output": str(tmp_path / "full")}, None) == {"cases": 1}
    assert replay({"output": str(tmp_path / "full")}, None) == {"cases": 1}
    assert events == ["busy", "sleep", "open", "run", "close", "open", "run", "close"]
    with pytest.raises(RuntimeError, match="other failure"):
        sweep._session_replay(lambda: (_ for _ in ()).throw(RuntimeError("other failure")),
                              sleep=lambda s: None)({"output": str(tmp_path)}, None)


def test_remap_cache_path_moves_only_paths_inside_namespace(tmp_path):
    src, dst = tmp_path / "ns", tmp_path / "copy"
    (src / "a" / "b").mkdir(parents=True)
    remap = {"from": str(src), "to": str(dst)}
    assert sweep.remap_cache_path(str(src / "a" / "b" / "m.json"), remap) == str(dst / "a" / "b" / "m.json")
    assert sweep.remap_cache_path("/x", None) == "/x"
    with pytest.raises(ValueError, match="outside"):
        sweep.remap_cache_path(str(tmp_path / "elsewhere" / "m.json"), remap)
    with pytest.raises(ValueError, match="from.*to"):
        sweep.remap_cache_path("/x", {"from": "/a"})


def test_bind_cached_cohort_remaps_and_verifies_copy(tmp_path, monkeypatch):
    previous = tmp_path / "full"
    previous.mkdir()
    ns, copy = tmp_path / "ns", tmp_path / "copy"
    for base in (ns, copy):
        cache = base / "src" / "model"
        cache.mkdir(parents=True)
        (cache / "inference.json").write_text(json.dumps({"model": {"checkpoint_sha256": "ck"}}))
        (cache / "cache_identity.json").write_text("{\"id\": 1}")
        (cache / "manifest.json").write_text("{}")
    fiber, normal = tmp_path / "fiber.json", tmp_path / "normal.json"
    fiber.write_text("{}"); normal.write_text("{}")
    config = {"sources": [{"id": "S", "fiber_manifest": str(ns / "src" / "model" / "manifest.json"),
                           "fibers": [str(fiber)], "normal_manifest": str(normal)}],
              "trace_args": list(BASE_ARGS)}
    (previous / "summary.json").write_text(json.dumps({"completed": True, "config": config,
                                                       "failures": 3, "cases": 2}))
    (previous / "inference_summary.json").write_text(json.dumps({"cache_roots": {"k": str(ns / "src" / "model")}}))
    monkeypatch.setattr(sweep, "sha256_file", lambda p: "h")
    workload = {"cached_evaluation": str(previous), "checkpoint_sha256": "ck"}
    remap = {"from": str(ns), "to": str(copy)}
    binding = sweep.bind_cached_cohort(workload, cache_remap=remap)
    assert binding["evaluation"]["sources"][0]["fiber_manifest"] == str(copy / "src" / "model" / "manifest.json")
    assert binding["cache_remap"] == remap
    (copy / "src" / "model" / "cache_identity.json").write_text("{\"id\": 2}")
    with pytest.raises(ValueError, match="cache copy differs"):
        sweep.bind_cached_cohort(workload, cache_remap=remap)


def test_tube_chunks_covers_margin_around_line_and_contains_crossings():
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import reference_chunks, tube_chunks
    shape, chunk = (256, 256, 512), 64
    # A straight line along X at Y=Z=100 output voxels, base factor 2.
    points = [[20.0, 200.0, 200.0], [900.0, 200.0, 200.0]]
    crossings = reference_chunks(points, 2, shape, chunk)
    tube = tube_chunks(points, 0.0, 2, shape, chunk)
    assert crossings <= tube
    wide = tube_chunks(points, 2 * 70.0, 2, shape, chunk)  # 70 output voxels
    assert tube < wide
    # 70 voxels from y=100 reaches y=30..170, i.e. chunk rows 0..2 in Y and Z,
    # but never row 3 (starts at 192).
    assert {index[1] for index in wide} == {0, 1, 2} and {index[0] for index in wide} == {0, 1, 2}
    assert all(0 <= index[2] < 8 for index in wide)
    # Nothing beyond the volume and nothing at the far X end the line never reaches.
    assert max(index[2] for index in wide) <= (450 + 70) // 64
    with pytest.raises(ValueError):
        tube_chunks(points, -1.0, 2, shape, chunk)


def test_reference_backgrounds_apply_tube_margin(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace_3d import cached_trace_profile as cached
    (tmp_path / "cache_identity.json").write_text(json.dumps({
        "identity": {"source": {"level": 1}},
        "geometry": {"shape": [256, 256, 512], "tile": 64, "overlap": 0, "border": 0, "scale": 1, "chunk": 64}}))
    stores = {"k": SimpleNamespace(root=tmp_path)}
    plan = {"sources": [{"source_id": "S", "fibers": [{"runs": [
        {"points_base_xyz": [[20.0, 200.0, 200.0], [900.0, 200.0, 200.0]]}]}]}]}
    narrow = cached.reference_backgrounds({"reference_plan": plan}, stores, {"S": "k"})
    wide = cached.reference_backgrounds({"reference_plan": plan,
        "reference_prefetch_margin_base_voxels": 140}, stores, {"S": "k"})
    assert narrow["k"] < wide["k"]
    with pytest.raises(ValueError, match="positive"):
        cached.reference_backgrounds({"reference_plan": plan,
            "reference_prefetch_margin_base_voxels": 0}, stores, {"S": "k"})


def test_fixed_workload_ignores_runtime_only_settings():
    a = {"x": 1, "evaluation": {"trace_args": ["--a"], "gpu_manager": {"idle_release_seconds": 30}},
         "beam_sweep": {"variants": [{"name": "baseline"}], "workers": 64, "inference": "demand"}}
    b = {"x": 1, "evaluation": {"trace_args": ["--a"], "gpu_manager": {"idle_release_seconds": 3600}},
         "beam_sweep": {"variants": [{"name": "baseline"}, {"name": "v", "parameters": {"beam-width": 4}}],
                        "workers": 64, "inference": "demand", "reference_tube_base_voxels": 250}}
    assert sweep._fixed_workload(a) == sweep._fixed_workload(b)
    c = dict(b, beam_sweep=dict(b["beam_sweep"], workers=32, decoded_cache_bytes=1 << 30))
    assert sweep._fixed_workload(a) == sweep._fixed_workload(c)
    d = dict(b, beam_sweep=dict(b["beam_sweep"], inference="cached"))
    assert sweep._fixed_workload(a) != sweep._fixed_workload(d)


def test_validate_completed_restamps_superseded_identity(tmp_path):
    pointer = tmp_path / "result.json"
    row = {"status": "failed", "name": "v"}
    pointer.write_text(json.dumps({"variant": {"name": "v", "parameters": {}}, "matrix_identity_sha256": "old",
                                   "attempt_root": str(tmp_path), "metrics_sha256": None, "row": row}))
    with pytest.raises(ValueError, match="provenance changed"):
        sweep._validate_completed(pointer, {"name": "v", "parameters": {}}, "new")
    assert sweep._validate_completed(pointer, {"name": "v", "parameters": {}}, "new", {"old"}) == row
    assert json.loads(pointer.read_text())["matrix_identity_sha256"] == "new"


def test_run_variant_passes_native_environment(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import cached_trace_profile as cached
    binding = {"evaluation": {"trace_args": list(BASE_ARGS), "sources": []}, "input_sha256": {},
               "native_environment": {"LD_PRELOAD": "/lib/x.so"}, "decoded_cache_bytes": 2 << 30}
    monkeypatch.setattr(sweep, "sha256_file", lambda p: "h")
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: json.dumps({"sources": [{"fibers": [{"runs": [1]}]}]}))
    seen = {}
    def run_cached(evaluation, binary):
        seen.update(evaluation)
        summary = _summary([[]]); summary.update(cases=2, wall_seconds=1.0)
        return summary
    monkeypatch.setattr(cached, "write_metric_comparison", lambda result, path: {"after": {
        "global_result": {"failures": 0, "mean_segment_length_mm": 1, "median_segment_length_mm": 1}, "scrolls": {}, "collections": {}}})
    sweep.run_variant(binding, {"name": "v", "parameters": {}}, tmp_path / "bin", tmp_path / "a", run_cached=run_cached)
    assert seen["native_environment"] == {"LD_PRELOAD": "/lib/x.so"} and seen["decoded_cache_bytes"] == 2 << 30
    assert sweep._fixed_workload({"beam_sweep": {"native_environment": {"A": "1"}}}) == sweep._fixed_workload({"beam_sweep": {}})


def test_completed_pointer_records_binary_hash(tmp_path, monkeypatch):
    # The sweep identity must not pin the binary; each variant records its own.
    monkeypatch.setattr(sweep, "sha256_file", lambda p: "bin-" + Path(p).name)
    from pathlib import Path as _P
    assert "native_sha256" not in json.dumps(sweep._fixed_workload({"native_binary": "/x/bin"}))
