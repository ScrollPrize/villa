"""Ordered native evaluation, cancellation and callable/CLI equivalence."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from test_fiber_chunk_trace import ft, inputs, handles, chunk_config, trace_config
from test_fiber_live_bridge import Coordinator, bridge
from vc import fiber_evaluation as evaluation

pytestmark = pytest.mark.skipif(not ft.case_queue_supported(), reason="native case queue unavailable")


def fixture_cache(inputs):
    field, normals = handles(inputs)
    cache = ft.ChunkCache(field, normals, trace_config(), chunk_config(), str(inputs / "cache"), "tube")
    return field, normals, cache


def test_batch_matches_serial_and_is_single_use(inputs):
    field, normals, cache = fixture_cache(inputs)
    points = np.array([[16., 16., 16.], [220., 16., 16.]])
    cfg = ft.ChunkReplayConfig()
    cfg.error_threshold_base = 2
    cases = [ft.EvaluationCase(points), ft.EvaluationCase(points[::-1].copy()),
             ft.EvaluationCase(points, end_arc_base=100)]
    control = ft.EvaluationControl()
    result = cache.replay_many(cases, normals, cfg, control, workers=1, contexts_per_worker=3)
    serial = [cache.replay(points, normals, cfg), cache.replay(points[::-1].copy(), normals, cfg),
              cache.replay(points, normals, cfg, end_arc_base=100)]
    assert list(map(evaluation.replay_dict, result)) == list(map(evaluation.replay_dict, serial))
    assert control.status() == dict(total=3, completed=3, started=True, finished=True, cancelled=False, error="")
    with pytest.raises(ValueError, match="already used"):
        cache.replay_many(cases, normals, cfg, control)
    for kwargs in ({"workers": 0}, {"contexts_per_worker": 0}, {"reverse_interval_base": -1}):
        c = ft.EvaluationControl()
        with pytest.raises(ValueError):
            cache.replay_many(cases, normals, cfg, c, **kwargs)
        assert c.status()["finished"] and c.status()["error"]
    bad = ft.EvaluationCase(points, end_arc_base=1000)
    with pytest.raises(ValueError, match="arc interval"):
        cache.replay_many([bad], normals, cfg, ft.EvaluationControl())
    c = ft.EvaluationControl()
    c.cancel("cancel before start")
    with pytest.raises(RuntimeError, match="cancel before start"):
        cache.replay_many(cases, normals, cfg, c)
    assert c.status()["completed"] == 0


def test_cancel_joins_publication_and_duplicate_cache_waiters(inputs):
    received = threading.Event()
    with Coordinator(lambda *args: received.set()) as server:
        field, normals, cache = fixture_cache(inputs)
        client = bridge(server)
        field.set_live_bridge(client, "tube")
        case = ft.EvaluationCase(np.array([[16., 16., 16.], [220., 16., 16.]]))
        control = ft.EvaluationControl()
        with ThreadPoolExecutor(1) as pool:
            work = pool.submit(cache.replay_many, [case] * 4, normals, ft.ChunkReplayConfig(), control, 1, 4)
            assert received.wait(5)
            control.cancel("operator cancelled")
            with pytest.raises(RuntimeError, match="operator cancelled"):
                work.result(timeout=5)
        assert control.status()["cancelled"] and control.status()["finished"]
        assert control.status()["completed"] == 0
        # A cancelled bridge is fail-closed. Retry with a fresh session/control,
        # while retaining the chunk cache whose builder failed.
        with Coordinator(lambda s, conn, request: s.ready(conn, request)) as retry:
            replacement = bridge(retry)
            field.set_live_bridge(replacement, "tube")
            again = cache.replay_many([case], normals, ft.ChunkReplayConfig(), ft.EvaluationControl(), 1, 2)
            assert again[0].completed_reference_arc_base == 204
            replacement.cancel("close retry transport")
        client.cancel("close test transport")


def config(inputs):
    c = chunk_config()
    return dict(version=1, prediction_manifest="pred.json", normal_manifest="norm.json",
                source_identity="cli-tube", chunk_cache="cli-cache", enable_width=True,
                trace=trace_config().to_dict(),
                chunks={key: getattr(c, key) for key in evaluation.CHUNK_KEYS},
                replay=dict(error_threshold_base=2), workers=2, contexts_per_worker=2,
                cases=[dict(id="forward", points_base_xyz=[[16, 16, 16], [220, 16, 16]]),
                       dict(id="backward", points_base_xyz=[[220, 16, 16], [16, 16, 16]])])


def test_callable_and_cli(inputs):
    cfg = config(inputs)
    progress = []
    result = evaluation.evaluate(cfg, base_directory=inputs, progress=progress.append)
    assert progress[-1]["completed"] == 2
    assert result["failure_count"] == 0
    assert result["retained_trace_length_base"] > 0
    assert [row["id"] for row in result["cases"]] == ["forward", "backward"]
    path = inputs / "eval.json"
    path.write_text(json.dumps(cfg))
    output = inputs / "results.json"
    assert evaluation.main([str(path), str(output)]) == 0
    assert json.loads(output.read_text()) == json.loads(json.dumps(result))
    assert not list(inputs.glob("results.json.*.tmp"))
    cfg["cases"].append(cfg["cases"][0])
    with pytest.raises(ValueError, match="unique"):
        evaluation.evaluate(cfg, base_directory=inputs)
    cfg["unexpected"] = True
    with pytest.raises(ValueError, match="unknown evaluation"):
        evaluation.evaluate(cfg, base_directory=inputs)
    # Failed CLI runs do not overwrite the previous complete result.
    previous = output.read_bytes()
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        evaluation.main([str(path), str(output)])
    assert output.read_bytes() == previous


def test_progress_exception_cancels_live_work(inputs):
    control = ft.EvaluationControl()
    with Coordinator(lambda *args: None) as server:
        cfg = config(inputs)
        cfg["live_endpoint"] = dict(port=server.port, token="test-token", source="tube")
        def interrupt(status):
            raise KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            evaluation.evaluate(cfg, base_directory=inputs, control=control, progress=interrupt)
        assert control.status()["finished"] and control.status()["cancelled"]


def test_reverse_batch_matches_serial(inputs):
    field, normals, cache = fixture_cache(inputs)
    points = np.array([[16., 16., 16.], [220., 16., 16.]])
    cfg = ft.ChunkReplayConfig()
    serial = cache.replay(points, normals, cfg, reverse_interval_base=32)
    batch = cache.replay_many([ft.EvaluationCase(points)], normals, cfg,
                             ft.EvaluationControl(), reverse_interval_base=32)
    assert evaluation.replay_dict(serial) == evaluation.replay_dict(batch[0])


@pytest.mark.parametrize("reverse_interval", [0, 32])
def test_beam_batch_callable_and_serial(inputs, reverse_interval):
    field, normals = handles(inputs)
    trace = trace_config()
    trace.width_relative_tolerance = .4
    replay = ft.ChunkReplayConfig()
    points = np.array([[16., 16., 16.], [220., 16., 16.]])
    cases = [ft.EvaluationCase(points), ft.EvaluationCase(points[::-1].copy()),
             ft.EvaluationCase(points, end_arc_base=100)]
    serial = [ft.beam_replay(field, normals, normals, trace, c, replay, reverse_interval) for c in cases]
    control = ft.EvaluationControl()
    batch = ft.beam_replay_many(field, normals, normals, trace, cases, replay, control, 2, 2, reverse_interval)
    assert list(map(evaluation.replay_dict, batch)) == list(map(evaluation.replay_dict, serial))
    assert control.status()["completed"] == 3
    cfg = config(inputs)
    cfg.update(backend="beam", reverse_interval_base=reverse_interval, trace=trace.to_dict(), replay={})
    del cfg["chunks"], cfg["chunk_cache"]
    report = evaluation.evaluate(cfg, base_directory=inputs)
    assert [{k: v for k, v in row.items() if k != "id"} for row in report["cases"]] == list(
        map(evaluation.replay_dict, serial[:2]))
    assert report["failure_count"] == 0


def test_beam_cancel_waiting_for_publication(inputs):
    received = threading.Event()
    with Coordinator(lambda *args: received.set()) as server:
        field, normals = handles(inputs)
        client = bridge(server)
        field.set_live_bridge(client, "beam")
        control = ft.EvaluationControl()
        case = ft.EvaluationCase(np.array([[16., 16., 16.], [220., 16., 16.]]))
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(ft.beam_replay_many, field, normals, normals, trace_config(),
                                 [case] * 4, ft.ChunkReplayConfig(), control, 1, 4)
            assert received.wait(5)
            control.cancel("cancel beam")
            with pytest.raises(RuntimeError, match="cancel beam"):
                future.result(timeout=5)
        assert control.status()["finished"] and control.status()["completed"] == 0
        client.cancel("close")


def test_reference_selection_preserves_spans_and_direction(inputs):
    annotation = dict(type="vc3d_fiber", version=1, optimization_mode="native_fiber_trace3d",
                      line_points=[[0, 16, 16], [16, 16, 16], [100, 16, 16], [220, 16, 16], [240, 16, 16]],
                      control_points=[[16, 16, 16], [220, 16, 16]])
    cases, plan = evaluation.reference_cases(annotation, "a")
    assert len(cases) == 2
    assert plan["eligible_length_base_voxels"] == 204
    assert [row["reason"] for row in plan["excluded"]] == ["extrapolated", "extrapolated"]
    assert cases[0]["points_base_xyz"] == cases[1]["points_base_xyz"][::-1]
    annotation["evaluation_direction"] = "reverse"
    assert len(evaluation.reference_cases(annotation, "a")[0]) == 1
    path = inputs / "fiber.json"
    path.write_text(json.dumps(annotation))
    cfg = config(inputs)
    cfg["cases"] = [dict(id="a", fiber_path="fiber.json")]
    result = evaluation.evaluate(cfg, base_directory=inputs)
    assert [r["id"] for r in result["cases"]] == ["a/run0/reverse"]
    assert result["reference_selection"][0]["sha256"]
    annotation["optimization_mode"] = "lasagna"
    assert evaluation.reference_cases(annotation, "a")[0] == []
    annotation["control_points"][0][0] += .01
    with pytest.raises(RuntimeError, match="exact line point"):
        evaluation.reference_cases(annotation, "a")


def test_reference_selection_splits_at_lasagna_spans():
    # Valid annotation metadata, not a second implementation of selection.
    config = dict(step_voxels=1, cone_angle_degrees=30, cone_angle_step_degrees=5,
        cone_grid_size=3, beam_width=1, beam_prune_distance_voxels=1, beam_lookahead_steps=0,
        smoothness_weight=0, smoothness_normal_weight=0, smoothness_tangent_weight=0,
        smoothness_free_angle_degrees=0, cumulative_smoothness_steps=0,
        cumulative_smoothness_tangent_weight=0, initial_free_angle_degrees=0,
        max_step_factor=3, meeting_accept_max_error_ratio=.1, endpoint_accept_threshold_base_voxels=20)
    def metadata(mode):
        return dict(optimizer="native_fiber_trace3d", metadata_version=3, tracer_version=2,
            interp_goal=mode, interp_mode=mode, metric=None, msg="", normal_manifest="", fiber_manifest="",
            trace_to_base_scale=1, meeting_error_base_voxels=None, meeting_error_ratio=None,
            meeting_source="", failure_code="", failure_detail="", lasagna_failure_code="",
            lasagna_failure_detail="", config=config)
    points = [[i * 10, 0, 0] for i in range(6)]
    controls = [dict(position=p, segment_to_next=metadata(mode))
                for p, mode in zip(points[1:4], ["cspline", "lasagna", "cspline"])]
    controls.append(dict(position=points[4]))
    annotation = dict(type="vc3d_fiber", version=3, optimization_mode="native_fiber_trace3d",
                      line_points=points, control_points=controls)
    cases, plan = evaluation.reference_cases(annotation, "mixed")
    assert len(cases) == 4
    assert [r["line_indices"] for r in plan["runs"]] == [[1, 2], [3, 4]]
    assert plan["eligible_length_base_voxels"] == 20
    assert [e["reason"] for e in plan["excluded"]] == ["extrapolated", "extrapolated", "lasagna"]
