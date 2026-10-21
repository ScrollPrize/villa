"""Native chunk API, ownership, concurrent calls and optional cached real data."""
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from test_fiber_trace import fiber_trace as ft, _write_u8_zarr, _manifest


def chunk_config():
    c = ft.ChunkConfig()
    c.core_base, c.halo_base, c.grid_base, c.nms_base = 64, 16, 4, 4
    c.tube_half_base, c.tube_step_base = 8, 4
    c.rim_normal_base = c.rim_tangent_base = 4
    c.join_minimum_base = 8
    return c


def trace_config():
    return ft.TraceConfig(step_voxels=16, beam_width=3, beam_lookahead_steps=2,
                          cone_angle_degrees=0, parallel_threads=1,
                          smoothness_weight=0, smoothness_normal_weight=0,
                          smoothness_tangent_weight=0, cumulative_smoothness_tangent_weight=0,
                          replay_presence_cutoff=.1, replay_presence_modulated=True)


@pytest.fixture
def inputs(tmp_path):
    shape = (32, 32, 256)
    z, y, _ = np.indices(shape)
    channels = {
        "presence": np.rint(255 * np.exp(-((y-16)**2+(z-16)**2)/8)),
        "nx": np.full(shape, 255), "ny": np.full(shape, 128),
        "grad_mag": np.full(shape, 255), "normal_nx": np.full(shape, 128),
        "normal_ny": np.full(shape, 255),
    }
    for name, data in channels.items():
        _write_u8_zarr(tmp_path / name, data)
    pred = _manifest({name: name for name in ("presence", "nx", "ny")},
                     {"base_shape_zyx": list(shape)})
    # Width channels retain physical stored values, unlike compact u8 directions.
    width = tmp_path / "width"
    _write_u8_zarr(width, np.full(shape, 20))
    metadata = json.loads((width / ".zarray").read_text())
    metadata["dtype"] = "<f4"
    (width / ".zarray").write_text(json.dumps(metadata))
    (width / "0.0.0").write_bytes(np.full(shape, 20, dtype="<f4").tobytes())
    pred["groups"]["width"] = {"zarr": "width", "scaledown": 0, "channels": ["width"]}
    (tmp_path / "pred.json").write_text(json.dumps(pred))
    norm = _manifest({"nx": "normal_nx", "ny": "normal_ny", "grad_mag": "grad_mag"},
                     {"grad_mag_encode_scale": 255., "grad_mag_factor": 1., "base_shape_zyx": list(shape)})
    (tmp_path / "norm.json").write_text(json.dumps(norm))
    return tmp_path


def handles(root):
    return (ft.open_prediction_field(str(root / "pred.json"), scaledown_power=0, enable_width=True),
            ft.open_normal_sampler(str(root / "norm.json"), 1))


def test_chunk_lifetimes_concurrency_and_reload(inputs):
    field, normal = handles(inputs)
    trace, chunks = trace_config(), chunk_config()
    trace.width_relative_tolerance = .4
    trace.width_cosine_weight = True
    chunks.join_width_weight = True
    assert ft.TraceConfig.from_dict(trace.to_dict()).to_dict() == trace.to_dict()
    cache = ft.ChunkCache(field, normal, trace, chunks, str(inputs / "cache"), "tube-v1")
    # Cache copies configuration, but must retain the field and normal owners.
    trace.step_voxels = 0
    chunks.core_base = 0
    del trace, chunks, field, normal
    gc.collect()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(cache.get, [(0, 0, 0)] * 4))
    assert results[0].seeds > 0
    assert results[0].key == [0, 0, 0]
    expected = results[0].lines_base
    for other in results[1:]:
        for a, b in zip(expected, other.lines_base, strict=True):
            np.testing.assert_array_equal(a, b)
    route = cache.route([16, 16, 16], [1, 0, 0], np.eye(3), 160, 100)
    assert route["reason"] == "chunk_length_limit"
    assert len(route["joins"]) >= 2
    assert route["points_base"][-1, 0] == pytest.approx(176)
    with pytest.raises(ValueError, match="shape"):
        cache.route([16, 16, 16], [1, 0, 0], np.ones((2, 3)), 160, 100)
    with pytest.raises(ValueError, match="positive definite"):
        cache.route([16, 16, 16], [1, 0, 0], -np.eye(3), 160, 100)
    del results, cache
    gc.collect()
    field, normal = handles(inputs)
    trace, chunks = trace_config(), chunk_config()
    trace.width_relative_tolerance, trace.width_cosine_weight = .4, True
    chunks.join_width_weight = True
    disk = ft.ChunkCache(field, normal, trace, chunks, str(inputs / "cache"), "tube-v1")
    for a, b in zip(expected, disk.get([0, 0, 0]).lines_base, strict=True):
        np.testing.assert_array_equal(a, b)
    wrong = ft.ChunkCache(field, normal, trace, chunks, str(inputs / "cache"), "other-model")
    with pytest.raises(RuntimeError, match="identity"):
        wrong.get([0, 0, 0])
    mismatch = ft.open_normal_sampler(str(inputs / "norm.json"), 2)
    with pytest.raises(ValueError, match="scales differ"):
        ft.ChunkCache(field, mismatch, trace, chunks, str(inputs / "other"), "v1")


def test_chunk_replay_and_reverse_checks(inputs):
    field, normal = handles(inputs)
    cache = ft.ChunkCache(field, normal, trace_config(), chunk_config(), str(inputs / "cache"), "tube")
    cfg = ft.ChunkReplayConfig()
    cfg.error_threshold_base = 2
    reference = np.array([[16., 16., 16.], [220., 16., 16.]])
    result = cache.replay(reference, normal, cfg)
    assert result.completed_reference_arc_base == 204
    assert result.failures == [], [(f.reason, f.reference_arc_base) for f in result.failures]
    assert len(result.segments) == 1
    filtered = cache.replay(reference, normal, cfg, reverse_interval_base=32)
    assert filtered.completed_reference_arc_base == 204
    assert filtered.reverse_checks > 0
    assert filtered.failures == []
    # Tight reverse checks can reject a splice with unequal vertex spacing.
    # These must be reported as filtered spans, not GT-distance failures.
    assert filtered.filtered_stops
    assert filtered.segments[0].termination_reason == "reverse_consistency_filtered"
    segment = result.segments[0]
    assert segment.matches[-1].threshold_measurement.threshold_error_ratio == pytest.approx(0, abs=1e-10)
    points = segment.points_base
    del result, segment, cache, field
    gc.collect()
    assert points.shape[1] == 3 and np.isfinite(points).all()
    field, normal = handles(inputs)
    cache = ft.ChunkCache(field, normal, trace_config(), chunk_config(), str(inputs / "cache"), "tube")
    for kwargs in ({"begin_arc_base": -1}, {"end_arc_base": 1000}, {"reverse_interval_base": -1}):
        with pytest.raises(ValueError):
            cache.replay(reference, normal, cfg, **kwargs)


def test_real_paris4_chunk_replay(tmp_path):
    names = ["VC_TEST_WIDTH_MANIFEST", "VC_TEST_WIDTH_FIBER", "VC_TEST_REVERSE_NORMAL_MANIFEST"]
    if not all(os.environ.get(name) for name in names):
        pytest.skip("set the three documented cached Paris4 fixture paths")
    manifest, fiber_path, normal_manifest = (os.environ[name] for name in names)
    fiber = json.loads(Path(fiber_path).read_text())
    # The fixture's VC3D line coordinates are already base XYZ.
    line = np.asarray(fiber["line_points"], dtype=np.float64)
    first_cp = np.asarray(fiber["control_points"][0]["position"])
    first_index = np.argmin(np.linalg.norm(line-first_cp, axis=1))
    line = np.ascontiguousarray(line[first_index:])
    field = ft.open_prediction_field(manifest, scaledown_power=0, enable_width=True)
    normal = ft.open_normal_sampler(normal_manifest, field.trace_to_base_scale)
    c = chunk_config()
    c.halo_base, c.grid_base, c.nms_base = 32, 8, 8
    c.tube_half_base, c.tube_step_base = 16, 8
    c.seed_presence, c.tube_presence, c.rim_drop = .4, .2, .05
    c.rim_normal_base, c.rim_tangent_base = 16, 32
    c.join_minimum_base = 32
    t = trace_config()
    t.trace_to_base_scale = field.trace_to_base_scale
    t.step_voxels = 16 / field.trace_to_base_scale
    t.cone_angle_degrees, t.cone_angle_step_degrees = 25, 15
    t.width_relative_tolerance, t.width_cosine_weight = .4, True
    cache = ft.ChunkCache(field, normal, t, c, str(tmp_path / "cache"), "paris4-binding-test")
    if hasattr(ft, "case_queue_supported") and ft.case_queue_supported():
        key = np.floor(line[0] / c.core_base).astype(int).tolist()
        built = cache.get_many([key, key], workers=2, contexts_per_worker=2)
        assert built[0].seeds > 0
        for a, b in zip(built[0].lines_base, built[1].lines_base, strict=True):
            np.testing.assert_array_equal(a, b)
    replay = ft.ChunkReplayConfig()
    replay.tangential_threshold_factor = 2
    result = cache.replay(line, normal, replay, end_arc_base=256)
    if hasattr(ft, "EvaluationControl") and ft.case_queue_supported():
        from vc.fiber_evaluation import replay_dict
        control = ft.EvaluationControl()
        queued = cache.replay_many([ft.EvaluationCase(line, end_arc_base=256)] * 2,
                                  normal, replay, control, workers=2, contexts_per_worker=2)
        assert control.status()["completed"] == 2
        assert all(replay_dict(r) == replay_dict(result) for r in queued)
        from vc.fiber_evaluation import evaluate, CHUNK_KEYS
        report = evaluate(dict(version=1, prediction_manifest=manifest,
            normal_manifest=normal_manifest, source_identity="paris4-binding-test",
            chunk_cache=str(tmp_path / "cache"), enable_width=True,
            trace=t.to_dict(), chunks={key: getattr(c, key) for key in CHUNK_KEYS},
            replay=dict(tangential_threshold_factor=2),
            cases=[dict(id="paris4-forward", points_base_xyz=line.tolist(), end_arc_base=256)]))
        assert {k: v for k, v in report["cases"][0].items() if k != "id"} == replay_dict(result)
        print("Paris4 queued evaluation: 2 cases exactly equal to serial replay")
        direct_case = ft.EvaluationCase(line, end_arc_base=256)
        direct = ft.beam_replay(field, normal, normal, t, direct_case, replay)
        batch = ft.beam_replay_many(field, normal, normal, t, [direct_case] * 2,
                                   replay, ft.EvaluationControl(), 2, 2)
        assert all(replay_dict(r) == replay_dict(direct) for r in batch)
        # Shared annotation selector retains exact source coordinates and arcs.
        from vc.fiber_evaluation import reference_cases
        selected, plan = reference_cases(fiber, "paris4")
        assert selected and plan["eligible_length_base_voxels"] > 256
        direct_report = evaluate(dict(version=1, backend="beam", prediction_manifest=manifest,
            normal_manifest=normal_manifest, source_identity="paris4-direct", enable_width=True,
            trace=t.to_dict(), replay=dict(tangential_threshold_factor=2),
            cases=[dict(id="paris4", points_base_xyz=line.tolist(), end_arc_base=256)]))
        assert {k: v for k, v in direct_report["cases"][0].items() if k != "id"} == replay_dict(direct)
        from vc.fiber_evaluation_vis import render_html, write_gallery
        assert render_html(direct_report).count('<svg ') == 3
        if directory := os.environ.get("VC_TEST_EVALUATION_GALLERY"):
            write_gallery(direct_report, Path(directory) / "paris4-beam.html")
            write_gallery(report, Path(directory) / "paris4-chunk.html")
            (Path(directory) / "paris4-beam.json").write_text(json.dumps(direct_report))
            print(f"Paris4 offline geometry galleries: {directory}")
        print(f"Paris4 direct beam: batch/callable exactly serial; {len(direct.segments)} segments, "
              f"{len(direct.failures)} failures; {len(selected)} selected directions")
    assert len(result.segments) == 1
    assert result.failures == []
    assert result.completed_reference_arc_base == 256
    disk = ft.ChunkCache(field, normal, t, c, str(tmp_path / "cache"), "paris4-binding-test")
    again = disk.replay(line, normal, replay, end_arc_base=256)
    assert len(result.failures) == len(again.failures)
    for a, b in zip(result.segments, again.segments, strict=True):
        np.testing.assert_array_equal(a.points_base, b.points_base)
    print(f"Paris4 Python replay: {len(result.segments)} segments, {len(result.failures)} failures; disk exact")
