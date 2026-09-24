"""Tests for the ``vc.fiber_trace`` bindings against a synthetic on-disk dataset.

Run with the vesuvius interpreter that has the editable volume-cartographer
install, e.g. ``.venv/bin/python -m pytest ../volume-cartographer/python/tests``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

fiber_trace = pytest.importorskip("vc.fiber_trace")

SHAPE = (16, 16, 64)  # z, y, x
START = (8.0, 8.0, 8.0)  # x, y, z
TARGET = (56.0, 8.0, 8.0)


def _write_u8_zarr(directory: Path, values: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    values = np.ascontiguousarray(values, dtype=np.uint8)
    (directory / ".zarray").write_text(json.dumps({
        "zarr_format": 2,
        "shape": list(values.shape),
        "chunks": list(values.shape),
        "dtype": "|u1",
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": ".",
    }))
    (directory / "0.0.0").write_bytes(values.tobytes())


def _manifest(groups: dict[str, str], extra: dict | None = None) -> dict:
    manifest = {
        "version": 2,
        "source_to_base": 1.0,
        "groups": {
            name: {"zarr": zarr, "scaledown": 0, "channels": [name]}
            for name, zarr in groups.items()
        },
        "base_shape_zyx": list(SHAPE),
    }
    manifest.update(extra or {})
    return manifest


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("fiber_trace_synthetic")
    full = np.full(SHAPE, 255, np.uint8)
    mid = np.full(SHAPE, 128, np.uint8)
    # Prediction: presence everywhere, fiber axis along +x (nx=1, ny=0).
    _write_u8_zarr(root / "presence.zarr", full)
    _write_u8_zarr(root / "pred_nx.zarr", full)
    _write_u8_zarr(root / "pred_ny.zarr", mid)
    (root / "pred.lasagna.json").write_text(json.dumps(_manifest({
        "presence": "presence.zarr", "nx": "pred_nx.zarr", "ny": "pred_ny.zarr",
    })))
    # Normals: sheet normal along +y (nx=0, ny=1), full gradient magnitude.
    _write_u8_zarr(root / "grad_mag.zarr", full)
    _write_u8_zarr(root / "norm_nx.zarr", mid)
    _write_u8_zarr(root / "norm_ny.zarr", full)
    (root / "normals.lasagna.json").write_text(json.dumps(_manifest(
        {"grad_mag": "grad_mag.zarr", "nx": "norm_nx.zarr", "ny": "norm_ny.zarr"},
        {"grad_mag_encode_scale": 255.0, "grad_mag_factor": 1.0},
    )))
    field = fiber_trace.open_prediction_field(str(root / "pred.lasagna.json"),
                                             cache_bytes=64 << 20, scaledown_power=0)
    normals = fiber_trace.open_normal_sampler(str(root / "normals.lasagna.json"),
                                             field.trace_to_base_scale, cache_bytes=64 << 20)
    return root, field, normals


def config(**overrides):
    values = dict(step_voxels=4.0, beam_width=8, beam_lookahead_steps=2,
                  cone_angle_degrees=25.0, cone_angle_step_degrees=5.0,
                  parallel_threads=1, trace_to_base_scale=1.0)
    values.update(overrides)
    return fiber_trace.TraceConfig(**values)


def one_way(field, normals, hook=None, **kwargs):
    planes = [fiber_trace.TargetPlane("explicit", TARGET, (1.0, 0.0, 0.0))]
    return fiber_trace.trace_one_way(field, START, TARGET, (1.0, 0.0, 0.0), planes,
                                     accept_threshold_voxels=4.0, budget_span_voxels=48.0,
                                     config=config(), normal_sampler=normals, hook=hook, **kwargs)


def test_field_scales_follow_the_manifest(dataset):
    root, field, _ = dataset
    assert field.trace_to_base_scale == 1.0
    assert field.prediction_to_base_scale == 1.0
    assert field.option_count == 1
    finer = fiber_trace.open_prediction_field(str(root / "pred.lasagna.json"), cache_bytes=8 << 20,
                                             scaledown_power=2)
    assert finer.trace_to_base_scale == pytest.approx(0.25)


def test_no_hook_reaches_target(dataset):
    _, field, normals = dataset
    result = one_way(field, normals)
    assert result.reached_target_plane
    assert result.reason == "target_plane"
    points = result.points
    assert points.shape[1] == 3 and points.dtype == np.float64
    np.testing.assert_allclose(points[0], START)
    assert points[-1, 0] == pytest.approx(TARGET[0], abs=4.0)
    np.testing.assert_allclose(points[:, 1:], 8.0, atol=1e-3)


def test_observe_and_identity_hooks_are_bit_identical(dataset):
    _, field, normals = dataset
    baseline = one_way(field, normals)
    events = []

    def observer(pool):
        events.append(pool)
        return None

    observed = one_way(field, normals, hook=observer)
    assert np.array_equal(observed.points, baseline.points)
    assert observed.reason == baseline.reason and observed.steps == baseline.steps
    assert len(events) == (baseline.steps - 1) // 2
    for index, pool in enumerate(events):
        assert pool.round == index
        assert pool.step == 2 * (index + 1)
        assert pool.phase == "trace"
        assert 8 <= len(pool) <= 32
        assert pool.path_offsets[0] == 0 and pool.path_offsets[-1] == len(pool.path_points)
        paths = pool.paths()
        assert len(paths) == len(pool)
        for path in paths:
            assert path.shape == (pool.step + 1, 3)
            np.testing.assert_allclose(path[0], START)
        assert pool.losses.dtype == np.float32 and np.all(np.diff(pool.losses) >= 0)
        assert not pool.reached.any()
        assert pool.previous_step_direction.shape == (len(pool), 3)

    identity = one_way(field, normals, hook=lambda pool: (pool.losses.copy(), False))
    assert np.array_equal(identity.points, baseline.points)
    assert identity.steps == baseline.steps

    calls = []
    every_two = one_way(field, normals, hook=lambda pool: calls.append(pool.round), hook_every_rounds=2)
    assert np.array_equal(every_two.points, baseline.points)
    assert len(calls) == (len(events) + 1) // 2


def test_hook_stop_and_rerank(dataset):
    _, field, normals = dataset
    stopped = one_way(field, normals, hook=lambda pool: (None, True))
    assert stopped.reason.startswith("hook_stop")
    assert not stopped.reached_target_plane
    assert stopped.steps == 2

    expected = {}

    def prefer_last(pool):
        expected["path"] = pool.paths()[-1].copy()
        return -np.arange(len(pool), dtype=np.float32), True

    reranked = one_way(field, normals, hook=prefer_last)
    assert reranked.reason.startswith("hook_stop")
    assert np.array_equal(reranked.points, expected["path"])


def test_hook_exceptions_propagate(dataset):
    _, field, normals = dataset

    def boom(pool):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        one_way(field, normals, hook=boom)
    with pytest.raises(ValueError):
        one_way(field, normals, hook=lambda pool: (np.zeros(len(pool) + 1, np.float32), False))
    with pytest.raises(ValueError):
        one_way(field, normals, hook_every_rounds=0, hook=lambda pool: None)


def test_segment_helpers_and_config_roundtrip(dataset):
    _, field, normals = dataset
    line = np.array([START, TARGET])
    segment = fiber_trace.trace_segment(field, line, 0, 1, config(), normal_sampler=normals)
    assert segment.accepted, (segment.reason, segment.detail)
    assert segment.fused_line.shape[1] == 3
    np.testing.assert_allclose(segment.fused_line[0], START)
    np.testing.assert_allclose(segment.fused_line[-1], TARGET)

    planes = fiber_trace.target_local_planes(field, line, 1, 0, TARGET)
    assert [plane.name for plane in planes] == ["line_prev", "inferred_direction"]
    # The inferred plane normal is the prediction axis at the target, signed
    # toward the source index, so it points back along -x here.
    np.testing.assert_allclose(planes[1].normal, (-1.0, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(planes[0].normal, (-1.0, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(planes[0].point, TARGET)
    np.testing.assert_allclose(fiber_trace.reference_tangent_toward(line, 0, 1), (1.0, 0.0, 0.0))

    cfg = config(beam_width=5, lookahead_parent_cap=7)
    as_dict = cfg.to_dict()
    assert as_dict["beam_width"] == 5 and as_dict["lookahead_parent_cap"] == 7
    assert fiber_trace.TraceConfig.from_dict(as_dict).to_dict() == as_dict
    assert fiber_trace.TraceConfig(**as_dict).to_dict() == as_dict
    with pytest.raises(AttributeError):
        fiber_trace.TraceConfig(bogus=1)


def test_extrapolation_and_whole_fiber_metric(dataset):
    _, field, normals = dataset
    tail = fiber_trace.trace_extrapolation(field, START, (1.0, 0.0, 0.0), 16.0, config(),
                                          normal_sampler=normals)
    assert tail.reached_trace_length
    assert tail.points[-1, 0] == pytest.approx(START[0] + 16.0, abs=1e-3)

    line = np.stack([np.linspace(8.0, 56.0, 49), np.full(49, 8.0), np.full(49, 8.0)], 1)
    fiber = fiber_trace.FiberInput(line, line[[0, 24, 48]], np.array([0, 24, 48]))
    metric = fiber_trace.trace_whole_fiber_metric(field, fiber, working_to_base_scale=1.0,
                                                  error_threshold_base_voxels=4.0, config=config(),
                                                  normal_sampler=normals)
    assert metric.segment_count == 2
    assert metric.restart_count == 0
    assert all(segment.success for segment in metric.segments)
    assert metric.stitched_trace.shape[1] == 3
