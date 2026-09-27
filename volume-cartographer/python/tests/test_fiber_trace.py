"""Tests for the ``vc.fiber_trace`` bindings against a synthetic on-disk dataset.

Run with the vesuvius interpreter that has the editable volume-cartographer
install, e.g. ``.venv/bin/python -m pytest ../volume-cartographer/python/tests``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import os
import vc
if os.environ.get("VC_BUILD_PYTHON"):
    vc.__path__.insert(0, str(Path(os.environ["VC_BUILD_PYTHON"]) / "vc"))
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


def test_scales_config_and_segment(dataset):
    root, field, normals = dataset
    cfg = config()
    assert field.trace_to_base_scale == 1.
    assert field.option_count == 1
    assert fiber_trace.TraceConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    finer = fiber_trace.open_prediction_field(str(root / 'pred.lasagna.json'), scaledown_power=2)
    assert finer.trace_to_base_scale == .25
    with pytest.raises(ValueError, match='must match'):
        fiber_trace.trace_segment(finer, np.array([START,TARGET]), 0, 1, cfg)
    with pytest.raises(AttributeError):
        fiber_trace.TraceConfig(unknown=3)
    result = fiber_trace.trace_segment(field, np.array([START,TARGET]), 0, 1, cfg, normals)
    assert result.accepted, result.reason
    np.testing.assert_allclose(result.fused_line[[0,-1]], [START,TARGET])
    # Array ownership must survive both its result and subsequent calls.
    import gc
    points = result.fused_line
    expected = points.copy()
    del result
    gc.collect()
    fiber_trace.trace_segment(field, np.array([START,TARGET]), 0, 1, cfg, normals)
    np.testing.assert_array_equal(points, expected)


def test_extrapolate_without_sheet_normals(dataset):
    _, field, _ = dataset
    cfg = config(smoothness_normal_weight=0., smoothness_tangent_weight=0., cumulative_smoothness_tangent_weight=0.)
    for sign in [-1., 1.]:
        result = fiber_trace.trace_extrapolation(field, (32.,8.,8.), (sign,0.,0.), 16., cfg)
        assert result.reached_trace_length, result.reason
        np.testing.assert_allclose(result.points[-1], [32.+sign*16.,8.,8.], atol=1e-5)
    for distance, direction in [(-1., (1.,0.,0.)), (16.,(0.,0.,0.)), (float('nan'),(1.,0.,0.))]:
        with pytest.raises(ValueError):
            fiber_trace.trace_extrapolation(field, START, direction, distance, cfg)
    with pytest.raises(ValueError):
        fiber_trace.trace_segment(field, np.array([START,TARGET]), 0, 2, cfg)
    with pytest.raises(ValueError):
        fiber_trace.trace_segment(field, np.array([START,(float('nan'),8.,8.)]), 0, 1, cfg)


def test_native_direction_decoding_and_missing_data():
    directions = fiber_trace.decode_directions(np.array([255,128,128,0,255],np.uint8),
                                               np.array([128,255,128,128,255],np.uint8))
    np.testing.assert_allclose(directions[:4], [[1,0,0],[0,1,0],[0,0,1],[0,0,0]], atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(directions[4]), 1.)
    with pytest.raises(ValueError):
        fiber_trace.decode_directions(np.ones(1,np.uint8), np.ones(2,np.uint8))


def test_extrapolation_validates_config_and_required_normals(dataset):
    _, field, _ = dataset
    with pytest.raises(ValueError, match='normal sampler is required'):
        fiber_trace.trace_extrapolation(field, START, (1.,0.,0.), 16., config())
    with pytest.raises(ValueError, match='step voxels must be positive'):
        fiber_trace.trace_extrapolation(field, START, (1.,0.,0.), 16., config(step_voxels=0.))
