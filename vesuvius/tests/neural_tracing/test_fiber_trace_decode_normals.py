"""The uint8 normal decoder lives in vesuvius and is what lasagna re-exports."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from vesuvius.neural_tracing.normals import decode_normals

_LASAGNA_PYRAMID = Path(__file__).resolve().parents[3] / "lasagna" / "omezarr_pyramid.py"


def test_decode_normals_basic_values():
    nx, ny, nz = decode_normals(
        np.array([128, 255, 1], np.uint8), np.array([128, 128, 128], np.uint8)
    )
    np.testing.assert_allclose(nx, [0.0, 1.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(ny, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(nz, [1.0, 0.0, 0.0], atol=1e-6)
    assert nx.dtype == ny.dtype == nz.dtype == np.float32


def test_decode_normals_clamps_nz_outside_the_unit_circle():
    nx, ny, nz = decode_normals(np.array([255], np.uint8), np.array([255], np.uint8))
    np.testing.assert_allclose(nx, [1.0], atol=1e-6)
    np.testing.assert_allclose(ny, [1.0], atol=1e-6)
    np.testing.assert_array_equal(nz, np.array([0.0], np.float32))


def test_decode_lasagna_normals_xyz_matches_previous_cases():
    from vesuvius.neural_tracing.fiber_trace.geometry import decode_lasagna_normals_xyz

    normals, valid = decode_lasagna_normals_xyz(
        np.array([128, 128, 255, 0], dtype=np.uint8),
        np.array([128, 255, 128, 0], dtype=np.uint8),
    )
    np.testing.assert_allclose(normals[0], np.array([0.0, 0.0, 1.0]), atol=1e-6)
    np.testing.assert_allclose(normals[1], np.array([0.0, 1.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(normals[2], np.array([1.0, 0.0, 0.0]), atol=1e-6)
    assert bool(valid.all())
    assert np.isfinite(normals[3]).all()
    assert float(np.linalg.norm(normals[3])) == pytest.approx(1.0, abs=1e-6)


def _load_lasagna_pyramid():
    spec = importlib.util.spec_from_file_location(
        "_lasagna_omezarr_pyramid_under_test", _LASAGNA_PYRAMID
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not _LASAGNA_PYRAMID.exists(), reason="monorepo lasagna sources not present")
def test_lasagna_reexports_shared_decoder():
    try:
        module = _load_lasagna_pyramid()
    except ImportError as exc:  # lasagna's own optional deps
        pytest.skip(f"cannot import lasagna/omezarr_pyramid.py: {exc}")
    assert module._decode_normals is decode_normals
    rng = np.random.default_rng(0)
    nx_u8 = rng.integers(0, 256, size=(7, 9), dtype=np.uint8)
    ny_u8 = rng.integers(0, 256, size=(7, 9), dtype=np.uint8)
    for ours, theirs in zip(decode_normals(nx_u8, ny_u8), module._decode_normals(nx_u8, ny_u8)):
        np.testing.assert_array_equal(ours, theirs)
    nx_out, ny_out = module._moment_pool2x_normals(
        np.full((2, 2, 2), 128, np.uint8),
        np.full((2, 2, 2), 128, np.uint8),
    )
    assert nx_out.shape == ny_out.shape == (1, 1, 1)
    assert int(nx_out[0, 0, 0]) == 128
    assert int(ny_out[0, 0, 0]) == 128


def test_geometry_imports_without_lasagna(monkeypatch):
    blocked = [
        name
        for name in sys.modules
        if name == "lasagna"
        or name.startswith("lasagna.")
        or name == "omezarr_pyramid"
        or name.startswith("vesuvius.neural_tracing.fiber_trace")
    ]
    for name in blocked:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "lasagna", None)
    monkeypatch.setitem(sys.modules, "omezarr_pyramid", None)
    import vesuvius.neural_tracing.fiber_trace.geometry as geometry

    normals, valid = geometry.decode_lasagna_normals_xyz(
        np.array([128], np.uint8), np.array([128], np.uint8)
    )
    np.testing.assert_allclose(normals[0], [0.0, 0.0, 1.0], atol=1e-6)
    assert bool(valid[0])
