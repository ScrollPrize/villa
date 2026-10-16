"""The fiber_trace normal decoder must match lasagna's, and must not require lasagna to be installed."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace.geometry import _decode_normals_u8

_LASAGNA_PYRAMID = Path(__file__).resolve().parents[3] / "lasagna" / "omezarr_pyramid.py"


def test_decode_normals_u8_basic_values():
    nx, ny, nz = _decode_normals_u8(np.array([128, 255, 1], np.uint8), np.array([128, 128, 128], np.uint8))
    np.testing.assert_allclose(nx, [0.0, 1.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(ny, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(nz, [1.0, 0.0, 0.0], atol=1e-6)
    assert nx.dtype == ny.dtype == nz.dtype == np.float32


@pytest.mark.skipif(not _LASAGNA_PYRAMID.exists(), reason="monorepo lasagna sources not present")
def test_decode_normals_u8_matches_lasagna():
    spec = importlib.util.spec_from_file_location("_lasagna_omezarr_pyramid_under_test", _LASAGNA_PYRAMID)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # lasagna's own optional deps
        pytest.skip(f"cannot import lasagna/omezarr_pyramid.py: {exc}")
    rng = np.random.default_rng(0)
    nx_u8 = rng.integers(0, 256, size=(7, 9), dtype=np.uint8)
    ny_u8 = rng.integers(0, 256, size=(7, 9), dtype=np.uint8)
    for ours, theirs in zip(_decode_normals_u8(nx_u8, ny_u8), module._decode_normals(nx_u8, ny_u8)):
        np.testing.assert_array_equal(ours, theirs)


def test_geometry_imports_without_lasagna(monkeypatch):
    for name in [n for n in sys.modules if n == "lasagna" or n.startswith("lasagna.") or n == "omezarr_pyramid"
                 or n.startswith("vesuvius.neural_tracing.fiber_trace")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "lasagna", None)
    monkeypatch.setitem(sys.modules, "omezarr_pyramid", None)
    import vesuvius.neural_tracing.fiber_trace.geometry as geometry

    assert geometry._lasagna_decode_normals is geometry._decode_normals_u8
