"""Regression tests for simple TIFXYZ writer input validation.

Run from repository root:
  python -m pytest spiral-fitting/tests/test_tifxyz_failfast.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tifxyz import save_tifxyz


def grid():
    return np.array([[[1., 2., 3.], [4., 5., 6.]],
                     [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_fails_without_output(tmp_path, bad):
    z = grid()
    z[0, 0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        save_tifxyz(z, tmp_path, "surface", 1, 1, "test")
    assert not (tmp_path / "surface").exists()


@pytest.mark.parametrize("bad", [0, -1, np.inf, np.nan])
def test_invalid_step_size_fails_without_output(tmp_path, bad):
    with pytest.raises(ValueError, match="step_size"):
        save_tifxyz(grid(), tmp_path, "surface", bad, 1, "test")
    assert not (tmp_path / "surface").exists()


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, 2, 2), (2, 2)])
def test_invalid_shape_fails_without_output(tmp_path, shape):
    with pytest.raises(ValueError, match="shape"):
        save_tifxyz(np.zeros(shape, dtype=np.float32), tmp_path, "surface", 1, 1, "test")
    assert not (tmp_path / "surface").exists()


def test_valid_roundtrip_and_strict_json(tmp_path):
    z = grid()
    assert save_tifxyz(z, tmp_path, "surface", 1, 1, "test")
    root = tmp_path / "surface"
    metadata = json.loads((root / "meta.json").read_text(),
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    assert metadata["bbox"] == [[3., 2., 1.], [12., 11., 10.]]
    for k, i in [("z", 0), ("y", 1), ("x", 2)]:
        with Image.open(root / (k + ".tif")) as im:
            np.testing.assert_array_equal(np.asarray(im), z[..., i])


def test_existing_partial_sentinel_semantics_unchanged(tmp_path):
    z = grid()
    z[1, 1] = [-1., 10., -1.]
    save_tifxyz(z, tmp_path, "surface", 1, 1, "test")
    metadata = json.loads((tmp_path / "surface" / "meta.json").read_text())
    assert metadata["bbox"][0] == [-1., 2., -1.]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0])
def test_invalid_voxel_size_fails_without_output(tmp_path, bad):
    with pytest.raises(ValueError, match="voxel_size_um"):
        save_tifxyz(grid(), tmp_path, "surface", 1, bad, "test")
    assert not (tmp_path / "surface").exists()


def test_unserializable_source_fails_without_output(tmp_path):
    with pytest.raises(TypeError):
        save_tifxyz(grid(), tmp_path, "surface", 1, 1, object())
    assert not (tmp_path / "surface").exists()


def test_nonfinite_metadata_fails_without_output(tmp_path):
    with pytest.raises(ValueError):
        save_tifxyz(grid(), tmp_path, "surface", 1, 1, {"invalid": float("nan")})
    assert not (tmp_path / "surface").exists()


def test_float32_overflow_fails_without_output(tmp_path):
    z = grid().astype(np.float64)
    z[0, 0, 0] = 1e100
    with pytest.raises(ValueError, match="non-finite"):
        save_tifxyz(z, tmp_path, "surface", 1, 1, "test")
    assert not (tmp_path / "surface").exists()
