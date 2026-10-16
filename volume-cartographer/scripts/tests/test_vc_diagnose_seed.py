"""Tests for vc_diagnose_seed seed triage.

Pure-local: a small synthetic zarr volume stands in for a prediction
volume. No network, no growth binary.
"""
import os
import sys

import pytest

np = pytest.importorskip("numpy")
zarr = pytest.importorskip("zarr")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vc_diagnose_seed as triage

N = 128


@pytest.fixture()
def vol(tmp_path):
    """128^3 volume: a dense slab (x < 24, all y/z) plus one isolated
    blob (radius 5) far from the slab and from every face."""
    arr = np.zeros((N, N, N), dtype=np.uint8)
    arr[:, :, 0:24] = 255  # extended slab, dense through the growth horizon
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
    blob = (xx - 100) ** 2 + (yy - 100) ** 2 + (zz - 100) ** 2 < 25
    arr[blob] = 255
    store = zarr.open(str(tmp_path / "v.zarr"), mode="w",
                      shape=(N, N, N), chunks=(32, 32, 32), dtype="u1")
    store[:] = arr
    return triage.load_volume(str(tmp_path / "v.zarr"))


def test_healthy_seed_on_extended_sheet(vol):
    r = triage.triage_seed(vol, 12, 64, 64)
    assert r["class"] == "HEALTHY"


def test_trapped_seed_on_isolated_blob(vol):
    r = triage.triage_seed(vol, 100, 100, 100)
    assert r["class"] == "TRAPPED"


def test_void_seed_in_empty_space(vol):
    r = triage.triage_seed(vol, 100, 30, 30)
    assert r["class"] == "SEED_VOID"
    assert r["edge"] is False


def test_sparse_seed_single_voxel(tmp_path):
    arr = np.zeros((N, N, N), dtype=np.uint8)
    arr[64, 64, 64] = 255
    store = zarr.open(str(tmp_path / "s.zarr"), mode="w",
                      shape=(N, N, N), chunks=(32, 32, 32), dtype="u1")
    store[:] = arr
    v = triage.load_volume(str(tmp_path / "s.zarr"))
    r = triage.triage_seed(v, 64, 64, 64)
    assert r["class"] == "SPARSE_REGION"


def test_out_of_bounds_seed(vol):
    for seed in [(-1, 0, 0), (128, 0, 0), (0, 0, 999)]:
        r = triage.triage_seed(vol, *seed)
        assert r["class"] == "SEED_OUT_OF_BOUNDS"


def test_edge_flag_near_face(vol):
    r = triage.triage_seed(vol, 12, 64, 2)
    assert r["edge"] is True
    assert r["class"] == "HEALTHY"


def test_deterministic(vol):
    a = triage.triage_seed(vol, 12, 64, 64)
    b = triage.triage_seed(vol, 12, 64, 64)
    assert a == b
