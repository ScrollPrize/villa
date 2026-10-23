"""Large distances: float32 exactness of edt.edtsq, scipy rounding, and the halo bound."""
import numpy as np
import pytest
import zarr

pytest.importorskip("scipy.ndimage")

from vesuvius.image_proc.run.zarr_tasks.tasks import edt_dilate as m
from .test_edt_dilate_backends import BACKENDS, _create


def _run(tmp_path, monkeypatch, vol, dist, black_border=True):
    _create(zarr.open_group(str(tmp_path / "in.zarr"), mode="w"), "0", data=vol, chunks=vol.shape)
    out = zarr.open_group(str(tmp_path / "out.zarr"), mode="w")
    _create(out, "0", shape=vol.shape, chunks=vol.shape, dtype=np.uint8, fill_value=0)
    for k, v in dict(
        input_path=str(tmp_path / "in.zarr"), output_path=str(tmp_path / "out.zarr"), resolution="0"
    ).items():
        monkeypatch.setitem(m._WORKER_STATE, k, v)
    z, y, x = vol.shape
    m._edt_dilate_worker(((0, 0, 0), (0, z, 0, y, 0, x), dist, black_border))
    return out["0"][:]


@pytest.mark.parametrize("backend", BACKENDS)
def test_squared_distance_above_float32_exact_range(tmp_path, monkeypatch, backend):
    # 4097**2 = 2**24 + 2**13 + 1 is odd and > 2**24, so float32 rounds it down to
    # 4097**2 - 1. With distance**2 = 4097**2 - 0.5 the voxel at 4097 must stay 0;
    # thresholding edt.edtsq's float32 output would set it.
    monkeypatch.setattr(m, "HAS_EDT", backend == "edt")
    n = 4097
    assert float(np.float32(n * n)) == n * n - 1
    vol = np.zeros((1, 1, n + 3), np.uint8)
    vol[0, 0, 0] = 1
    dist = float(np.sqrt(n * n - 0.5))
    assert n - 1 < dist < n and (n - 1) ** 2 <= np.float64(dist) ** 2 < n * n
    got = _run(tmp_path, monkeypatch, vol, dist)[0, 0]
    assert got[: n].all() and not got[n:].any()


def test_scipy_rint_recovers_integer_squares():
    # float64 sqrt -> square -> rint is exact for integer squares far beyond any region.
    k = np.array([2**24 + 1, 2**40 + 12345, 2**48 - 1, 3 * (2**23) ** 2], dtype=np.float64)
    assert np.array_equal(np.rint(np.square(np.sqrt(k))), k)


def test_halo_is_clipped_to_volume():
    assert m._halo(3.2, (100, 100, 100)) == 4
    assert m._halo(1e6, (10, 20, 30)) == 30
    assert m._halo(1e300, (10, 20, 30)) == 30


@pytest.mark.parametrize("backend", BACKENDS)
def test_huge_distance_fills_volume(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(m, "HAS_EDT", backend == "edt")
    vol = np.zeros((5, 6, 7), np.uint8)
    vol[0, 0, 0] = 1
    assert _run(tmp_path, monkeypatch, vol, 1e6).all()


def test_prepare_warns_on_large_region(tmp_path, monkeypatch):
    vol = np.zeros((8, 8, 8), np.uint8)
    _create(zarr.open_group(str(tmp_path / "in.zarr"), mode="w"), "0", data=vol, chunks=vol.shape)
    cfg = m.EdtDilateConfig(
        input_zarr=str(tmp_path / "in.zarr"), output_zarr=str(tmp_path / "out.zarr"), num_workers=1,
        distance=1e6, chunk_size=(2, 2, 2),
    )

    class _Stop(Exception):
        pass

    class _StopBlosc:  # the check runs before any output is created
        BITSHUFFLE = 2

        def __init__(self, *a, **k):
            raise _Stop

    monkeypatch.setattr(m, "Blosc", _StopBlosc)
    monkeypatch.setattr(m, "_REGION_VOXEL_WARN", 8**3 - 1)
    with pytest.warns(RuntimeWarning, match="halo of 8"), pytest.raises(_Stop):
        m.EdtDilateTask(cfg).prepare()
    monkeypatch.setattr(m, "_REGION_VOXEL_WARN", 8**3)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(_Stop):
            m.EdtDilateTask(cfg).prepare()
