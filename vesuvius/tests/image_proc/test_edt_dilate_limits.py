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


def test_scipy_rint_recovers_integer_squares(monkeypatch):
    # Through the production helper: a real scipy EDT with squared distances past 2**24 ...
    inverted = np.ones((1, 1, 5000), bool)
    inverted[0, 0, 0] = False
    got = m._scipy_edtsq(inverted)[0, 0]
    assert got.dtype == np.float64 and np.array_equal(got, np.arange(5000.0) ** 2)
    # ... and the helper's square/rint step on float64 distances sqrt(k) up to the 2**51 bound
    # (regions that large cannot be allocated, so feed scipy's output directly).
    k = np.array(
        [2**24 + 1, 2**40 + 12345, 2**48 - 1, 3 * (2**23) ** 2, 2**51 - 1, 2**51 - 3, 2**50 + 1],
        dtype=np.float64,
    )
    monkeypatch.setattr(m, "scipy_edt", lambda a: np.sqrt(k))
    assert np.array_equal(m._scipy_edtsq(None), k)


def test_rint_bound_is_enforced():
    # Above 2**51 rint(fl(sqrt(k))**2) does return wrong integers, e.g. here:
    bad = np.arange(2**51, 2**51 + 4096, dtype=np.float64)
    assert (np.rint(np.square(np.sqrt(bad))) != bad).any()
    m._check_backend((2**25, 2**24, 1))  # (2**50 + 2**48) < 2**51: accepted
    with pytest.raises(ValueError, match="2\\*\\*51"):
        m._check_backend((2**25, 2**25, 1))  # exactly 2**51


def test_missing_scipy_large_region_raises_clear_error(tmp_path, monkeypatch):
    # edt installed, scipy not: regions past edt's float32 range have no backend.
    monkeypatch.setattr(m, "HAS_EDT", True)
    monkeypatch.setattr(m, "scipy_edt", None)
    monkeypatch.setattr(m, "_FLOAT32_EXACT_INT", 2)
    vol = np.zeros((4, 4, 4), np.uint8)
    vol[0, 0, 0] = 1
    with pytest.raises(ImportError, match="needs scipy"):
        _run(tmp_path, monkeypatch, vol, 1.0)
    cfg = m.EdtDilateConfig(
        input_zarr=str(tmp_path / "in.zarr"), output_zarr=str(tmp_path / "out2.zarr"),
        num_workers=1, distance=1.0, chunk_size=(4, 4, 4),
    )
    with pytest.raises(ImportError, match="needs scipy"):
        m.EdtDilateTask(cfg).prepare()
    assert not (tmp_path / "out2.zarr").exists()  # checked before creating output


def test_missing_scipy_import_defines_none(monkeypatch):
    # Simulate an environment without scipy: the module must still import with edt.
    import builtins, importlib, sys
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("scipy"):
            raise ImportError("simulated: no scipy")
        return real(name, *a, **k)

    pytest.importorskip("edt")
    monkeypatch.setattr(builtins, "__import__", fake)
    for mod in [k for k in sys.modules if k == "scipy" or k.startswith("scipy.")]:
        monkeypatch.delitem(sys.modules, mod)
    name = m.__name__
    monkeypatch.delitem(sys.modules, name)
    try:
        fresh = importlib.import_module(name)
        assert fresh.HAS_EDT and fresh.scipy_edt is None
    finally:
        monkeypatch.setattr(builtins, "__import__", real)
        sys.modules[name] = m


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


def test_prepare_warning_counts_huge_regions_exactly(tmp_path, monkeypatch):
    # (2**22)**3 = 2**66 voxels: np.prod(..., dtype=int64) wraps to 0 and never warned.
    n = 2**22
    _create(zarr.open_group(str(tmp_path / "in.zarr"), mode="w"), "0",
            shape=(n, n, n), chunks=(n, n, n), dtype=np.uint8, fill_value=0)
    cfg = m.EdtDilateConfig(
        input_zarr=str(tmp_path / "in.zarr"), output_zarr=str(tmp_path / "out.zarr"),
        num_workers=1, distance=0.0, chunk_size=(n, n, n),
    )

    class _Stop(Exception):
        pass

    class _StopBlosc:
        BITSHUFFLE = 2

        def __init__(self, *a, **k):
            raise _Stop

    monkeypatch.setattr(m, "Blosc", _StopBlosc)
    with pytest.warns(RuntimeWarning, match=str(2**66)), pytest.raises(_Stop):
        m.EdtDilateTask(cfg).prepare()
