"""Affine crops share point sampling's interpolation, caches and boundaries."""

import pickle

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("vc_reader._native")

from test_points import _linear, array, data  # noqa: F401, E402
from test_zarr_reader import BLOSC, CHUNKS, MISSING, _serve, _write_zarr  # noqa: E402
from vc_reader.reader import ZarrReader  # noqa: E402


def coordinates(shape, origin, axes):
    indices = np.indices(shape, dtype=np.float64)
    return (np.asarray(origin) + indices[0, ..., None] * np.asarray(axes)[:, 0]
            + indices[1, ..., None] * np.asarray(axes)[:, 1]
            + indices[2, ..., None] * np.asarray(axes)[:, 2])


@pytest.mark.parametrize("read_mode", ["mmap", "slab"])
@pytest.mark.parametrize("memory", [0, 8 << 20])
@pytest.mark.parametrize("order", [0, 1])
def test_affine_geometry(array, read_mode, memory, order):
    root, exp = array
    # Identity, reflected/permuted axes, oblique anisotropic crop, shear,
    # plane, repeated point, empty crop and completely outside the array.
    grids = [
        ((11, 13, 17), (10, 12, 14), np.eye(3)),
        ((7, 9, 11), (35, 48, 43), [[0, 0, -1], [0, -2, 0], [-1, 0, 0]]),
        ((17, 19, 21), (-0.5, 10.3, 20.7), [[0.8, -0.4, 0.2], [0.3, 1.2, -0.5], [0.1, 0.6, 0.9]]),
        ((1, 21, 19), (13.2, 9.5, 15.1), [[0, 0.3, -0.1], [0, 1, 0.2], [0, 0.4, 1]]),
        ((2, 3, 4), (-0.5, 0.5, 1.5), np.zeros((3, 3))),
        ((0, 3, 4), (0, 0, 0), np.eye(3)),
        ((2, 3, 4), (100, 200, 300), np.eye(3)),
    ]
    dtype = np.float64 if order else None
    with ZarrReader.open(root, read_mode=read_mode, memory_cache=memory) as r:
        results = r.read_affine_many(iter(grids), order=order, dtype=dtype)
        for got, (shape, origin, axes) in zip(results, grids):
            pts = coordinates(shape, origin, axes)
            assert got.shape == shape
            assert got.dtype == (np.float64 if order else exp.dtype)
            np.testing.assert_allclose(got, r.sample(pts, order=order, dtype=np.float64), rtol=1e-12)
            if order:
                np.testing.assert_allclose(got.ravel(), _linear(exp, pts), rtol=1e-12)
        # Changing an output never changes a cached chunk.
        results[0].fill(0)
        np.testing.assert_array_equal(r.read_affine(*grids[0], order=0), exp[10:21, 12:25, 14:31])


@pytest.mark.parametrize("dtype", ["u1", ">u2", "<i4", "f4", ">f8", "bool"])
def test_affine_dtypes_and_out(tmp_path, dtype):
    pytest.importorskip("numcodecs")
    source = np.arange(9*11*13).reshape(9, 11, 13).astype(dtype)
    root = str(tmp_path / "a.zarr")
    _write_zarr(root, source, (3, 4, 5))
    grid = ((5, 6, 7), (-0.5, 2.25, 1.5), np.eye(3))
    pts = coordinates(*grid)
    with ZarrReader.open(root) as r:
        for order, output_type in [(0, r.dtype), (1, np.float32), (1, np.float64)]:
            batch = np.empty((2, *grid[0]), output_type)
            out = batch[0]
            assert r.read_affine(*grid, order=order, dtype=output_type, out=out) is out
            np.testing.assert_allclose(out, r.sample(pts, order=order, dtype=output_type))
            outputs = [batch[0], batch[1]]
            got = r.read_affine_many([grid, grid], order=order, dtype=output_type, out=outputs)
            assert all(a is b for a, b in zip(got, outputs))
            np.testing.assert_array_equal(got[0], got[1])
        assert r.read_affine(*grid).dtype == np.float32
        assert r.read_affine_many([]) == []


def test_affine_rejects_bad_inputs(array):
    root, _ = array
    grid = ((2, 3, 4), (0, 0, 0), np.eye(3))
    with ZarrReader.open(root) as r:
        for shape in [(2, 3), (-1, 3, 4), (2.5, 3, 4), (2**62, 3, 4)]:
            with pytest.raises(ValueError, match="shape"):
                r.read_affine(shape, grid[1], grid[2])
        for origin in [(0, 0), (0, np.nan, 0), (0, np.inf, 0)]:
            with pytest.raises(ValueError, match="origin"):
                r.read_affine(grid[0], origin, grid[2])
        for axes in [np.eye(2), np.full((3, 3), np.nan)]:
            with pytest.raises(ValueError, match="axes"):
                r.read_affine(grid[0], grid[1], axes)
        for kwargs in [dict(order=2), dict(dtype="f2"), dict(order=0, dtype="f4")]:
            with pytest.raises(ValueError):
                r.read_affine(*grid, **kwargs)
        readonly = np.empty(grid[0], np.float32)
        readonly.flags.writeable = False
        unaligned = np.ndarray(grid[0], np.float32, buffer=bytearray(97), offset=1)
        for out in [readonly, unaligned, np.empty(grid[0], np.float64), np.empty((2, 3, 8), np.float32)[..., ::2]]:
            with pytest.raises(ValueError, match="out"):
                r.read_affine(*grid, out=out)
        with pytest.raises(ValueError, match="one array"):
            r.read_affine_many([grid], out=[])
        out = np.full(grid[0], -99, np.float32)
        with pytest.raises(ValueError, match="overlap"):
            r.read_affine_many([grid, grid], out=[out, out])
        with pytest.raises(ValueError, match="origin"):
            r.read_affine_many([grid, (grid[0], (0, 0), grid[2])], out=[out, None])
        assert np.all(out == -99)  # validation finishes before any output is written


def test_affine_requires_3d(tmp_path):
    pytest.importorskip("numcodecs")
    root = str(tmp_path / "2d.zarr")
    _write_zarr(root, np.zeros((4, 5), np.uint8), (2, 3))
    with ZarrReader.open(root) as r:
        with pytest.raises(ValueError, match="3-D"):
            r.read_affine((1, 2, 3), (0, 0, 0), np.eye(3))


@pytest.mark.parametrize("order", [0, 1])
def test_affine_spans_point_batches(array, order):
    root, _ = array
    grid = ((65, 129, 129), (-0.3, 1.5, 2.5), np.diag([0.5, 0.3, 0.3]))
    with ZarrReader.open(root) as r:
        got = r.read_affine(*grid, order=order)
        np.testing.assert_allclose(got, r.sample(coordinates(*grid), order=order), rtol=1e-6)


def test_affine_http_and_pickle(tmp_path, data):
    _write_zarr(str(tmp_path / "a.zarr"), data, CHUNKS, compressor=BLOSC, missing=MISSING)
    grid = ((15, 17, 19), (12.3, 13.5, 14.7), np.eye(3))
    server, base = _serve(tmp_path)
    try:
        with ZarrReader.open(f"{base}/a.zarr", cache_dir=tmp_path / "cache") as r:
            expected = r.sample(coordinates(*grid))
            np.testing.assert_allclose(r.read_affine(*grid), expected)
            with pickle.loads(pickle.dumps(r)) as child:
                np.testing.assert_allclose(child.read_affine(*grid), expected)
            r.flush()
        with ZarrReader.open(f"{base}/a.zarr", cache_dir=tmp_path / "cache", offline=True) as r:
            np.testing.assert_allclose(r.read_affine(*grid), expected)
    finally:
        server.shutdown()
