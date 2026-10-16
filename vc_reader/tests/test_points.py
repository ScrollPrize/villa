"""ZarrReader.read_points, integer-array indexing and sample() (nearest and n-linear)."""

from __future__ import annotations

import pickle
import platform

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("vc_reader._native", reason="the native extension is not built")

from test_zarr_reader import BLOSC, CHUNKS, MISSING, SHAPE, _expected, _serve, _write_zarr  # noqa: E402

from vc_reader.reader import ZarrReader  # noqa: E402

FILL = 7


@pytest.fixture(scope="module")
def data():
    return np.random.default_rng(1).integers(0, 255, SHAPE, dtype=np.uint8)


@pytest.fixture(scope="module", params=["raw", "blosc"])
def array(request, tmp_path_factory, data):
    pytest.importorskip("numcodecs")
    root = str(tmp_path_factory.mktemp(request.param) / "a.zarr")
    _write_zarr(root, data, CHUNKS, missing=MISSING, compressor=BLOSC if request.param == "blosc" else None)
    return root, _expected(data, MISSING, CHUNKS)


def _gather(exp, pts):
    """Reference for read_points: stored values, fill outside."""
    pts = np.asarray(pts).reshape(-1, exp.ndim)
    inside = np.all((pts >= 0) & (pts < np.asarray(exp.shape)), axis=1)
    out = np.full(len(pts), FILL, exp.dtype)
    out[inside] = exp[tuple(pts[inside].T)]
    return out


def _linear(exp, pts):
    """Reference n-linear interpolation (voxel i at coordinate i, fill outside), float64."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, exp.ndim)
    padded = np.pad(exp.astype(np.float64), 1, constant_values=FILL)  # every neighbour of a sample is in here
    out = np.empty(len(pts))
    for n, p in enumerate(pts):
        if not np.all(np.isfinite(p)):
            out[n] = np.nan
            continue
        if np.any((p <= -1) | (p >= np.asarray(exp.shape))):
            out[n] = FILL
            continue
        lo = np.floor(p).astype(np.int64)
        f = p - lo
        acc = 0.0
        for corner in np.ndindex(*(2,) * exp.ndim):
            c = np.asarray(corner)
            w = np.prod(np.where(c == 1, f, 1 - f))
            if w:
                acc += w * padded[tuple(lo + c + 1)]
        out[n] = acc
    return out


# Without FMA contraction (x86-64 builds), sample() sums exactly as _linear_ordered does.
MAX_ULP = 0 if platform.machine().lower() in ("x86_64", "amd64") else 4


def _linear_ordered(exp, pts):
    """n-linear interpolation formed in the reader's order: weights multiplied axis by axis,
    corners summed with bit a stepping along the a-th off-grid axis. Fill outside, NaN if not finite."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, exp.ndim)
    out = np.full(len(pts), np.nan)
    finite = np.all(np.isfinite(pts), axis=1)
    inside = finite & np.all((pts > -1) & (pts < np.asarray(exp.shape)), axis=1)
    out[finite & ~inside] = FILL
    lo = np.floor(pts[inside])
    frac = pts[inside] - lo
    lo = lo.astype(np.int64) + 1  # into the padded array
    padded = np.pad(exp.astype(np.float64), 1, constant_values=FILL)
    vals = np.empty(len(lo))
    off_grid = frac != 0
    for pattern in np.unique(off_grid, axis=0):
        sel = np.all(off_grid == pattern, axis=1)
        axes = np.flatnonzero(pattern)
        acc = np.zeros(sel.sum())
        for corner in range(2 ** len(axes)):
            w = np.ones(sel.sum())
            idx = lo[sel].copy()
            for a, i in enumerate(axes):
                if corner >> a & 1:
                    w = w * frac[sel, i]
                    idx[:, i] += 1
                else:
                    w = w * (1.0 - frac[sel, i])
            acc = acc + w * padded[tuple(idx.T)]
        vals[sel] = acc
    out[inside] = vals
    return out


@pytest.mark.parametrize("read_mode", ["mmap", "slab"])
@pytest.mark.parametrize("memory", [0, 64 << 20])
def test_read_points(array, read_mode, memory):
    root, exp = array
    rng = np.random.default_rng(2)
    pts = rng.integers(-4, max(SHAPE) + 4, size=(5000, 3))  # some outside, some in the missing chunks
    with ZarrReader.open(root, read_mode=read_mode, memory_cache=memory) as r:
        got = r.read_points(pts)
        assert got.dtype == exp.dtype and got.shape == (5000,)
        np.testing.assert_array_equal(got, _gather(exp, pts))
        grid = pts[:24].reshape(2, 3, 4, 3)  # leading dimensions are kept
        np.testing.assert_array_equal(r.read_points(grid), _gather(exp, grid).reshape(2, 3, 4))
        assert r.read_points(np.empty((0, 3), np.int64)).shape == (0,)


def test_read_points_rejects_bad_input(array):
    root, _ = array
    with ZarrReader.open(root) as r:
        with pytest.raises(ValueError, match="shape"):
            r.read_points([[1, 2]])
        with pytest.raises(TypeError, match="integers"):
            r.read_points([[1.5, 2, 3]])


def test_index_arrays_follow_numpy(array):
    root, exp = array
    z, y, x = np.array([0, 5, 36, -1]), np.array([[1], [49]]), np.array([2, 3, 44, -45])
    with ZarrReader.open(root) as r:
        np.testing.assert_array_equal(r[z, y, x], exp[z, y, x])  # broadcast to (2, 4), negatives wrap
        np.testing.assert_array_equal(r[3, [1, 2], [4, 5]], exp[3, [1, 2], [4, 5]])
        np.testing.assert_array_equal(r[[0, 1], 2, 3], exp[[0, 1], 2, 3])
        for bad in [(np.array([37]), 0, 0), (0, [1, 2], slice(None)), (np.array([True]), 0, 0)]:
            with pytest.raises(IndexError):
                r[bad]


def test_sample_linear(array):
    root, exp = array
    rng = np.random.default_rng(3)
    pts = rng.uniform(-2, max(SHAPE) + 1, size=(400, 3))
    pts[:50] = np.round(pts[:50])  # on the grid: exactly the stored values
    pts[50:60, 1] = 49.0  # the far edge itself, where the next neighbour is outside
    pts[60] = [np.nan, 1, 1]
    with ZarrReader.open(root) as r:
        got = r.sample(pts, dtype=np.float64)
        np.testing.assert_allclose(got, _linear(exp, pts), rtol=1e-12, atol=1e-9, equal_nan=True)
        assert np.isnan(got[60])
        on_grid = pts[:50].astype(np.int64)
        np.testing.assert_array_equal(got[:50], _gather(exp, on_grid).astype(np.float64))
        f32 = r.sample(pts)
        assert f32.dtype == np.float32
        np.testing.assert_allclose(f32, got.astype(np.float32), equal_nan=True)
        assert r.sample(pts.reshape(20, 20, 3)).shape == (20, 20)


def test_sample_nearest(array):
    root, exp = array
    pts = np.array([[0.4, 0.5, 1.5], [36.6, 2.49, 3.0], [-0.6, 0, 0], [-0.5, 0, 0], [np.inf, 0, 0], [5.5, 6.5, 7.5]])
    with ZarrReader.open(root) as r:
        got = r.sample(pts, order=0)
        assert got.dtype == exp.dtype
        np.testing.assert_array_equal(got, _gather(exp, np.where(np.isfinite(pts), np.rint(pts), -1).astype(np.int64)))
        with pytest.raises(ValueError, match="order"):
            r.sample(pts, order=3)


@pytest.mark.parametrize("dtype", [">u2", "<i4", "f4", ">f8", "bool"])
def test_sample_dtypes(tmp_path, dtype):
    pytest.importorskip("numcodecs")
    exp = (np.random.default_rng(4).integers(-50, 50, (20, 21, 22)) / (1 if dtype[-2] in "iu" else 3)).astype(dtype)
    if dtype == "bool":
        exp = exp != 0
    root = str(tmp_path / "a.zarr")
    _write_zarr(root, exp, (8, 8, 8))
    pts = np.random.default_rng(5).uniform(0, 19, (300, 3))
    with ZarrReader.open(root) as r:  # every neighbour is inside: the fill value plays no part
        got = r.sample(pts, dtype=np.float64)
        ref = _linear(exp.astype(np.float64), pts)
        np.testing.assert_allclose(got, ref, rtol=1e-6)
        ints = np.random.default_rng(6).integers(0, 20, (100, 3))
        np.testing.assert_array_equal(r.read_points(ints), exp[tuple(ints.T)])


def test_sample_2d(tmp_path):
    pytest.importorskip("numcodecs")
    exp = np.arange(30 * 40, dtype=np.uint16).reshape(30, 40)
    root = str(tmp_path / "a.zarr")
    _write_zarr(root, exp, (8, 16))
    pts = np.random.default_rng(7).uniform(-1, 41, (200, 2))
    with ZarrReader.open(root) as r:
        np.testing.assert_allclose(r.sample(pts, dtype=np.float64), _linear(exp, pts), rtol=1e-12)


@pytest.mark.parametrize("chunks", [(1, 8, 7), (5, 6, 9), (8, 16, 16)])
@pytest.mark.parametrize("read_mode", ["mmap", "slab"])
def test_point_chunk_geometry(tmp_path, chunks, read_mode):
    """Power-of-two, arbitrary and unit chunk axes use the same voxel boundaries."""
    pytest.importorskip("numcodecs")
    rng = np.random.default_rng(17)
    exp = rng.integers(0, 65535, (19, 22, 31), dtype=np.uint16)
    root = str(tmp_path / "geometry.zarr")
    _write_zarr(root, exp, chunks)
    pts = rng.uniform(-1.5, np.asarray(exp.shape) + 0.5, size=(3000, 3))
    boundaries = np.asarray(chunks) * rng.integers(0, 4, (1000, 3))
    pts = np.concatenate([pts, boundaries - 0.25, boundaries, boundaries + 0.5])
    with ZarrReader.open(root, read_mode=read_mode) as r:
        ints = np.floor(pts).astype(np.int64)
        np.testing.assert_array_equal(r.read_points(ints), _gather(exp, ints))
        np.testing.assert_array_equal(r.sample(pts, order=0), _gather(exp, np.rint(pts).astype(np.int64)))
        np.testing.assert_array_max_ulp(r.sample(pts, dtype=np.float64), _linear_ordered(exp, pts), maxulp=MAX_ULP)


def test_sample_4d(tmp_path):
    pytest.importorskip("numcodecs")
    rng = np.random.default_rng(18)
    exp = rng.integers(0, 255, (5, 6, 7, 8), dtype=np.uint8)
    root = str(tmp_path / "4d.zarr")
    _write_zarr(root, exp, (3, 4, 5, 4))
    pts = rng.uniform(-1.5, np.asarray(exp.shape) + 0.5, size=(1000, 4))
    pts[::3] = np.rint(pts[::3])
    with ZarrReader.open(root) as r:
        np.testing.assert_array_max_ulp(r.sample(pts, dtype=np.float64), _linear_ordered(exp, pts), maxulp=MAX_ULP)


def test_points_over_http(tmp_path, data):
    pytest.importorskip("numcodecs")
    _write_zarr(str(tmp_path / "a.zarr"), data, CHUNKS, missing=MISSING, compressor=BLOSC)
    exp = _expected(data, MISSING, CHUNKS)
    httpd, base = _serve(tmp_path)
    try:
        pts = np.random.default_rng(8).integers(-2, 52, (2000, 3))
        with ZarrReader.open(f"{base}/a.zarr", cache_dir=tmp_path / "cache") as r:
            np.testing.assert_array_equal(r.read_points(pts), _gather(exp, pts))
            fpts = pts + 0.25
            np.testing.assert_allclose(r.sample(fpts, dtype=np.float64), _linear(exp, fpts), rtol=1e-12)
            r2 = pickle.loads(pickle.dumps(r))
            np.testing.assert_array_equal(r2.read_points(pts), _gather(exp, pts))
            r2.close()
    finally:
        httpd.shutdown()


@pytest.mark.parametrize("read_mode", ["mmap", "slab"])
@pytest.mark.parametrize("memory", [0, 64 << 20])
def test_sample_linear_order_everywhere(array, read_mode, memory):
    """Points read from one chunk, from several (chunk edges), from missing chunks, at the array's
    edges, partly on the grid, and many in one chunk (split across threads) all sum the same way."""
    root, exp = array
    rng = np.random.default_rng(11)
    sh, ch = np.asarray(SHAPE), np.asarray(CHUNKS)
    edges = rng.integers(0, sh // ch + 2, size=(6000, 3)) * ch - 1 + rng.choice([0.0, 0.3, 0.999], size=(6000, 3))
    wide = rng.uniform(-1.5, sh + 0.5, size=(6000, 3))
    wide[:2000, 1] = np.round(wide[:2000, 1])  # on the grid along one axis
    wide[2000:2500] = np.round(wide[2000:2500])  # on the grid
    dense = 16 + rng.uniform(0.5, 14.5, size=(80000, 3))  # all in chunk (1, 1, 1), which is missing
    dense2 = rng.uniform(0.5, 14.5, size=(80000, 3))  # all in chunk (0, 0, 0)
    pts = np.concatenate([edges, wide, dense, dense2])
    with ZarrReader.open(root, read_mode=read_mode, memory_cache=memory) as r:
        got = r.sample(pts, dtype=np.float64)
        np.testing.assert_array_max_ulp(got, _linear_ordered(exp, pts), maxulp=MAX_ULP)
        np.testing.assert_array_equal(r.sample(pts), got.astype(np.float32))


@pytest.mark.parametrize("read_mode", ["mmap", "slab"])
@pytest.mark.parametrize("memory", [0, 64 << 20])
def test_sample_plane(array, read_mode, memory):
    """A sampling grid on a tilted plane (a surface patch) crossing many chunks, including a
    missing one and the array's edge: consecutive points in one chunk are planned together, and
    every point still sums the same way."""
    root, exp = array
    u = np.array([0.1, 0.2, 0.97]); u /= np.linalg.norm(u)
    w = np.array([0.95, 0.3, 0.0]); w -= u * (u @ w); w /= np.linalg.norm(w)
    v = np.cross(w, u)
    a, b, c = np.meshgrid(np.arange(3), np.arange(70), np.arange(70), indexing="ij")
    pts = np.array([10.0, 48.0, -1.5]) + 0.73 * (b[..., None] * v + c[..., None] * u) + 0.5 * a[..., None] * w
    with ZarrReader.open(root, read_mode=read_mode, memory_cache=memory) as r:
        got = r.sample(pts, dtype=np.float64)
        assert got.shape == pts.shape[:-1]
        np.testing.assert_array_max_ulp(got.ravel(), _linear_ordered(exp, pts), maxulp=MAX_ULP)
        nearest = np.where((pts > -1) & (pts < np.asarray(SHAPE) + 1), np.rint(pts), -1).astype(np.int64)
        np.testing.assert_array_equal(r.sample(pts, order=0), _gather(exp, nearest).reshape(pts.shape[:-1]))


def test_point_reads_span_batches(array):
    """More points than one native batch (2^20): every output still lands in its place."""
    root, exp = array
    rng = np.random.default_rng(12)
    pts = rng.uniform(-1.5, np.asarray(SHAPE) + 0.5, size=((1 << 20) + 5000, 3))
    with ZarrReader.open(root) as r:
        ints = np.floor(pts).astype(np.int64)
        np.testing.assert_array_equal(r.read_points(ints), _gather(exp, ints))
        nearest = np.where((pts > -1) & (pts < np.asarray(SHAPE) + 1), np.rint(pts), -1).astype(np.int64)
        np.testing.assert_array_equal(r.sample(pts, order=0), _gather(exp, nearest))
        np.testing.assert_array_max_ulp(r.sample(pts, dtype=np.float64), _linear_ordered(exp, pts), maxulp=MAX_ULP)
