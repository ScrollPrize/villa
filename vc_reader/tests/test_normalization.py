"""Opt-in floating crop outputs: native conversion and per-crop statistics."""
import json
import pickle

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("vc_reader._native")
from test_zarr_reader import _write_zarr, _serve, BLOSC
from vc_reader.reader import ZarrReader


def reference(raw, dtype, mode):
    a = raw.astype(dtype).astype(np.float64)
    if not a.size or mode is None:
        return a.astype(dtype)
    selected = a[a != 0] if mode == "zscore" else a.ravel()
    out = np.zeros_like(a)
    if not selected.size or selected.min() == selected.max():
        return out.astype(dtype)
    if mode == "minmax":
        out = (a - selected.min()) / (selected.max() - selected.min())
    else:
        mask = a != 0
        out[mask] = (a[mask] - selected.mean()) / selected.std()
    return out.astype(dtype)


def write(root, data, chunks, **kwargs):
    pytest.importorskip("numcodecs")
    _write_zarr(str(root), data, chunks, **kwargs)
    # Complete chunks below; use zero fill for missing chunks and exterior padding.
    path = root / ".zarray"
    meta = json.loads(path.read_text())
    meta["fill_value"] = 0
    path.write_text(json.dumps(meta))
    return root


@pytest.fixture(params=[None, BLOSC], ids=["raw", "blosc"])
def source(tmp_path, request):
    data = (np.arange(16*18*20).reshape(16, 18, 20) % 251).astype(np.uint16)
    data[:3] = 0
    return write(tmp_path / "a.zarr", data, (4, 6, 5), compressor=request.param, missing=[(1, 1, 1)])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", [None, "minmax", "zscore"])
@pytest.mark.parametrize("read_mode", ["mmap", "slab"])
def test_block_outputs(source, dtype, mode, read_mode):
    regions = [((-3, 2, 3), (12, 13, 14)), ((8, 0, 0), (6, 9, 10)),
               ((40, 0, 0), (3, 4, 5)), ((0, 0, 0), (0, 2, 3))]
    with ZarrReader.open(source, read_mode=read_mode, memory_cache=0) as r:
        raw = r.read_many(regions)
        outputs = r.read_many(regions, dtype=dtype, normalization=mode)
        for (start, size), before, got in zip(regions, raw, outputs):
            assert got.dtype == dtype
            np.testing.assert_allclose(got, reference(before, dtype, mode), rtol=2e-6, atol=1e-7)
            out = np.empty(size, dtype)
            assert r.read(start, size, out=out, dtype=dtype, normalization=mode) is out
            np.testing.assert_array_equal(out, got)
            if mode == "zscore":
                assert np.all(got[before == 0] == 0)
        for old, new in zip(raw, r.read_many(regions)):
            np.testing.assert_array_equal(old, new)
        assert r.dtype == np.dtype("uint16")


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", [None, "minmax", "zscore"])
def test_affine_outputs(source, order, dtype, mode):
    grids = [((12, 13, 14), (-2.5, 1.25, 2.75), [[1, 0.1, 0], [0, .9, .2], [.1, 0, .8]]),
             ((2, 4, 5), (3, 4, 5), np.zeros((3, 3))),
             ((2, 3, 4), (-100, 0, 0), np.eye(3)), ((0, 2, 3), (0, 0, 0), np.eye(3))]
    with ZarrReader.open(source, memory_cache=8 << 20) as r:
        raw = [r.read_affine(*g, order=order, dtype=dtype if order else None) for g in grids]
        outs = [np.empty(g[0], dtype) for g in grids]
        got = r.read_affine_many(grids, order=order, dtype=dtype, normalization=mode, out=outs)
        for a, b, out in zip(got, raw, outs):
            assert a is out
            np.testing.assert_allclose(a, reference(b, dtype, mode), rtol=2e-6, atol=1e-7)
            if mode == "zscore":
                assert np.all(a[b == 0] == 0)


@pytest.mark.parametrize("source_dtype", ["u1", ">u2", ">i4", "i8", ">u8", "bool", "f4", ">f8"])
def test_cast_source_types(tmp_path, source_dtype):
    raw = np.arange(120).reshape(4, 5, 6).astype(source_dtype)
    if raw.dtype.kind == "i": raw -= 30
    root = write(tmp_path / "a.zarr", raw, raw.shape)
    with ZarrReader.open(root) as r:
        for dtype in (np.float32, np.float64):
            np.testing.assert_array_equal(r.read((0, 0, 0), raw.shape, dtype=dtype), raw.astype(dtype))
            np.testing.assert_array_equal(r.read_affine(raw.shape, (0, 0, 0), np.eye(3), order=0, dtype=dtype), raw.astype(dtype))
        np.testing.assert_array_equal(r.read((0, 0, 0), raw.shape), raw)


@pytest.mark.parametrize("ndim", [1, 2, 4])
def test_converted_box_ranks(tmp_path, ndim):
    shape = (6,) * ndim
    data = np.arange(6**ndim).reshape(shape).astype(">u2")
    root = write(tmp_path / "a.zarr", data, (3,) * ndim)
    with ZarrReader.open(root) as r:
        start, size = (-1,) * ndim, (8,) * ndim
        raw = r.read(start, size)
        for mode in (None, "minmax", "zscore"):
            np.testing.assert_allclose(r.read(start, size, dtype=np.float64, normalization=mode),
                                       reference(raw, np.float64, mode), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("raw", [np.zeros((2, 3, 4)), np.full((2, 3, 4), 7.),
                                np.array([0., 7., 7., 0.]).reshape(1, 2, 2)])
def test_zero_and_constant(tmp_path, raw):
    root = write(tmp_path / "a.zarr", raw, raw.shape)
    with ZarrReader.open(root) as r:
        for mode in ("minmax", "zscore"):
            got = r.read((0, 0, 0), raw.shape, normalization=mode)
            assert got.dtype == np.float32
            np.testing.assert_array_equal(got, reference(raw, np.float32, mode))


def test_mask_zeros_not_mean(source):
    with ZarrReader.open(source) as r:
        raw = r.read((-2, 0, 0), (12, 14, 15))
        got = r.read((-2, 0, 0), raw.shape, normalization="zscore", dtype=np.float64)
        mask = raw != 0
        assert np.all(got[~mask] == 0)
        assert abs(got[mask].mean()) < 1e-12
        assert abs(got[mask].std() - 1) < 1e-12


@pytest.mark.parametrize("mode", ["minmax", "zscore"])
def test_nonfinite_rejected(tmp_path, mode):
    for value in (np.nan, np.inf, -np.inf):
        raw = np.array([0, 1, value, 3.], np.float64).reshape(1, 2, 2)
        root = write(tmp_path / str(value), raw, raw.shape)
        with ZarrReader.open(root) as r:
            with pytest.raises(ValueError, match="finite"):
                r.read((0, 0, 0), raw.shape, normalization=mode)
            with pytest.raises(ValueError, match="finite"):
                r.read_affine(raw.shape, (0, 0, 0), np.eye(3), normalization=mode)
            np.testing.assert_array_equal(r.read((0, 0, 0), raw.shape, dtype=np.float64), raw)


def test_output_validation_and_defaults(source):
    shape = (4, 5, 6)
    with ZarrReader.open(source) as r:
        for read in [lambda **kw: r.read((0, 0, 0), shape, **kw),
                     lambda **kw: r.read_affine(shape, (0, 0, 0), np.eye(3), order=0, **kw)]:
            assert read().dtype == r.dtype
            for kwargs in [dict(normalization="unknown"), dict(normalization="zscore", dtype="uint16"),
                           dict(dtype="float16"), dict(dtype="int8"), dict(normalization="minmax", out=np.empty(shape, "u2"))]:
                with pytest.raises(ValueError): read(**kwargs)
            with pytest.raises(ValueError): read(out=np.empty(shape, np.float32))  # out never opts into casting
            assert read(normalization="zscore").dtype == np.float32
        out = np.empty((4, 5, 12), np.float64)[..., ::2]
        assert r.read((0, 0, 0), shape, out, dtype=np.float64, normalization="minmax") is out
        np.testing.assert_array_equal(out, r.read((0, 0, 0), shape, dtype=np.float64, normalization="minmax"))
        assert r.read_many([], dtype=np.float32, normalization="zscore") == []
        assert r.read_affine_many([], normalization="zscore") == []


def test_population_statistics_across_batches_and_threads(tmp_path):
    shape = (65, 129, 129)  # more than the native point batch, and many reduction segments
    raw = (np.arange(np.prod(shape)) % 200).reshape(shape).astype(np.uint8)
    root = write(tmp_path / "a.zarr", raw, shape)
    expected = reference(raw, np.float64, "zscore")
    previous = None
    for threads in (1, 4):
        with ZarrReader.open(root, threads=threads) as r:
            got = r.read_affine(shape, (0, 0, 0), np.eye(3), dtype=np.float64, normalization="zscore")
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
            if previous is not None: np.testing.assert_array_equal(got, previous)
            previous = got


@pytest.mark.parametrize("values", [[-1e308, 1e308, 0, 5e307], [1e-310, 2e-310, 0, 3e-310],
                                   [1e12, 1e12 + 0.25, 0, 1e12 + 0.5]])
def test_extreme_finite_statistics(tmp_path, values):
    raw = np.array(values).reshape(1, 2, 2)
    root = write(tmp_path / "a.zarr", raw, raw.shape)
    with ZarrReader.open(root) as r:
        for mode in ("minmax", "zscore"):
            got = r.read((0, 0, 0), raw.shape, dtype=np.float64, normalization=mode)
            assert np.isfinite(got).all()
            if mode == "zscore":
                assert abs(got[raw != 0].mean()) < 1e-12
                assert abs(got[raw != 0].std() - 1) < 1e-12
            else:
                assert got.min() == 0 and got.max() == 1


def test_normalization_http_pickle_offline(tmp_path):
    raw = np.arange(120, dtype=np.uint16).reshape(4, 5, 6)
    write(tmp_path / "a.zarr", raw, raw.shape, compressor=BLOSC)
    server, base = _serve(tmp_path)
    try:
        with ZarrReader.open(f"{base}/a.zarr", cache_dir=tmp_path / "cache") as r:
            np.testing.assert_allclose(r.read((0, 0, 0), raw.shape, normalization="zscore"), reference(raw, np.float32, "zscore"))
            r.flush()
            with pickle.loads(pickle.dumps(r)) as other:
                np.testing.assert_array_equal(other.read((0, 0, 0), raw.shape), raw)
        with ZarrReader.open(f"{base}/a.zarr", cache_dir=tmp_path / "cache", offline=True) as r:
            np.testing.assert_allclose(r.read((0, 0, 0), raw.shape, normalization="minmax"), reference(raw, np.float32, "minmax"))
    finally:
        server.shutdown()
