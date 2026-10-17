"""Reader dtype conversion remains available after normalization moves to augment."""
import json
import pickle

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("vc_reader._native")
from test_zarr_reader import _write_zarr, _serve, BLOSC
from vc_reader.reader import ZarrReader


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
@pytest.mark.parametrize("read_mode", ["mmap", "slab"])
def test_block_outputs(source, dtype, read_mode):
    regions = [((-3, 2, 3), (12, 13, 14)), ((8, 0, 0), (6, 9, 10)),
               ((40, 0, 0), (3, 4, 5)), ((0, 0, 0), (0, 2, 3))]
    with ZarrReader.open(source, read_mode=read_mode, memory_cache=0) as r:
        raw = r.read_many(regions)
        outputs = r.read_many(regions, dtype=dtype)
        for (start, size), before, got in zip(regions, raw, outputs):
            assert got.dtype == dtype
            np.testing.assert_allclose(got, before.astype(dtype), rtol=2e-6, atol=1e-7)
            out = np.empty(size, dtype)
            assert r.read(start, size, out=out, dtype=dtype) is out
            np.testing.assert_array_equal(out, got)
        for old, new in zip(raw, r.read_many(regions)):
            np.testing.assert_array_equal(old, new)
        assert r.dtype == np.dtype("uint16")


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_affine_outputs(source, order, dtype):
    grids = [((12, 13, 14), (-2.5, 1.25, 2.75), [[1, 0.1, 0], [0, .9, .2], [.1, 0, .8]]),
             ((2, 4, 5), (3, 4, 5), np.zeros((3, 3))),
             ((2, 3, 4), (-100, 0, 0), np.eye(3)), ((0, 2, 3), (0, 0, 0), np.eye(3))]
    with ZarrReader.open(source, memory_cache=8 << 20) as r:
        raw = [r.read_affine(*g, order=order, dtype=dtype if order else None) for g in grids]
        outs = [np.empty(g[0], dtype) for g in grids]
        got = r.read_affine_many(grids, order=order, dtype=dtype, out=outs)
        for a, b, out in zip(got, raw, outs):
            assert a is out
            np.testing.assert_allclose(a, b.astype(dtype), rtol=2e-6, atol=1e-7)


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
        np.testing.assert_array_equal(r.read(start, size, dtype=np.float64), raw.astype(np.float64))


def test_output_validation_and_defaults(source):
    shape = (4, 5, 6)
    with ZarrReader.open(source) as r:
        for read in [lambda **kw: r.read((0, 0, 0), shape, **kw),
                     lambda **kw: r.read_affine(shape, (0, 0, 0), np.eye(3), order=0, **kw)]:
            assert read().dtype == r.dtype
            for kwargs in [dict(dtype="float16"), dict(dtype="int8"),
                           dict(dtype=np.float64, out=np.empty(shape,"u2"))]:
                with pytest.raises(ValueError): read(**kwargs)
            with pytest.raises(ValueError): read(out=np.empty(shape, np.float32))  # out never opts into casting
            assert read(dtype=np.float32).dtype == np.float32
            with pytest.raises(TypeError,match="normalization"):
                read(normalization="zscore")
        out = np.empty((4, 5, 12), np.float64)[..., ::2]
        assert r.read((0, 0, 0), shape, out, dtype=np.float64) is out
        np.testing.assert_array_equal(out, r.read((0, 0, 0), shape, dtype=np.float64))
        assert r.read_many([], dtype=np.float32) == []
        assert r.read_affine_many([]) == []


def test_float_output_http_pickle_offline(tmp_path):
    raw = np.arange(120, dtype=np.uint16).reshape(4, 5, 6)
    write(tmp_path / "a.zarr", raw, raw.shape, compressor=BLOSC)
    server, base = _serve(tmp_path)
    try:
        with ZarrReader.open(f"{base}/a.zarr", cache_dir=tmp_path / "cache") as r:
            np.testing.assert_allclose(r.read((0, 0, 0), raw.shape), raw.astype(np.float32))
            r.flush()
            with pickle.loads(pickle.dumps(r)) as other:
                np.testing.assert_array_equal(other.read((0, 0, 0), raw.shape), raw)
        with ZarrReader.open(f"{base}/a.zarr", cache_dir=tmp_path / "cache", offline=True) as r:
            np.testing.assert_allclose(r.read((0, 0, 0), raw.shape), raw.astype(np.float32))
    finally:
        server.shutdown()


def test_normalization_removed_from_batch_readers(source):
    with ZarrReader.open(source) as reader:
        for read in (reader.read_many, reader.read_affine_many):
            with pytest.raises(TypeError,match="normalization"):
                read([],normalization="minmax")


@pytest.mark.parametrize("value", [np.nan,np.inf,-np.inf])
def test_reader_conversion_preserves_nonfinite_values(tmp_path,value):
    raw = np.array([0.,1.,value,3.]).reshape(1,2,2)
    root = write(tmp_path / "a.zarr",raw,raw.shape)
    with ZarrReader.open(root) as reader:
        np.testing.assert_array_equal(reader.read((0,0,0),raw.shape,dtype=np.float64),raw)
