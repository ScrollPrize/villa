from __future__ import annotations

import functools
import http.server
import json
import os
import pickle
import threading

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("vc_reader._native", reason="the native extension is not built")

from vc_reader.reader import MemoryCache, ZarrReader  # noqa: E402
from vc_reader.reader.disk_cache import (  # noqa: E402
    VC3D_DELTA3D_MARKER, decoded_cache_path, delta3d_zarray, vc3d_volume_id)
from vc_reader.reader.zarr_meta import ArrayMeta  # noqa: E402

SHAPE = (37, 50, 45)
CHUNKS = (16, 16, 16)
BLOSC = {"id": "blosc", "cname": "zstd", "clevel": 1, "shuffle": 1, "blocksize": 0}


def _write_zarr(root, data, chunks, *, compressor=None, filters=None, sep="/", missing=(), order="C"):
    """A zarr v2 array written by hand (no zarr-python), with chosen chunks left out."""
    import numcodecs

    os.makedirs(root, exist_ok=True)
    meta = dict(zarr_format=2, shape=list(data.shape), chunks=list(chunks), dtype=data.dtype.str,
                fill_value=7, order=order, compressor=compressor, filters=filters, dimension_separator=sep)
    with open(os.path.join(root, ".zarray"), "w") as f:
        json.dump(meta, f)
    codec = numcodecs.get_codec(compressor) if compressor else None
    fl = [numcodecs.get_codec(x) for x in filters or ()]
    grid = [-(-s // c) for s, c in zip(data.shape, chunks)]
    for idx in np.ndindex(*grid):
        if idx in missing:
            continue
        block = np.full(chunks, 7, dtype=data.dtype)
        sl = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, data.shape))
        block[tuple(slice(0, x.stop - x.start) for x in sl)] = data[sl]
        buf = block.tobytes(order=order)
        for f in fl:
            buf = f.encode(buf)
        if codec:
            buf = codec.encode(buf)
        path = os.path.join(root, sep.join(map(str, idx)))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(bytes(buf))


def _expected(data, missing, chunks):
    exp = data.copy()
    for idx in missing:
        exp[tuple(slice(i * c, (i + 1) * c) for i, c in zip(idx, chunks))] = 7
    return exp


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, SHAPE, dtype=np.uint8)


MISSING = {(1, 1, 1), (0, 3, 2)}
VARIANTS = {"raw": dict(), "raw-dot-sep": dict(sep="."), "blosc": dict(compressor=BLOSC)}


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def local_array(request, tmp_path_factory, data):
    pytest.importorskip("numcodecs")
    root = str(tmp_path_factory.mktemp(request.param) / "a.zarr")
    _write_zarr(root, data, CHUNKS, missing=MISSING, **VARIANTS[request.param])
    return root, _expected(data, MISSING, CHUNKS)


REGIONS = [
    ((0, 0, 0), SHAPE),  # whole array
    ((3, 5, 7), (20, 30, 25)),  # unaligned, many chunks
    ((16, 16, 16), (16, 16, 16)),  # exactly the missing chunk
    ((30, 40, 40), (10, 12, 9)),  # crosses the far edges
    ((-5, -3, 2), (12, 10, 10)),  # crosses the near edges
    ((40, 0, 0), (3, 3, 3)),  # fully outside
    ((5, 5, 5), (1, 1, 1)),
    ((2, 2, 2), (10, 2, 30)),  # few rows across many planes
]


def _ref(exp, start, size):
    out = np.full(size, 7, dtype=exp.dtype)
    lo = [max(s, 0) for s in start]
    hi = [min(s + n, d) for s, n, d in zip(start, size, exp.shape)]
    if all(h > l for l, h in zip(lo, hi)):
        out[tuple(slice(l - s, h - s) for l, h, s in zip(lo, hi, start))] = exp[tuple(slice(l, h) for l, h in zip(lo, hi))]
    return out


@pytest.mark.parametrize("read_mode", ["mmap", "slab"])
@pytest.mark.parametrize("memory", [0, 64 << 20])
def test_local_reads(local_array, memory, read_mode):
    root, exp = local_array
    with ZarrReader.open(root, memory_cache=memory, read_mode=read_mode) as r:
        assert r.shape == SHAPE and r.chunks == CHUNKS and r.fill_value == 7
        for start, size in REGIONS:
            for _ in range(2):  # second pass hits the memory cache when enabled
                np.testing.assert_array_equal(r.read(start, size), _ref(exp, start, size))


@pytest.mark.parametrize("read_mode", ["mmap", "slab"])
def test_read_many(local_array, read_mode):
    root, exp = local_array
    r = ZarrReader.open(root, read_mode=read_mode, threads=4)
    outs = r.read_many(REGIONS + REGIONS)  # repeated regions share chunk reads
    for (start, size), got in zip(REGIONS + REGIONS, outs):
        np.testing.assert_array_equal(got, _ref(exp, start, size))


def test_read_many_into_given_outputs(local_array):
    root, exp = local_array
    r = ZarrReader.open(root, threads=4)
    regions = REGIONS[:3]
    outs = [np.empty(size, np.uint8) for _, size in regions]
    outs[1] = None  # allocated by the reader
    got = r.read_many(regions, out=outs)
    assert got[0] is outs[0] and got[2] is outs[2]
    for (start, size), a in zip(regions, got):
        np.testing.assert_array_equal(a, _ref(exp, start, size))
    # A non-contiguous destination is filled through a temporary.
    big = np.zeros((40, 40, 40), np.uint8)
    view = big[::2, :16, :16][:8]
    r.read((0, 0, 0), (8, 16, 16), out=view)
    np.testing.assert_array_equal(view, exp[:8, :16, :16])
    with pytest.raises(ValueError):
        r.read_many(regions, out=outs[:2])
    with pytest.raises(ValueError):
        r.read((0, 0, 0), (2, 2, 2), out=np.empty((2, 2, 3), np.uint8))


def test_getitem(local_array):
    root, exp = local_array
    r = ZarrReader.open(root)
    np.testing.assert_array_equal(r[...], exp)
    np.testing.assert_array_equal(r[3:20, -10:, 5], exp[3:20, -10:, 5])
    np.testing.assert_array_equal(r[-1], exp[-1])
    np.testing.assert_array_equal(r[10:5], exp[10:5])
    with pytest.raises(IndexError):
        r[::2]
    with pytest.raises(IndexError):
        r[SHAPE[0]]


def test_matches_zarr_python(local_array):
    zarr = pytest.importorskip("zarr")
    root, _ = local_array
    z = zarr.open_array(root, mode="r")
    r = ZarrReader.open(root)
    for start, size in REGIONS[:5]:
        sl = tuple(slice(max(s, 0), min(s + n, d)) for s, n, d in zip(start, size, SHAPE))
        ours = r.read([x.start for x in sl], [x.stop - x.start for x in sl])
        np.testing.assert_array_equal(ours, z[sl])


@pytest.mark.parametrize("variant", [
    dict(compressor={"id": "zlib", "level": 1}),
    dict(compressor=BLOSC, filters=[{"id": "delta", "dtype": "|u1"}]),
    dict(order="F"),
])
def test_unsupported_arrays_raise(tmp_path, data, variant):
    root = str(tmp_path / "a.zarr")
    _write_zarr(root, data, CHUNKS, **variant)
    with pytest.raises(ValueError, match="vc_reader cannot read"):
        ZarrReader.open(root)


def test_memory_cache(local_array):
    root, exp = local_array
    cache = MemoryCache(1 << 20)
    a, b = ZarrReader.open(root, memory_cache=cache), ZarrReader.open(root, memory_cache=cache)
    a.read((0, 0, 0), (16, 16, 32))
    misses = cache.misses
    np.testing.assert_array_equal(b.read((0, 0, 0), (16, 16, 32)), exp[:16, :16, :32])
    assert cache.hits >= 2 and cache.misses == misses  # served by the shared cache
    small = MemoryCache(3 * 16**3)
    r = ZarrReader.open(root, memory_cache=small)
    r[...]
    assert len(small) <= 3 and small.nbytes <= small.max_bytes
    with pytest.raises(TypeError):
        ZarrReader.open(root, memory_cache="lots")


def test_output_pool_reuses_buffers_and_overwrites_them(local_array):
    from vc_reader.reader import output_pool_stats, set_output_pool_bytes

    root, exp = local_array
    r = ZarrReader.open(root)
    region = ((30, 40, 40), (16, 16, 16))  # crosses the far edges and touches a missing chunk
    a = r.read(*region)
    a[...] = 255  # dirty the buffer, then hand it back to the pool
    del a
    hits = output_pool_stats()["hits"]
    b = r.read(*region)
    assert output_pool_stats()["hits"] == hits + 1  # the same buffer, reused
    np.testing.assert_array_equal(b, _ref(exp, *region))  # every byte rewritten: data and fill
    c = r.read(*region)  # b is alive: a different buffer
    assert c.ctypes.data != b.ctypes.data
    try:
        set_output_pool_bytes(0)
        del b, c
        assert output_pool_stats()["retained_bytes"] == 0
        np.testing.assert_array_equal(r.read(*region), _ref(exp, *region))
    finally:
        set_output_pool_bytes(1 << 30)


def test_pickle_roundtrip(local_array):
    root, exp = local_array
    r = ZarrReader.open(root, memory_cache=1 << 20)
    r.read((0, 0, 0), (8, 8, 8))
    r2 = pickle.loads(pickle.dumps(r))
    assert len(r2.memory_cache) == 0 and r2.memory_cache.max_bytes == 1 << 20
    np.testing.assert_array_equal(r2[...], exp)


def test_closed_reader(local_array):
    root, _ = local_array
    r = ZarrReader.open(root)
    r.close()
    with pytest.raises(ValueError, match="closed"):
        r.read((0, 0, 0), (1, 1, 1))


# -- HTTP and the disk cache ------------------------------------------------------


class _Handler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, like S3
    log_message = lambda *a: None  # noqa: E731

    def do_GET(self):  # noqa: N802
        self.server.requests.append(self.path)
        super().do_GET()


def _serve(root):
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Handler, directory=str(root)))
    httpd.requests = []
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture(scope="module")
def server(tmp_path_factory, data):
    root = tmp_path_factory.mktemp("www")
    _write_zarr(str(root / "raw.zarr"), data, CHUNKS, missing=MISSING)
    _write_zarr(str(root / "blosc.zarr"), data, CHUNKS, missing=MISSING, compressor=BLOSC)
    httpd, base = _serve(root)
    yield httpd, base
    httpd.shutdown()


@pytest.mark.parametrize("name", ["raw.zarr", "blosc.zarr"])
def test_http_reads(server, data, name):
    httpd, base = server
    exp = _expected(data, MISSING, CHUNKS)
    with ZarrReader.open(f"{base}/{name}") as r:
        for start, size in REGIONS:
            np.testing.assert_array_equal(r.read(start, size), _ref(exp, start, size))
        assert r.stats()["remote_missing"] >= 1


def test_disk_cache(server, data, tmp_path):
    httpd, base = server
    exp = _expected(data, MISSING, CHUNKS)
    url = f"{base}/blosc.zarr"
    r = ZarrReader.open(url, cache_dir=tmp_path)
    np.testing.assert_array_equal(r[...], exp)
    r.flush()  # cache files are written after the read returns
    root = decoded_cache_path(tmp_path, url)
    assert os.path.exists(os.path.join(root, "1/1/1.missing"))
    assert os.path.getsize(os.path.join(root, "0/0/0")) == 16**3  # stored decoded

    # A second reader is served from disk: only the metadata request goes out.
    httpd.requests.clear()
    r2 = ZarrReader.open(url, cache_dir=tmp_path)
    np.testing.assert_array_equal(r2[...], exp)
    assert httpd.requests == ["/blosc.zarr/.zarray"]
    assert r2.stats()["remote_fetches"] == 0

    # The cache is a plain uncompressed zarr array.
    np.testing.assert_array_equal(ZarrReader.open(root)[...], exp)
    zarr = pytest.importorskip("zarr")
    np.testing.assert_array_equal(zarr.open_array(root, mode="r")[...], exp)


def test_read_only_cache(server, data, tmp_path):
    _, base = server
    url = f"{base}/raw.zarr"
    r = ZarrReader.open(url, cache_dir=tmp_path, cache_writes=False)
    np.testing.assert_array_equal(r[...], _expected(data, MISSING, CHUNKS))
    assert r.write_layer is None and not any(tmp_path.iterdir())


def test_disk_cache_rejects_changed_source(server, tmp_path):
    _, base = server
    url = f"{base}/raw.zarr"
    ZarrReader.open(url, cache_dir=tmp_path)
    zattrs = os.path.join(decoded_cache_path(tmp_path, url), ".zattrs")
    meta = json.load(open(zattrs))
    meta["vc_reader"]["source_zarray"]["chunks"] = [8, 8, 8]
    json.dump(meta, open(zattrs, "w"))
    with pytest.raises(ValueError, match="different array"):
        ZarrReader.open(url, cache_dir=tmp_path)


def test_cache_writes_finish_after_the_read(server, data, tmp_path):
    httpd, base = server
    url = f"{base}/raw.zarr"
    exp = _expected(data, MISSING, CHUNKS)
    root = decoded_cache_path(tmp_path, url)
    r = ZarrReader.open(url, cache_dir=tmp_path)
    httpd.requests.clear()
    np.testing.assert_array_equal(r[...], exp)
    # Rereading while writes may still be queued never fetches a chunk twice.
    np.testing.assert_array_equal(r[...], exp)
    chunk_requests = [p for p in httpd.requests if not p.endswith(".zarray")]
    assert len(chunk_requests) == len(set(chunk_requests)) == 3 * 4 * 3
    r.close()  # finishes the queued writes
    assert os.path.getsize(os.path.join(root, "2/3/2")) == 16**3
    assert os.path.exists(os.path.join(root, "0/3/2.missing"))
    # A reader dropped without close() still writes what it fetched.
    other = tmp_path / "other"
    r = ZarrReader.open(url, cache_dir=other)
    r.read((0, 0, 0), (16, 16, 16))
    del r
    import gc

    gc.collect()
    assert os.path.getsize(os.path.join(decoded_cache_path(other, url), "0/0/0")) == 16**3


@pytest.mark.parametrize("name", ["raw.zarr", "blosc.zarr"])
def test_compressed_cache(server, data, tmp_path, name):
    vc_delta3d = pytest.importorskip("vc_delta3d")  # only to check the stored payloads
    httpd, base = server
    url = f"{base}/{name}"
    exp = _expected(data, MISSING, CHUNKS)
    with ZarrReader.open(url, cache_dir=tmp_path, compress_cache=True) as r:
        np.testing.assert_array_equal(r[...], exp)
    root = decoded_cache_path(tmp_path, url)
    assert json.load(open(os.path.join(root, ".zarray")))["compressor"] == {"id": "vc-delta3d", "quant": 1}
    stored = open(os.path.join(root, "0/0/0"), "rb").read()
    assert stored[:4] == b"D3D1" and stored[6] <= 1  # (random test data does not shrink)
    np.testing.assert_array_equal(np.frombuffer(vc_delta3d.Delta3D().decode(stored), np.uint8).reshape(CHUNKS),
                                  exp[:16, :16, :16])
    assert os.path.exists(os.path.join(root, "1/1/1.missing"))
    # Served from the compressed cache alone, by any reader (the flag only matters when creating it).
    httpd.requests.clear()
    r2 = ZarrReader.open(url, cache_dir=tmp_path)
    np.testing.assert_array_equal(r2[...], exp)
    assert httpd.requests == [f"/{name}/.zarray"] and r2.write_layer.stored.codec_id == "vc-delta3d"
    # It is an ordinary zarr v2 array (zarr-python reads it once vc_delta3d is imported).
    zarr = pytest.importorskip("zarr")
    np.testing.assert_array_equal(zarr.open_array(root, mode="r")[...], exp)


def test_compress_flag_keeps_existing_cache_format(server, data, tmp_path):
    _, base = server
    url = f"{base}/raw.zarr"
    with ZarrReader.open(url, cache_dir=tmp_path) as r:
        r.read((0, 0, 0), (16, 16, 16))
    with ZarrReader.open(url, cache_dir=tmp_path, compress_cache=True) as r:
        assert r.write_layer.stored.codec_id == ""  # appends stay uncompressed, like the cache
        r.read((16, 0, 0), (16, 16, 16))
    assert os.path.getsize(os.path.join(decoded_cache_path(tmp_path, url), "1/0/0")) == 16**3


def test_compress_cache_needs_3d_uint(tmp_path):
    root = tmp_path / "www"
    data = np.zeros((8, 8), np.float32)
    os.makedirs(root / "f.zarr")
    json.dump(dict(zarr_format=2, shape=[8, 8], chunks=[4, 4], dtype="<f4", fill_value=0, order="C",
                   compressor=None, filters=None), open(root / "f.zarr" / ".zarray", "w"))
    httpd, base = _serve(root)
    try:
        with pytest.raises(ValueError, match="compress_cache needs 3-D"):
            ZarrReader.open(f"{base}/f.zarr", cache_dir=tmp_path / "cache", compress_cache=True)
        ZarrReader.open(f"{base}/f.zarr", cache_dir=tmp_path / "cache").read((0, 0), data.shape)
    finally:
        httpd.shutdown()


def test_lossy_own_cache_is_skipped(server, data, tmp_path):
    vc_delta3d = pytest.importorskip("vc_delta3d")
    httpd, base = server
    url = f"{base}/raw.zarr"
    exp = _expected(data, MISSING, CHUNKS)
    with ZarrReader.open(url, cache_dir=tmp_path, compress_cache=True) as r:
        r.read((0, 0, 0), (16, 16, 16))
    root = decoded_cache_path(tmp_path, url)
    meta = json.load(open(os.path.join(root, ".zarray")))
    meta["compressor"]["quant"] = 5  # a cache declared lossy
    json.dump(meta, open(os.path.join(root, ".zarray"), "w"))
    with open(os.path.join(root, "0/0/0"), "wb") as f:
        f.write(vc_delta3d.Delta3D(quant=5).encode(exp[:16, :16, :16].copy()))
    r = ZarrReader.open(url, cache_dir=[tmp_path], cache_writes=False)
    assert r.cache_layers == []  # not used for reads
    np.testing.assert_array_equal(r.read((0, 0, 0), (16, 16, 16)), exp[:16, :16, :16])
    r = ZarrReader.open(url, cache_dir=[tmp_path], cache_writes=False, allow_lossy_cache=True)
    httpd.requests.clear()
    got = r.read((0, 0, 0), (16, 16, 16))
    assert httpd.requests == [] and 0 < np.abs(got.astype(int) - exp[:16, :16, :16]).max() <= 2


def test_missing_remote_array_raises(server):
    _, base = server
    with pytest.raises(FileNotFoundError):
        ZarrReader.open(f"{base}/does-not-exist.zarr")


def test_cache_path_uses_s3_form_for_known_buckets():
    https = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/v.zarr/2"
    s3 = "s3://vesuvius-challenge-open-data/PHercParis4/volumes/v.zarr/2"
    expected = os.path.join("/c", "s3", "vesuvius-challenge-open-data", "PHercParis4/volumes/v.zarr/2")
    assert decoded_cache_path("/c", https) == decoded_cache_path("/c", s3) == expected
    assert decoded_cache_path("/c", "http://127.0.0.1:9/a.zarr") == os.path.join("/c", "http", "127.0.0.1:9", "a.zarr")


def _foreign_cache(tmp_path, url, data, chunks=CHUNKS):
    """A cache as vesuvius' RemoteChunkedArray writes it: decoded .zarray, its own .zattrs, chunk (0,0,0)."""
    root = decoded_cache_path(tmp_path, url)
    os.makedirs(os.path.join(root, "0", "0"))
    meta = dict(zarr_format=2, shape=list(data.shape), chunks=list(chunks), dtype="|u1", fill_value=7,
                compressor=None, filters=None, order="C", dimension_separator="/")
    with open(os.path.join(root, ".zarray"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(root, ".zattrs"), "w") as f:
        json.dump({"remote_url": url, "level": 0}, f)
    marked = np.full(chunks, 99, np.uint8)  # distinguishable from the remote data
    with open(os.path.join(root, "0", "0", "0"), "wb") as f:
        f.write(marked.tobytes())
    return root


def test_reuses_foreign_cache(server, data, tmp_path):
    httpd, base = server
    url = f"{base}/blosc.zarr"
    root = _foreign_cache(tmp_path, url, data)
    zattrs_before = open(os.path.join(root, ".zattrs")).read()
    r = ZarrReader.open(url, cache_dir=tmp_path)
    httpd.requests.clear()
    assert (r.read((0, 0, 0), (16, 16, 16)) == 99).all()  # served from the existing cache
    assert httpd.requests == []
    np.testing.assert_array_equal(r.read((16, 0, 0), (16, 16, 16)), data[16:32, :16, :16])  # filled from remote
    r.flush()
    assert os.path.getsize(os.path.join(root, "1", "0", "0")) == 16**3  # appended, decoded
    assert open(os.path.join(root, ".zattrs")).read() == zattrs_before


def test_rejects_incompatible_foreign_cache(server, data, tmp_path):
    _, base = server
    url = f"{base}/blosc.zarr"
    _foreign_cache(tmp_path, url, data, chunks=(8, 8, 8))
    with pytest.raises(ValueError, match="different array"):
        ZarrReader.open(url, cache_dir=tmp_path)


# -- VC3D caches -----------------------------------------------------------------------


def test_vc3d_volume_id_matches_vc3d():
    # Directory VC3D itself created for this volume (fnv1a64 of the URL, as in Volume.cpp).
    url = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
           "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr")
    assert vc3d_volume_id(url) == "20260411134726-2.400um-0.2m-78keV-masked.zarr-6ff766d971e6d4b6"
    assert vc3d_volume_id(url + "/") == vc3d_volume_id(url)


@pytest.fixture(scope="module")
def group_server(tmp_path_factory, data):
    """An OME-Zarr-like group ``vol.zarr`` whose level 0 is blosc-compressed."""
    root = tmp_path_factory.mktemp("www-group")
    (root / "vol.zarr").mkdir()
    (root / "vol.zarr" / ".zgroup").write_text('{"zarr_format": 2}')
    _write_zarr(str(root / "vol.zarr" / "0"), data, CHUNKS, missing=MISSING, compressor=BLOSC)
    httpd, base = _serve(root)
    yield httpd, base, root / "vol.zarr"
    httpd.shutdown()


def _vc3d_mirror(cache_root, group_url, src_group, chunk_keys=()):
    """A VC3D ZarrMirror cache of level 0 holding some of the source's chunk files verbatim."""
    vdir = cache_root / vc3d_volume_id(group_url)
    (vdir / "0").mkdir(parents=True)
    (vdir / ".zgroup").write_bytes((src_group / ".zgroup").read_bytes())
    (vdir / "0" / ".zarray").write_bytes((src_group / "0" / ".zarray").read_bytes())
    for key in chunk_keys:
        dst = vdir / "0" / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((src_group / "0" / key).read_bytes())
    return vdir


def test_vc3d_mirror_read_and_append(group_server, data, tmp_path):
    httpd, base, src = group_server
    group = f"{base}/vol.zarr"
    exp = _expected(data, MISSING, CHUNKS)
    vdir = _vc3d_mirror(tmp_path, group, src, ["0/0/0"])
    (vdir / "0" / "0" / "0" / "1.empty").write_bytes(b"")  # VC3D's missing marker (chunk really exists remotely)

    r = ZarrReader.open(group, level=0, cache_dir=tmp_path)
    assert r.write_layer.kind == "vc3d-mirror"
    httpd.requests.clear()
    np.testing.assert_array_equal(r.read((0, 0, 0), (16, 16, 16)), exp[:16, :16, :16])
    assert (r.read((0, 0, 16), (16, 16, 16)) == 7).all()  # the .empty marker answers: fill
    assert httpd.requests == []

    # A miss is fetched and appended in VC3D's format: the source bytes verbatim, .empty for absent chunks.
    np.testing.assert_array_equal(r.read((0, 16, 0), (16, 16, 16)), exp[:16, 16:32, :16])
    r.read((16, 16, 16), (16, 16, 16))  # (1, 1, 1) is not stored remotely
    r.flush()
    assert (vdir / "0" / "0" / "1" / "0").read_bytes() == (src / "0" / "0" / "1" / "0").read_bytes()
    assert (vdir / "0" / "1" / "1" / "1.empty").exists()
    r.close()


def test_vc3d_mirror_publishes_missing_level_metadata(group_server, tmp_path):
    _, base, src = group_server
    group = f"{base}/vol.zarr"
    vdir = tmp_path / vc3d_volume_id(group)
    vdir.mkdir()  # VC3D has the volume but has not touched level 0 yet
    r = ZarrReader.open(f"{group}/0", cache_dir=tmp_path)  # level inferred from the path
    assert json.loads((vdir / "0" / ".zarray").read_text()) == json.loads((src / "0" / ".zarray").read_text())
    r.close()
    assert ZarrReader.open(f"{group}/0", cache_dir=tmp_path, cache_writes=False).cache_layers  # now found


def test_vc3d_lease(group_server, tmp_path):
    import fcntl

    _, base, src = group_server
    group = f"{base}/vol.zarr"
    vdir = _vc3d_mirror(tmp_path, group, src)
    lock = tmp_path / f".{vdir.name}.vc_cache.lock"
    r = ZarrReader.open(group, level=0, cache_dir=tmp_path)
    with open(lock, "a+b") as fh:  # what a VC3D process replacing the cache would try
        with pytest.raises(BlockingIOError):
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r.close()
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)  # free once the reader is closed
        # While VC3D holds it exclusively, the cache is skipped (and not written to).
        with pytest.warns(UserWarning, match="being replaced"):
            r2 = ZarrReader.open(group, level=0, cache_dir=tmp_path)
        assert all(layer.kind != "vc3d-mirror" for layer in r2.cache_layers)


def test_vc3d_mirror_rejects_changed_source(group_server, tmp_path):
    _, base, src = group_server
    group = f"{base}/vol.zarr"
    vdir = _vc3d_mirror(tmp_path, group, src)
    meta = json.loads((vdir / "0" / ".zarray").read_text())
    meta["compressor"]["clevel"] = 9
    (vdir / "0" / ".zarray").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="mirrors a different array"):
        ZarrReader.open(group, level=0, cache_dir=tmp_path)


def test_several_cache_dirs(group_server, data, tmp_path):
    httpd, base, src = group_server
    group = f"{base}/vol.zarr"
    exp = _expected(data, MISSING, CHUNKS)
    raid, vc3d = tmp_path / "raid", tmp_path / "vc3d"
    raid.mkdir()
    _vc3d_mirror(vc3d, group, src, ["0/0/0"])
    # Only the second root holds the array: reads use it, and appends go to it.
    r = ZarrReader.open(group, level=0, cache_dir=[raid, vc3d])
    assert r.write_layer.root.startswith(str(vc3d))
    httpd.requests.clear()
    np.testing.assert_array_equal(r.read((0, 0, 0), (16, 16, 16)), exp[:16, :16, :16])
    assert httpd.requests == []
    assert not any(raid.iterdir())
    r.close()
    # With neither holding it, a new decoded cache is made under the first root.
    r = ZarrReader.open(f"{base}/vol.zarr", level=0, cache_dir=[tmp_path / "a", tmp_path / "b"])
    assert r.write_layer.kind == "decoded" and r.write_layer.root.startswith(str(tmp_path / "a"))


def test_vc3d_delta3d_cache(group_server, data, tmp_path):
    vc_delta3d = pytest.importorskip("vc_delta3d")  # only to write the fixture's payloads
    httpd, base, src = group_server
    group = f"{base}/vol.zarr"
    exp = _expected(data, MISSING, CHUNKS)
    source_meta = json.loads((src / "0" / ".zarray").read_text())
    vdir = tmp_path / vc3d_volume_id(group)
    (vdir / "0" / "0" / "0").mkdir(parents=True)
    (vdir / VC3D_DELTA3D_MARKER).write_bytes(b"D3D1\n")
    (vdir / "0" / ".zarray").write_text(json.dumps(delta3d_zarray(ArrayMeta.from_zarray(source_meta))))
    (vdir / "0" / "0" / "0" / "0").write_bytes(vc_delta3d.Delta3D(quant=1).encode(exp[:16, :16, :16].copy()))
    lossy = vc_delta3d.Delta3D(quant=5).encode(exp[:16, :16, 16:32].copy())
    (vdir / "0" / "0" / "0" / "1").write_bytes(lossy)
    edge = np.full(CHUNKS, 7, np.uint8)  # stored edge chunks are full-size, padded with fill
    edge[:, :, :13] = exp[:16, :16, 32:45]
    (vdir / "0" / "0" / "0" / "2").write_bytes(vc_delta3d.Delta3D(quant=5).encode(edge))

    r = ZarrReader.open(group, level=0, cache_dir=tmp_path)
    assert r.write_layer.kind == "vc3d-delta3d"
    httpd.requests.clear()
    np.testing.assert_array_equal(r.read((0, 0, 0), (16, 16, 16)), exp[:16, :16, :16])
    assert httpd.requests == []
    # A quantized chunk is not the source's values: refetched exactly, and the cache entry replaced.
    np.testing.assert_array_equal(r.read((0, 0, 16), (16, 16, 16)), exp[:16, :16, 16:32])
    assert len(httpd.requests) == 1
    r.flush()
    assert (vdir / "0" / "0" / "0" / "1").read_bytes()[6] <= 1
    # Appends are Delta3D-encoded (losslessly); misses are marked under .vc_cache_empty.
    r.read((0, 16, 0), (16, 16, 16))
    r.read((16, 16, 16), (16, 16, 16))
    r.flush()
    stored = (vdir / "0" / "0" / "1" / "0").read_bytes()
    assert stored[:4] == b"D3D1" and stored[6] <= 1
    np.testing.assert_array_equal(
        np.frombuffer(vc_delta3d.Delta3D().decode(stored), np.uint8).reshape(CHUNKS), exp[:16, 16:32, :16])
    assert (vdir / ".vc_cache_empty" / "0" / "1" / "1" / "1.empty").exists()
    r.close()
    # With lossy caches allowed, a quantized chunk is served as stored (within its +-2 bound).
    r = ZarrReader.open(group, level=0, cache_dir=tmp_path, allow_lossy_cache=True)
    httpd.requests.clear()
    got = r.read((0, 0, 32), (16, 16, 13))
    assert httpd.requests == []
    err = np.abs(got.astype(int) - exp[:16, :16, 32:45])
    assert 0 < err.max() <= 2


def _forked_child(reader, root, q):
    try:
        a = reader.read((0, 0, 0), (16, 16, 32))  # rebuilds the native engine in this process
        reader.flush()
        reader.close()
        q.put(("ok", int(a.sum()), os.path.getsize(os.path.join(root, "0/0/1"))))
    except BaseException as exc:  # pragma: no cover - reported to the parent
        q.put(("error", repr(exc), None))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")  # the point of the test
def test_reader_survives_fork(server, data, tmp_path):
    import multiprocessing as mp

    _, base = server
    url = f"{base}/raw.zarr"
    exp = _expected(data, MISSING, CHUNKS)
    r = ZarrReader.open(url, cache_dir=tmp_path)
    r.read((16, 16, 0), (16, 16, 16))  # the parent's threads are running
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_forked_child, args=(r, decoded_cache_path(tmp_path, url), q))
    p.start()
    p.join(60)
    if p.is_alive():
        p.kill()
        pytest.fail("the forked reader hung")
    status, total, size = q.get(timeout=5)
    assert status == "ok", total
    assert total == int(exp[:16, :16, :32].sum()) and size == 16**3  # read, and cached by the child
    # The parent is unaffected.
    np.testing.assert_array_equal(r.read((0, 0, 0), (16, 16, 16)), exp[:16, :16, :16])
    r.close()
