from __future__ import annotations

import functools
import http.server
import json
import os
import pickle
import threading

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("urllib3")

from vc_reader.reader import MemoryChunkCache, ZarrReader  # noqa: E402
from vc_reader.reader.disk_cache import MISSING_SUFFIX, decoded_cache_path  # noqa: E402

SHAPE = (37, 50, 45)
CHUNKS = (16, 16, 16)


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
VARIANTS = {
    "raw": dict(),
    "raw-dot-sep": dict(sep="."),
    "blosc": dict(compressor={"id": "blosc", "cname": "zstd", "clevel": 1, "shuffle": 1, "blocksize": 0}),
    "zlib+delta": dict(compressor={"id": "zlib", "level": 1}, filters=[{"id": "delta", "dtype": "|u1"}]),
    "fortran": dict(order="F"),
}


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


@pytest.mark.parametrize("read_mode", ["slab", "pread", "mmap"])
@pytest.mark.parametrize("memory", [0, 64 << 20])
def test_local_reads(local_array, memory, read_mode):
    root, exp = local_array
    with ZarrReader.open(root, memory_cache=memory, read_mode=read_mode) as r:
        assert r.shape == SHAPE and r.chunks == CHUNKS and r.fill_value == 7
        for start, size in REGIONS:
            for _ in range(2):  # second pass hits the memory cache when enabled
                np.testing.assert_array_equal(r.read(start, size), _ref(exp, start, size))


@pytest.mark.parametrize("read_mode", ["slab", "pread", "mmap"])
def test_read_many(local_array, read_mode):
    root, exp = local_array
    r = ZarrReader.open(root, read_mode=read_mode, workers=4)
    outs = r.read_many(REGIONS)
    for (start, size), got in zip(REGIONS, outs):
        np.testing.assert_array_equal(got, _ref(exp, start, size))


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


def test_chunk_and_cache_are_read_only(local_array):
    root, exp = local_array
    r = ZarrReader.open(root, memory_cache=1 << 20)
    c = r.chunk((0, 0, 0))
    np.testing.assert_array_equal(c, exp[:16, :16, :16])
    assert r.chunk((1, 1, 1)) is None
    assert not r.chunk((0, 0, 0)).flags.writeable
    assert r.memory_cache.hits >= 1


def test_memory_cache_eviction():
    cache = MemoryChunkCache(max_bytes=3000)
    for i in range(5):
        cache.put(("a", (i,)), np.zeros(1000, np.uint8))
    assert len(cache) == 3 and cache.nbytes == 3000
    cache.put(("a", (9,)), None)
    assert cache.nbytes <= 3000


def test_pickle_roundtrip(local_array):
    root, exp = local_array
    r = ZarrReader.open(root, memory_cache=1 << 20)
    r.read((0, 0, 0), (8, 8, 8))
    r2 = pickle.loads(pickle.dumps(r))
    assert len(r2.memory_cache) == 0 and r2.memory_cache.max_bytes == 1 << 20
    np.testing.assert_array_equal(r2[...], exp)


# -- HTTP and the disk cache ------------------------------------------------------


class _Handler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, like S3
    log_message = lambda *a: None  # noqa: E731

    def do_GET(self):  # noqa: N802
        self.server.requests.append(self.path)
        super().do_GET()


@pytest.fixture(scope="module")
def server(tmp_path_factory, data):
    root = tmp_path_factory.mktemp("www")
    _write_zarr(str(root / "raw.zarr"), data, CHUNKS, missing=MISSING)
    _write_zarr(str(root / "blosc.zarr"), data, CHUNKS, missing=MISSING, compressor=VARIANTS["blosc"]["compressor"])
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Handler, directory=str(root)))
    httpd.requests = []
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.mark.parametrize("name", ["raw.zarr", "blosc.zarr"])
def test_http_reads(server, data, name):
    httpd, base = server
    exp = _expected(data, MISSING, CHUNKS)
    with ZarrReader.open(f"{base}/{name}") as r:
        for start, size in REGIONS:
            np.testing.assert_array_equal(r.read(start, size), _ref(exp, start, size))


def test_disk_cache(server, data, tmp_path):
    httpd, base = server
    exp = _expected(data, MISSING, CHUNKS)
    url = f"{base}/blosc.zarr"
    r = ZarrReader.open(url, cache_dir=tmp_path)
    np.testing.assert_array_equal(r[...], exp)
    root = decoded_cache_path(tmp_path, url)
    assert os.path.exists(os.path.join(root, "1/1/1" + MISSING_SUFFIX))
    assert os.path.getsize(os.path.join(root, "0/0/0")) == 16**3  # stored decoded

    # A second reader is served from disk: only the metadata request goes out.
    httpd.requests.clear()
    r2 = ZarrReader.open(url, cache_dir=tmp_path)
    np.testing.assert_array_equal(r2[...], exp)
    assert httpd.requests == ["/blosc.zarr/.zarray"]

    # The cache is a plain uncompressed zarr array.
    np.testing.assert_array_equal(ZarrReader.open(root)[...], exp)
    zarr = pytest.importorskip("zarr")
    np.testing.assert_array_equal(zarr.open_array(root, mode="r")[...], exp)


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


def test_concurrent_reads_fetch_each_chunk_once(server, data, tmp_path):
    httpd, base = server
    url = f"{base}/raw.zarr"
    r = ZarrReader.open(url, cache_dir=tmp_path, workers=8)
    httpd.requests.clear()
    threads = [threading.Thread(target=r.read, args=((0, 0, 0), SHAPE)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    chunk_requests = [p for p in httpd.requests if not p.endswith(".zarray")]
    assert len(chunk_requests) == len(set(chunk_requests))


def test_http_error_is_raised(server):
    _, base = server
    with pytest.raises(OSError):
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
    assert open(os.path.join(root, ".zattrs")).read() == zattrs_before


def test_rejects_incompatible_foreign_cache(server, data, tmp_path):
    _, base = server
    url = f"{base}/blosc.zarr"
    _foreign_cache(tmp_path, url, data, chunks=(8, 8, 8))
    with pytest.raises(ValueError, match="different array"):
        ZarrReader.open(url, cache_dir=tmp_path)


# -- VC3D caches -----------------------------------------------------------------------

from vc_reader.reader.disk_cache import (  # noqa: E402
    VC3D_DELTA3D_MARKER, Vc3dMirrorLayer, open_layers, vc3d_delta3d_zarray, vc3d_volume_id)


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
    _write_zarr(str(root / "vol.zarr" / "0"), data, CHUNKS, missing=MISSING, compressor=VARIANTS["blosc"]["compressor"])
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Handler, directory=str(root)))
    httpd.requests = []
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd, f"http://127.0.0.1:{httpd.server_address[1]}", root / "vol.zarr"
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
    assert isinstance(r.source.write_layer, Vc3dMirrorLayer)
    httpd.requests.clear()
    np.testing.assert_array_equal(r.read((0, 0, 0), (16, 16, 16)), exp[:16, :16, :16])
    assert (r.read((0, 0, 16), (16, 16, 16)) == 7).all()  # the .empty marker answers: fill
    assert httpd.requests == []

    # A miss is fetched and appended in VC3D's format: the source bytes verbatim, .empty for absent chunks.
    np.testing.assert_array_equal(r.read((0, 16, 0), (16, 16, 16)), exp[:16, 16:32, :16])
    assert (vdir / "0" / "0" / "1" / "0").read_bytes() == (src / "0" / "0" / "1" / "0").read_bytes()
    r.read((16, 16, 16), (16, 16, 16))  # (1, 1, 1) is not stored remotely
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
    assert ZarrReader.open(f"{group}/0", cache_dir=tmp_path, cache_writes=False).source.layers  # now found


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
        assert not any(isinstance(layer, Vc3dMirrorLayer) for layer in r2.source.layers)


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
    assert r.source.write_layer.root.startswith(str(vc3d))
    httpd.requests.clear()
    np.testing.assert_array_equal(r.read((0, 0, 0), (16, 16, 16)), exp[:16, :16, :16])
    assert httpd.requests == []
    assert not any(raid.iterdir())
    r.close()
    # With neither holding it, a new decoded cache is made under the first root.
    r = ZarrReader.open(f"{base}/vol.zarr", level=0, cache_dir=[tmp_path / "a", tmp_path / "b"])
    assert r.source.write_layer.kind == "decoded" and r.source.root.startswith(str(tmp_path / "a"))


def test_vc3d_delta3d_cache(group_server, data, tmp_path):
    vc_delta3d = pytest.importorskip("vc_delta3d")
    httpd, base, src = group_server
    group = f"{base}/vol.zarr"
    exp = _expected(data, MISSING, CHUNKS)
    source_meta = json.loads((src / "0" / ".zarray").read_text())
    vdir = tmp_path / vc3d_volume_id(group)
    (vdir / "0" / "0" / "0").mkdir(parents=True)
    (vdir / VC3D_DELTA3D_MARKER).write_bytes(b"D3D1\n")
    from vc_reader.reader.zarr_meta import ArrayMeta

    (vdir / "0" / ".zarray").write_text(json.dumps(vc3d_delta3d_zarray(ArrayMeta.from_zarray(source_meta))))
    (vdir / "0" / "0" / "0" / "0").write_bytes(vc_delta3d.Delta3D(quant=1).encode(exp[:16, :16, :16].copy()))
    lossy = vc_delta3d.Delta3D(quant=5).encode(exp[:16, :16, 16:32].copy())
    (vdir / "0" / "0" / "0" / "1").write_bytes(lossy)

    r = ZarrReader.open(group, level=0, cache_dir=tmp_path)
    assert r.source.write_layer.kind == "vc3d-delta3d"
    httpd.requests.clear()
    np.testing.assert_array_equal(r.read((0, 0, 0), (16, 16, 16)), exp[:16, :16, :16])
    assert httpd.requests == []
    # A quantized chunk is not the source's values: refetched, exact.
    np.testing.assert_array_equal(r.read((0, 0, 16), (16, 16, 16)), exp[:16, :16, 16:32])
    assert len(httpd.requests) == 1
    # Appends are Delta3D-encoded; misses are marked under .vc_cache_empty.
    r.read((0, 16, 0), (16, 16, 16))
    stored = (vdir / "0" / "0" / "1" / "0").read_bytes()
    assert stored[:4] == b"D3D1"
    r.read((16, 16, 16), (16, 16, 16))
    assert (vdir / ".vc_cache_empty" / "0" / "1" / "1" / "1.empty").exists()
