import multiprocessing as mp
import pickle

import numpy as np
import pytest
import zarr

from vc_reader.reader import (ZarrReader, ZarrWriter, create_group, create_pyramid_levels, ome_multiscales, open_node, open_zarr,
                              write_pyramid, write_pyramid_level)


def test_writes_read_back_through_zarr_and_vc_reader(tmp_path):
    w = ZarrWriter.create(tmp_path / "a", shape=(5, 6), chunks=(2, 3), dtype="u2", cname="zstd", shuffle=2, fill_value=7)
    data = np.arange(30, dtype=np.uint16).reshape(5, 6)
    w[:] = data
    w[1, 2] = 99
    w[3:5, :3] = 1
    data[1, 2] = 99
    data[3:5, :3] = 1
    np.testing.assert_array_equal(zarr.open_array(tmp_path / "a", mode="r")[:], data)
    np.testing.assert_array_equal(open_zarr(tmp_path / "a")[:], data)
    np.testing.assert_array_equal(w[:], data)
    assert w.shape == (5, 6) and w.chunks == (2, 3) and w.dtype == np.uint16 and w.fill_value == 7
    compressor = zarr.open_array(tmp_path / "a", mode="r").metadata.to_dict()["compressor"]
    assert compressor["cname"] == "zstd" and compressor["shuffle"] == 2


def test_overwrite_skip_empty_and_attributes(tmp_path):
    w = ZarrWriter.create(tmp_path / "a", shape=(4, 4), chunks=(2, 2), dtype="u1")
    w[:] = 1
    with pytest.raises(ValueError):
        w[:, :] = np.zeros((3, 3), np.uint8)
    w = ZarrWriter.create(tmp_path / "a", shape=(4, 4), chunks=(2, 2), dtype="u1", overwrite=True, skip_empty=True)
    assert not w.chunk_exists((0, 0))
    w[0:2, 0:2] = 0
    assert not w.chunk_exists((0, 0))
    w.attrs["a"] = {"b": [1, 2]}
    w.attrs.update(c=3)
    del w.attrs["c"]
    assert dict(w.attrs) == {"a": {"b": [1, 2]}}
    assert open_zarr(tmp_path / "a").attrs == {"a": {"b": [1, 2]}}


def test_group_children_and_removal(tmp_path):
    g = create_group(tmp_path / "g", attrs={"x": 1})
    child = g.create_array("0", shape=(2,), chunks=(2,), dtype="f4", compressor=None)
    child[:] = 1.5
    sub = g.create_group("sub")
    sub.create_array("deep", shape=(1,), chunks=(1,), dtype="u1")
    assert "0" in g and "sub" in g and "nope" not in g
    assert g.keys() == ["0", "sub"]
    assert isinstance(g["0"], ZarrWriter) and g["sub"].keys() == ["deep"]
    read = open_node(tmp_path / "g")
    assert read.array_keys() == ["0"] and read.attrs == {"x": 1}
    np.testing.assert_array_equal(read["0"][:], [1.5, 1.5])
    del g["sub"]
    assert g.keys() == ["0"]
    with pytest.raises(KeyError):
        del g["sub"]
    assert create_group(tmp_path / "g", overwrite=True).keys() == []


def _write_row(args):
    writer, row = args
    writer[row, :] = row


def test_pickled_writers_write_distinct_chunks_from_processes(tmp_path):
    w = ZarrWriter.create(tmp_path / "a", shape=(8, 4), chunks=(1, 4), dtype="u1")
    assert pickle.loads(pickle.dumps(w)).shape == (8, 4)
    with mp.get_context("spawn").Pool(2) as pool:
        pool.map(_write_row, [(w, row) for row in range(8)])
    np.testing.assert_array_equal(open_zarr(tmp_path / "a")[:, 0], np.arange(8))


def test_v3_sharded(tmp_path):
    w = ZarrWriter.create(tmp_path / "a", shape=(8, 8), chunks=(2, 2), dtype="u1", version=3, shards=(4, 4), fill_value=0)
    w[:] = 5
    assert zarr.open_array(tmp_path / "a", mode="r").shards == (4, 4)
    np.testing.assert_array_equal(open_zarr(tmp_path / "a")[:], np.full((8, 8), 5))


def test_create_like_keeps_layout_and_codec(tmp_path):
    source = ZarrWriter.create(tmp_path / "a", shape=(8, 8), chunks=(4, 4), dtype="u2", cname="zstd", shuffle=2, level=5,
                               separator="/", fill_value=3)
    copy = ZarrWriter.create_like(tmp_path / "b", ZarrReader.open(source.path))
    assert copy.shape == (8, 8) and copy.chunks == (4, 4) and copy.dtype == np.uint16 and copy.fill_value == 3
    assert ZarrReader.open(copy.path).compressor == ZarrReader.open(source.path).compressor
    assert ZarrReader.open(copy.path).meta.separator == "/"
    changed = ZarrWriter.create_like(tmp_path / "c", source, shape=(2, 2), chunks=(1, 1), dtype="u1", fill_value=0, separator=".")
    assert changed.shape == (2, 2) and changed.dtype == np.uint8 and changed.fill_value == 0
    assert ZarrReader.open(changed.path).meta.separator == "."


@pytest.mark.parametrize(("reduction", "expected"), [("mean", 6), ("nearest", 7), ("mode", 5)])
def test_write_pyramid_reductions(tmp_path, reduction, expected):
    level0 = ZarrWriter.create(tmp_path / "0", shape=(2, 2, 2), chunks=(2, 2, 2), dtype="u1")
    level0[...] = np.array([7, 7, 0, 5, 5, 5, 8, 8], np.uint8).reshape(2, 2, 2)
    write_pyramid(tmp_path, 1, reduction=reduction, overwrite=True)
    assert ZarrReader.open(tmp_path / "1")[0, 0, 0] == expected


def test_write_pyramid_level_ranges_and_multiscales(tmp_path):
    level0 = ZarrWriter.create(tmp_path / "0", shape=(1, 5, 5), chunks=(1, 2, 2), dtype="u1")
    level0[...] = 9
    create_pyramid_levels(tmp_path, 1, factors=(1, 2, 2))
    assert ZarrReader.open(tmp_path / "1").shape == (1, 3, 3)
    write_pyramid_level(tmp_path, 1, factors=(1, 2, 2), chunk_range=((0, 0, 0), (1, 1, 99)))
    level1 = ZarrReader.open(tmp_path / "1")
    assert level1[0, 0, 0] == 9 and level1[0, 2, 0] == 0
    entry = ome_multiscales(levels=1, factors=(1, 2, 2), reduction="nearest", axes=("x", "z", "y"))
    assert [axis["name"] for axis in entry["axes"]] == ["x", "z", "y"]
    assert entry["datasets"][1]["coordinateTransformations"][1]["translation"] == [0.0, 0.0, 0.0]


def test_writer_converts_to_numpy(tmp_path):
    writer = ZarrWriter.create(tmp_path / "a", shape=(2, 2), chunks=(2, 2), dtype="u1")
    writer[0, 1] = 5
    np.testing.assert_array_equal(np.asarray(writer), [[0, 5], [0, 0]])


@pytest.mark.parametrize("fill", [0.0, -0.0, 1.5, 0.1, 1e-7, 6.1e-5, 6.104e-5, 1023.5, 65504.0, 65519.0, 65520.0, "Infinity", "-Infinity", "NaN"])
def test_float16_fill_matches_numpy(tmp_path, fill):
    writer = ZarrWriter.create(tmp_path / "a", shape=(2,), chunks=(2,), dtype=np.float16, fill_value=fill)
    expected = np.float16(float(fill))
    assert writer.fill_value.tobytes() == expected.tobytes()


def test_float16_arrays_round_trip_with_zarr(tmp_path):
    data = np.array([[0.5, -1.25], [65504.0, 0.1]], np.float16)
    writer = ZarrWriter.create(tmp_path / "a", shape=data.shape, chunks=(1, 2), dtype=np.float16, cname="zstd", level=3)
    writer[...] = data
    np.testing.assert_array_equal(zarr.open_array(tmp_path / "a", mode="r")[:], data)
    reader = open_zarr(tmp_path / "a")
    assert reader.dtype == np.float16
    np.testing.assert_array_equal(reader[...], data)


@pytest.mark.parametrize("version", [2, 3])
def test_packbits_reads_and_writes_match_zarr(tmp_path, version):
    from zarr.codecs import BloscCodec, BloscShuffle, PackBits

    data = np.random.default_rng(0).random((4, 5, 3)) > 0.5
    w = ZarrWriter.create(tmp_path / "ours", shape=data.shape, chunks=(1, 5, 3), dtype=bool, packbits=True, cname="zstd",
                          level=1, fill_value=False, version=version, separator="/",
                          shards=(2, 5, 3) if version == 3 else None)
    w[:] = data
    np.testing.assert_array_equal(open_zarr(tmp_path / "ours")[:], data)
    assert np.asarray(zarr.open_array(tmp_path / "ours", mode="r")[:]).tolist() == data.tolist()
    like = ZarrWriter.create_like(tmp_path / "like", w)
    like[:] = data
    np.testing.assert_array_equal(open_zarr(tmp_path / "like")[:], data)
    if version == 3:
        theirs = zarr.create_array(tmp_path / "theirs", shape=data.shape, chunks=(1, 5, 3), shards=(2, 5, 3), dtype="bool",
                                   filters=[PackBits()], compressors=[BloscCodec(cname="zstd", clevel=1, shuffle=BloscShuffle.shuffle)])
        theirs[:] = data
        np.testing.assert_array_equal(open_zarr(tmp_path / "theirs")[:], data)
        np.testing.assert_array_equal(open_zarr(tmp_path / "theirs")[2, 1:3], data[2, 1:3])
        again = ZarrWriter.open(tmp_path / "theirs")
        again[0, 0, 0] = not data[0, 0, 0]
        assert bool(zarr.open_array(tmp_path / "theirs", mode="r")[0, 0, 0]) != bool(data[0, 0, 0])


def test_bytes_codecs_match_numcodecs():
    from numcodecs import Blosc

    from vc_reader.reader import compress_bytes, decompress_bytes

    records = np.arange(4096, dtype=np.uint64)
    config = {"id": "blosc", "cname": "zstd", "clevel": 1, "shuffle": 1}
    encoded = compress_bytes(records.tobytes(), config, itemsize=8)
    theirs = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
    assert np.frombuffer(theirs.decode(encoded), np.uint64).tolist() == records.tolist()
    decoded = decompress_bytes(theirs.encode(records), config, records.nbytes)
    assert np.frombuffer(decoded, np.uint64).tolist() == records.tolist()
    with pytest.raises(RuntimeError):
        decompress_bytes(encoded, config, records.nbytes - 1)


def test_stored_chunks_list_the_store(tmp_path):
    w = ZarrWriter.create(tmp_path / "a.zarr", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="|u1", compressor=None)
    assert w.stored_chunks() == []
    w[0:4, 4:8, 0:4] = 1
    w[4:8, 0:4, 4:8] = 2
    assert w.stored_chunks() == [(0, 1, 0), (1, 0, 1)]
    v3 = ZarrWriter.create(tmp_path / "b.zarr", shape=(8, 8), chunks=(4, 4), dtype="|u1", version=3)
    v3[4:8, 0:4] = 1
    assert v3.stored_chunks() == [(1, 0)]
    sharded = ZarrWriter.create(tmp_path / "c.zarr", shape=(8, 8), chunks=(2, 2), dtype="|u1", version=3, shards=(4, 4))
    sharded[2:4, 4:6] = 1
    assert sharded.stored_chunks() == [(1, 2)]


def test_a_token_stays_at_the_end_of_every_url(tmp_path):
    import http.server, threading, urllib.parse
    from vc_reader.reader import open_group, open_zarr

    expected = np.arange(512, dtype=np.uint8).reshape(8, 8, 8)
    group = create_group(tmp_path / "g.zarr")
    group.create_array("0", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="|u1", compressor=None)[:] = expected
    denied = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(tmp_path), **kwargs)

        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            if url.query != "token=a%2Fb":
                denied.append(self.path)
                self.send_error(403)
                return
            self.path = url.path
            super().do_GET()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_port}/g.zarr"
        assert np.array_equal(open_zarr(f"{base}/0?token=a%2Fb")[:], expected)
        assert np.array_equal(open_zarr(f"{base}/0/?token=a%2Fb")[:], expected)
        assert np.array_equal(open_group(f"{base}?token=a%2Fb")["0"][:], expected)
        assert denied == []
    finally:
        server.shutdown()
