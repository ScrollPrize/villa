import multiprocessing as mp
import pickle

import numpy as np
import pytest
import zarr

from vc_reader.reader import ZarrWriter, create_group, open_node, open_zarr


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
