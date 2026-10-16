"""ZarrGroup: OME-Zarr levels, children, listing, offline groups."""

from __future__ import annotations

import json
import pickle

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("vc_reader._native", reason="the native extension is not built")

from test_access import _creds, no_aws_environment, s3  # noqa: E402,F401  (fixtures)
from test_zarr_reader import BLOSC, CHUNKS, MISSING, _expected, _serve, _write_zarr  # noqa: E402

from vc_reader.errors import CacheMissError  # noqa: E402
from vc_reader.reader import ZarrGroup, ZarrReader, open_group  # noqa: E402

SHAPE = (37, 50, 45)


@pytest.fixture(scope="module")
def data():
    return np.random.default_rng(0).integers(0, 255, SHAPE, dtype=np.uint8)


def _downsample(a):
    return a[::2, ::2, ::2].copy()


def _make_group(root, data, *, multiscales=True):
    """vol.zarr: levels 0..2 (level 1 blosc), a "labels" subgroup with array "ink", and a stray directory."""
    pytest.importorskip("numcodecs")
    g = root / "vol.zarr"
    g.mkdir(parents=True)
    (g / ".zgroup").write_text('{"zarr_format": 2}')
    levels = [data, _downsample(data), _downsample(_downsample(data))]
    for i, lvl in enumerate(levels):
        _write_zarr(str(g / str(i)), lvl, CHUNKS, compressor=BLOSC if i == 1 else None)
    if multiscales:
        datasets = [{"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": [2.0 ** i] * 3}]}
                    for i in range(3)]
        (g / ".zattrs").write_text(json.dumps({"multiscales": [{"version": "0.4", "datasets": datasets}]}))
    (g / "labels").mkdir()
    (g / "labels" / ".zgroup").write_text('{"zarr_format": 2}')
    (g / "labels" / ".zattrs").write_text('{"kind": "ink"}')
    _write_zarr(str(g / "labels" / "ink"), (data > 128).astype(np.uint8), CHUNKS)
    (g / "notes").mkdir()  # not a zarr node
    return g, levels


def test_local_group(tmp_path, data):
    g_path, levels = _make_group(tmp_path, data)
    with ZarrGroup.open(g_path, memory_cache=32 << 20) as g:
        assert g.levels == ("0", "1", "2")
        assert g.multiscales[0]["datasets"][2]["coordinateTransformations"][0]["scale"] == [4.0] * 3
        for i, lvl in enumerate(levels):
            np.testing.assert_array_equal(g.level(i)[...], lvl)
        np.testing.assert_array_equal(g.level(-1)[...], levels[2])
        assert g.level(1) is g.level(1) and g.level(0).memory_cache is g.memory_cache is g.level(2).memory_cache
        assert g.keys() == ["0", "1", "2", "labels"]
        assert "labels" in g and "labels/ink" in g and "notes" not in g and "nope" not in g and "../x" not in g
        labels = g["labels"]
        assert isinstance(labels, ZarrGroup) and labels.attrs == {"kind": "ink"} and labels.keys() == ["ink"]
        np.testing.assert_array_equal(g["labels/ink"][...], (data > 128).astype(np.uint8))
        assert isinstance(g["0"], ZarrReader) and g.get("nope") is None
        with pytest.raises(KeyError):
            g["notes"]
        with pytest.raises(KeyError):
            g.array("labels")
        with pytest.raises(IndexError):
            g.level(3)


def test_levels_without_ome_metadata(tmp_path, data):
    g_path, levels = _make_group(tmp_path, data, multiscales=False)
    with open_group(g_path) as g:
        assert g.attrs == {} and g.multiscales == [] and g.levels == ("0", "1", "2")
        np.testing.assert_array_equal(g.level(2)[...], levels[2])


def test_open_errors(tmp_path, data):
    g_path, _ = _make_group(tmp_path, data)
    with pytest.raises(ValueError, match="is an array"):
        ZarrGroup.open(g_path / "0")
    with pytest.raises(FileNotFoundError):
        ZarrGroup.open(tmp_path / "missing.zarr")


def test_group_pickles(tmp_path, data):
    g_path, levels = _make_group(tmp_path, data)
    g = ZarrGroup.open(g_path, memory_cache=1 << 20)
    g.level(0)
    g2 = pickle.loads(pickle.dumps(g))
    assert g2.levels == g.levels and g2.memory_cache.max_bytes == 1 << 20
    np.testing.assert_array_equal(g2.level(1)[...], levels[1])
    g.close()
    g2.close()


def test_http_group(tmp_path, data):
    g_path, levels = _make_group(tmp_path / "www", data)
    httpd, base = _serve(tmp_path / "www")
    try:
        with ZarrGroup.open(f"{base}/vol.zarr", cache_dir=tmp_path / "cache") as g:
            assert g.levels == ("0", "1", "2")
            np.testing.assert_array_equal(g.level(1).read((0, 0, 0), (8, 8, 8)), levels[1][:8, :8, :8])
            assert isinstance(g["labels"], ZarrGroup)
            with pytest.raises(ValueError, match="plain HTTP"):
                g.keys()
        # Consolidated metadata lists the children without a listing API.
        meta = {"metadata": {".zgroup": {"zarr_format": 2}, "0/.zarray": json.loads((g_path / "0" / ".zarray").read_text()),
                             "labels/.zgroup": {"zarr_format": 2}, "labels/ink/.zarray": {}}, "zarr_consolidated_format": 1}
        (g_path / ".zmetadata").write_text(json.dumps(meta))
        with ZarrGroup.open(f"{base}/vol.zarr") as g:
            httpd.requests.clear()
            assert g.keys() == ["0", "labels"]
            np.testing.assert_array_equal(g["0"].read((0, 0, 0), (4, 4, 4)), data[:4, :4, :4])
            assert not any(p.endswith(".zarray") for p in httpd.requests)  # metadata came from .zmetadata
    finally:
        httpd.shutdown()


def test_offline_group(tmp_path, data):
    g_path, levels = _make_group(tmp_path / "www", data)
    httpd, base = _serve(tmp_path / "www")
    cache = tmp_path / "cache"
    try:
        with ZarrGroup.open(f"{base}/vol.zarr", cache_dir=cache) as g:
            g.level(0).read((0, 0, 0), (16, 16, 16))
            g.level(2).read((0, 0, 0), (10, 13, 12))
        httpd.requests.clear()
        with ZarrGroup.open(f"{base}/vol.zarr", cache_dir=cache, offline=True) as g:
            assert g.offline and g.levels == ("0", "1", "2")  # from the cached .zattrs
            np.testing.assert_array_equal(g.level(0).read((0, 0, 0), (16, 16, 16)), data[:16, :16, :16])
            np.testing.assert_array_equal(g.level(2)[...], levels[2])
            assert g.keys() == ["0", "2"]  # what the cache holds
            assert "1" not in g
            with pytest.raises(CacheMissError):
                g.level(0).read((16, 0, 0), (1, 1, 1))
        assert httpd.requests == []
        with pytest.raises(CacheMissError):
            ZarrGroup.open(f"{base}/other.zarr", cache_dir=cache, offline=True)
    finally:
        httpd.shutdown()


def test_signed_s3_group_listing(s3, data):
    httpd, base = s3
    with ZarrGroup.open("s3://bkt/vol.zarr", auth=_creds(base)) as g:
        httpd.requests.clear()
        assert g.keys() == ["0", "labels"]  # listed through ListObjectsV2 (one prefix per page); "notes" is no node
        assert sum("list-type=2" in p for p, _ in httpd.requests) == 3
        assert g.levels == ("0",)
        np.testing.assert_array_equal(g.level(0).read((0, 0, 0), SHAPE), _expected(data, MISSING, CHUNKS))
