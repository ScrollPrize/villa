"""Affine math, artifact URLs, metadata fetching/caching and the CLI."""

from __future__ import annotations

import gzip
import json
import urllib.error

import pytest

from conftest import FIXTURE
from vc_reader import Affine, Artifact, Catalog, _fetch, to_https, to_s3
from vc_reader.cli import main


def test_affine_compose_and_inverse():
    a = Affine.from_rows([[0.0, -2.0, 0.0, 5.0], [2.0, 0.0, 0.0, -1.0], [0.0, 0.0, 4.0, 3.0]])
    b = Affine.scale(0.5)
    p = (1.0, 2.0, 3.0)
    assert (a @ b).apply(p) == a.apply(b.apply(p))
    assert a.inverse().apply(a.apply(p)) == pytest.approx(p)
    assert (a.inverse() @ a).apply(p) == pytest.approx(p)
    with pytest.raises(ValueError):
        Affine.from_rows([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        Affine.scale(0).inverse()


def test_url_rewrites():
    s3 = "s3://vesuvius-challenge-open-data/PHercParis4/x.zarr/"
    https = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/x.zarr/"
    assert to_https(s3) == https and to_s3(https) == s3
    assert to_https("s3://vesuvius-challenge/PHerc0332/a.tif") == "https://data.aws.ash2txt.org/samples/PHerc0332/a.tif"
    assert to_https("s3://some-other-bucket/a") is None
    assert to_s3("https://example.org/a") is None


def test_artifact_url_preference():
    entry = {
        "type": "ome-zarr",
        "origins": [
            {"path": "samples/S/v.zarr/", "access_roots": [{"type": "https", "url": "https://data.aws.ash2txt.org", "usage": "public-read"}]},
            {"path": "S/v.zarr/", "access_roots": [{"type": "s3", "url": "s3://vesuvius-challenge-open-data", "usage": "public-read"}]},
        ],
    }
    a = Artifact.from_json(entry)
    assert a.is_directory
    assert a.url() == "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/S/v.zarr/"
    assert a.url("s3") == "s3://vesuvius-challenge-open-data/S/v.zarr/"
    assert a.url(None) == "s3://vesuvius-challenge-open-data/S/v.zarr/"
    assert a.urls() == ("https://data.aws.ash2txt.org/samples/S/v.zarr/", "s3://vesuvius-challenge-open-data/S/v.zarr/")
    with pytest.raises(ValueError):
        a.url("ftp")


def test_unknown_artifact_types_are_kept():
    a = Artifact.from_json({"type": "something-new", "origins": [], "parameters": {"level": 3}})
    assert a.type == "something-new" and a.grid_level is None and a.url() is None


def test_load_gzipped_file(tmp_path):
    gz = tmp_path / "metadata.json"
    gz.write_bytes(gzip.compress(FIXTURE.read_bytes()))
    assert set(Catalog.load(gz).samples) == set(Catalog.from_file(FIXTURE).samples)


class _FakeServer:
    """Stands in for _fetch._get; counts requests and serves 200/304/errors."""

    def __init__(self, body: bytes):
        self.body, self.calls, self.fail = body, [], False

    def __call__(self, url, headers, timeout):
        self.calls.append(dict(headers))
        if self.fail:
            raise urllib.error.URLError("offline")
        if headers.get("If-None-Match") == '"v1"':
            raise _fetch._NotModified()
        return self.body, {"ETag": '"v1"'}


def test_fetch_cache_cycle(tmp_path, monkeypatch):
    body = gzip.compress(json.dumps({"samples": {}, "models": {}}).encode())
    server = _FakeServer(body)
    monkeypatch.setattr(_fetch, "_get", server)
    url = "https://example.invalid/metadata.json"

    assert _fetch.fetch_metadata(url, cache_dir=tmp_path) == {"samples": {}, "models": {}}
    assert server.calls == [{}]
    _fetch.fetch_metadata(url, cache_dir=tmp_path)  # fresh cache: no request
    assert len(server.calls) == 1
    _fetch.fetch_metadata(url, cache_dir=tmp_path, refresh=True)  # revalidate -> 304
    assert server.calls[-1] == {"If-None-Match": '"v1"'}
    server.fail = True
    with pytest.warns(UserWarning, match="using cached copy"):
        assert _fetch.fetch_metadata(url, cache_dir=tmp_path, refresh=True) == {"samples": {}, "models": {}}
    with pytest.raises(urllib.error.URLError):
        _fetch.fetch_metadata(url, cache_dir=tmp_path / "empty")


def test_cli(capsys):
    assert main(["--metadata", str(FIXTURE), "samples", "paris"]) == 0
    out = capsys.readouterr().out
    assert "PHercParis4" in out and "Scroll 1" in out

    assert main(["--metadata", str(FIXTURE), "volumes", "scroll 1", "--resolution", "2.4"]) == 0
    out = capsys.readouterr().out
    assert "78keV" in out and "137keV" in out

    args = ["--metadata", str(FIXTURE), "labels", "pherc paris 4", "--resolution", "2.399", "--energy", "78", "--um", "9.6"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "level 2 (9.6 um)" in out and out.count("direct") == 3

    assert main(["--metadata", str(FIXTURE), "labels", "pherc paris 4", "--resolution", "2.399"]) == 1
    assert "energy_kev" in capsys.readouterr().err
