"""Tests for vesuvius.data.umbilicus. The download is stubbed; one network test reads the real dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vesuvius.data import umbilicus


def _line(points):
    return {"metadata": {"voxelsize_um": 9.362}, "control_points": [{"x": x, "y": y, "z": z} for x, y, z in points]}


def _row(scroll, scan, source, kind, content, folder="lines", **extra):
    row = {
        "scroll": scroll,
        "scan": scan,
        "file": f"{folder}/{scroll}/{scroll}_{scan}_{source}_umbilicus.json",
        "source": source,
        "kind": kind,
        "credit": f"credit for {source}",
        "licence": "CC BY 4.0",
        "sha256": hashlib.sha256(content).hexdigest(),
        "volume": None,
    }
    row.update(extra)
    return row


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A tiny copy of the dataset layout, served through the stubbed download."""
    files = {}

    def add(row, content):
        files[row["file"]] = content
        return row

    a = json.dumps(_line([(10, 20, 0), (11, 21, 100)])).encode()
    b = json.dumps(_line([(30, 40, 0), (31, 41, 100)])).encode()
    c = json.dumps(_line([(50, 60, 0)])).encode()
    old = json.dumps(_line([(1, 2, 0)])).encode()
    new = json.dumps(_line([(3, 4, 0)])).encode()
    other = json.dumps(_line([(5, 6, 0)])).encode()
    index = {
        "scrolls": {
            "ScrollA": {"scan": "100"},
            "ScrollB": {"scan": "200"},
            "ScrollC": {"scan": "300"},
        },
        "files": [
            add(_row("ScrollA", "100", "bucket", "person", a), a),
            add(_row("ScrollA", "100", "finder", "model", c), c),
            add(_row("ScrollB", "200", "drobkov", "person", a), a),
            add(_row("ScrollB", "200", "tauil", "person", b), b),
            add(_row("ScrollC", "300", "finder", "model", old,
                     superseded_by="lines/ScrollC/ScrollC_300_finder-polished_umbilicus.json"), old),
            add(_row("ScrollC", "300", "finder-polished", "model", new), new),
        ],
        "other_scans": [
            add(_row("ScrollA", "101", "bucket-mapped", "model", other, folder="other-scans"), other),
        ],
    }
    files["index.json"] = json.dumps(index).encode()
    calls = []

    def fake_download(filename, revision, cache_dir):
        calls.append((filename, revision))
        path = tmp_path / "hub" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[filename])
        return path

    monkeypatch.setattr(umbilicus, "_download", fake_download)
    return {"tmp": tmp_path, "files": files, "calls": calls}


def test_default_takes_person_line_over_model(hub):
    path, row = umbilicus.fetch_umbilicus("ScrollA")
    assert row["source"] == "bucket"
    assert json.loads(path.read_text())["control_points"][0] == {"x": 10, "y": 20, "z": 0}
    assert all(revision == umbilicus.DEFAULT_REVISION for _, revision in hub["calls"])


def test_superseded_line_is_skipped_unless_named(hub):
    assert umbilicus.fetch_umbilicus("ScrollC")[1]["source"] == "finder-polished"
    assert umbilicus.fetch_umbilicus("ScrollC", source="finder")[1]["source"] == "finder"


def test_source_picks_a_model_line(hub):
    assert umbilicus.fetch_umbilicus("ScrollA", source="finder")[1]["kind"] == "model"


def test_other_scan_is_found_in_other_scans(hub):
    path, row = umbilicus.fetch_umbilicus("ScrollA", scan="101")
    assert row["source"] == "bucket-mapped"
    assert path.name == "ScrollA_101_bucket-mapped_umbilicus.json"


def test_two_person_lines_are_ambiguous(hub):
    with pytest.raises(ValueError, match="drobkov, tauil"):
        umbilicus.fetch_umbilicus("ScrollB")
    assert umbilicus.fetch_umbilicus("ScrollB", source="tauil")[1]["source"] == "tauil"


def test_unknown_scroll_scan_and_source(hub):
    with pytest.raises(ValueError, match="known scrolls: ScrollA, ScrollB, ScrollC"):
        umbilicus.fetch_umbilicus("ScrollZ")
    with pytest.raises(ValueError, match="scans with one: 100, 101"):
        umbilicus.fetch_umbilicus("ScrollA", scan="999")
    with pytest.raises(ValueError, match="sources: bucket, finder"):
        umbilicus.fetch_umbilicus("ScrollA", source="nobody")


def test_sha256_mismatch_is_refused(hub):
    hub["files"]["lines/ScrollA/ScrollA_100_bucket_umbilicus.json"] = b'{"control_points": []}'
    with pytest.raises(ValueError, match="does not match the index"):
        umbilicus.fetch_umbilicus("ScrollA")


def test_into_writes_umbilicus_json_and_refuses_to_overwrite(hub):
    volpkg = hub["tmp"] / "scroll.volpkg"
    path, _ = umbilicus.fetch_umbilicus("ScrollA", into=volpkg)
    assert path == volpkg / "umbilicus.json"
    assert path.read_bytes() == hub["files"]["lines/ScrollA/ScrollA_100_bucket_umbilicus.json"]
    assert [p.name for p in volpkg.iterdir()] == ["umbilicus.json"]
    with pytest.raises(FileExistsError, match="exists"):
        umbilicus.fetch_umbilicus("ScrollA", into=volpkg)


def test_into_refuses_folder_with_estimated_umbilicus(hub):
    volpkg = hub["tmp"] / "other.volpkg"
    volpkg.mkdir()
    (volpkg / "estimated_umbilicus.json").write_text("{}")
    with pytest.raises(FileExistsError, match="estimated_umbilicus.json"):
        umbilicus.fetch_umbilicus("ScrollA", into=volpkg)
    assert not (volpkg / "umbilicus.json").exists()


def test_list_lines(hub):
    rows = umbilicus.list_umbilicus_lines("ScrollA")
    assert [row["source"] for row in rows] == ["bucket", "finder", "bucket-mapped"]
    assert len(umbilicus.list_umbilicus_lines()) == 7


def test_missing_huggingface_hub_gives_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_hub(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("no huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_hub)
    with pytest.raises(ImportError, match="pip install huggingface_hub"):
        umbilicus.fetch_umbilicus("PHerc0139")


@pytest.mark.network
def test_published_paris4_line(tmp_path):
    path, row = umbilicus.fetch_umbilicus("PHercParis4", cache_dir=tmp_path)
    assert row["source"] == "bucket" and row["kind"] == "person"
    points = json.loads(Path(path).read_text())["control_points"]
    assert len(points) == row["points"]
    assert [min(p["z"] for p in points), max(p["z"] for p in points)] == row["z"]
