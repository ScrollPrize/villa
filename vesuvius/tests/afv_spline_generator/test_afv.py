import json
import os
import sqlite3

import numpy as np
import pytest

from vc3d_fiber_format import parse_vc3d_fiber_format
from vesuvius.afv_spline_generator.afv import APPLICATION_ID, BLOCK_SEGMENTS, Fiber, write_afv


FRAME = {
    "vc_open_data_coordinate_space": "PHercTest/20250101000000",
    "vc_open_data_source_coordinate_level": 0,
    "vc_open_data_source_coordinate_scale_factor": 1,
}
ROOT = {"type": "vc3d_fiber_collection", "version": 1, **FRAME}


def fiber(name, count, family="V"):
    t = np.arange(count, dtype=np.float64)
    points = np.stack([0.5 * t + 0.1, np.sin(t / 10) + 100, 0.25 * t + 1000], axis=1)
    annotation = {
        "type": "vc3d_fiber",
        "version": 1,
        "hv_classification": {"manual_tag": family},
        "control_points": [points[0].tolist(), points[-1].tolist()],
    }
    return Fiber(name=name, family=family, points=points, annotation=annotation)


def stored_points(db, fiber_id):
    blocks = db.execute("SELECT id, first_segment, points FROM blocks WHERE fiber_id=? ORDER BY first_segment", (fiber_id,)).fetchall()
    points, expected_first = [], 0
    for block_id, first, blob in blocks:
        block = np.frombuffer(blob, dtype="<f8").reshape(-1, 3)
        assert first == expected_first and 2 <= len(block) <= BLOCK_SEGMENTS + 1
        if points:
            assert np.array_equal(points[-1][-1], block[0])
        bounds = db.execute("SELECT min_x, max_x, min_y, max_y, min_z, max_z FROM block_bounds WHERE id=?", (block_id,)).fetchone()
        lower, upper = np.array(bounds[0::2]), np.array(bounds[1::2])
        # RTree bounds are rounded outwards to single precision.
        assert np.all(lower <= block.min(axis=0)) and np.all(upper >= block.max(axis=0))
        points.append(block if not points else block[1:])
        expected_first += len(block) - 1
    return np.concatenate(points)


def test_round_trip(tmp_path):
    path = tmp_path / "fibers.afv"
    fibers = [fiber("long", 2 * BLOCK_SEGMENTS + 7), fiber("short", 2, "H")]
    info = write_afv(path, fibers, frame=FRAME, root=ROOT, metadata={"generator": {"name": "test"}})

    db = sqlite3.connect(path)
    assert db.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
    assert db.execute("PRAGMA user_version").fetchone()[0] == 1
    metadata = {key: json.loads(value) for key, value in db.execute("SELECT key, value FROM metadata")}
    assert metadata == {
        "complete": True,
        "uuid": info["uuid"],
        "frame": FRAME,
        "root": ROOT,
        "fiber_count": 2,
        "point_count": 2 * BLOCK_SEGMENTS + 9,
        "generator": {"name": "test"},
    }
    assert db.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='fibers_length'").fetchone()
    rows = db.execute("SELECT id, name, family, point_count, length, min_x, max_x, min_y, max_y, min_z, max_z, annotation FROM fibers ORDER BY id")
    for row, source in zip(rows, fibers):
        fiber_id, name, family, count, length, *bounds, annotation = row
        points = stored_points(db, fiber_id)
        assert (name, family, count) == (source.name, source.family, len(source.points))
        assert np.array_equal(points, source.points)
        assert length == pytest.approx(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        assert bounds == [v for axis in range(3) for v in (points[:, axis].min(), points[:, axis].max())]
        parsed = parse_vc3d_fiber_format({**json.loads(annotation), "line_points": points.tolist()})
        assert parsed.metadata["hv_classification"] == {"manual_tag": family}
    assert db.execute("SELECT count(*) FROM blocks WHERE fiber_id=1").fetchone()[0] == 3
    db.close()
    assert [p.name for p in tmp_path.iterdir()] == ["fibers.afv"]


def test_never_replaces_an_existing_file(tmp_path):
    path = tmp_path / "fibers.afv"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        write_afv(path, [fiber("a", 3)], frame=FRAME, root=ROOT)
    assert path.read_bytes() == b"keep"
    assert [p.name for p in tmp_path.iterdir()] == ["fibers.afv"]


@pytest.mark.skipif(os.name == "nt", reason="Windows renames without replacing")
def test_refuses_to_publish_without_hard_links(tmp_path, monkeypatch):
    def no_hard_links(source, destination):
        raise PermissionError("no hard links")

    monkeypatch.setattr(os, "link", no_hard_links)
    with pytest.raises(OSError, match="hard links"):
        write_afv(tmp_path / "fibers.afv", [fiber("a", 3)], frame=FRAME, root=ROOT)
    assert list(tmp_path.iterdir()) == []


def invalid(change):
    base = fiber("bad", 4)
    return Fiber(**{"name": base.name, "family": base.family, "points": base.points, "annotation": base.annotation, **change})


@pytest.mark.parametrize(
    "bad",
    [
        invalid({"points": np.zeros((1, 3))}),
        invalid({"points": np.array([[0, 0, 0], [np.nan, 1, 1]])}),
        invalid({"points": np.zeros((3, 2))}),
        invalid({"family": "X"}),
        invalid({"annotation": {"type": "vc3d_fiber", "line_points": []}}),
    ],
)
def test_invalid_fibers_leave_no_file(tmp_path, bad):
    with pytest.raises(ValueError):
        write_afv(tmp_path / "fibers.afv", [fiber("good", 3), bad], frame=FRAME, root=ROOT)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"frame": {}},
        {"frame": {**FRAME, "vc_open_data_source_coordinate_level": 1}},
        {"metadata": {"uuid": "chosen"}},
        {"path": "fibers.json"},
    ],
)
def test_rejects_invalid_collections(tmp_path, arguments):
    arguments = {"frame": FRAME, "root": ROOT, **arguments}
    path = tmp_path / arguments.pop("path", "fibers.afv")
    with pytest.raises(ValueError):
        write_afv(path, [fiber("a", 3)], **arguments)
    assert list(tmp_path.iterdir()) == []
