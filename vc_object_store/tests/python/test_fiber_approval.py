"""The fiber approval layer helpers share the native painting, staleness and row semantics."""
import uuid

import pytest

from vc_object_store import (
    FIBER_APPROVAL_NORMAL_POINT_TABLE, FIBER_APPROVAL_STRETCH_TABLE, FIBER_APPROVAL_TABLE,
    approved_fiber_line_normals, fiber_approval_rows, fiber_approval_status, fiber_approval_tables,
    fiber_line_fingerprint, fiber_stretch_state_at, set_fiber_normal_point, set_fiber_stretch_state,
)


def line(points=101):
    return [[float(x), 0.0, 0.0] for x in range(points)]


def test_painting_states_cuts_clears_and_joins():
    points, stretches = line(), []
    stretches = set_fiber_stretch_state(stretches, "geometry", 10, 40, "approved", points)
    stretches = set_fiber_stretch_state(stretches, "normals", 20, 30, "approved", points)
    assert len(stretches) == 2
    assert stretches[0]["fingerprint"] == fiber_line_fingerprint(points, 10, 40)
    stretches = set_fiber_stretch_state(stretches, "geometry", 20.5, 25.25, "broken", points)
    assert len(stretches) == 4
    assert fiber_stretch_state_at(stretches, "geometry", 15) == "approved"
    assert fiber_stretch_state_at(stretches, "geometry", 22) == "broken"
    assert fiber_stretch_state_at(stretches, "geometry", 30) == "approved"
    assert fiber_stretch_state_at(stretches, "normals", 22) == "approved"
    assert fiber_stretch_state_at(stretches, "geometry", 50) == "undefined"
    stretches = set_fiber_stretch_state(stretches, "geometry", 20.5, 25.25, "approved", points)
    assert [(s["aspect"], s["from"], s["to"]) for s in stretches] == [("geometry", 10, 40), ("normals", 20, 30)]
    stretches = set_fiber_stretch_state(stretches, "geometry", 0, 100, "undefined", points)
    assert [s["aspect"] for s in stretches] == ["normals"]
    with pytest.raises(ValueError):
        set_fiber_stretch_state(stretches, "geometry", 30, 30, "approved", points)
    with pytest.raises(ValueError):
        set_fiber_stretch_state(stretches, "colour", 10, 30, "approved", points)


def test_staleness_follows_the_changed_line_points():
    points = line()
    stretches = set_fiber_stretch_state([], "normals", 10, 20, "approved", points)
    stretches = set_fiber_stretch_state(stretches, "normals", 60, 70, "approved", points)
    stretches = set_fiber_stretch_state(stretches, "geometry", 60, 70, "broken", points)
    normal_points, index = set_fiber_normal_point([], 40.2, [0, 0, 2], points)
    assert index == 0
    normal_points, index = set_fiber_normal_point(normal_points, 40.4, [0, 1, 0], points)
    assert index == 0 and len(normal_points) == 1
    normal_points, index = set_fiber_normal_point(normal_points, 5, [0, 0, 3], points)
    assert index == 0 and len(normal_points) == 2 and normal_points[0]["normal"] == [0, 0, 1]
    layer = dict(fiber_uuid=str(uuid.uuid4()), stretches=stretches, normal_points=normal_points,
                 line_normals=[[0, 0, 1]] * len(points))
    status = fiber_approval_status(layer, points)
    assert status["stale_stretches"] == 0 and status["stale_normal_points"] == 0 and status["line_normals_usable"]
    normals = approved_fiber_line_normals(layer, points)
    assert len(normals) == len(points)
    assert normals[15][2] == 1 and normals[30][2] == 0 and normals[40][1] == 1 and normals[5][2] == 1 and normals[65][2] == 1
    points[65][1] = 3.0  # the fiber was edited inside the second stretch
    status = fiber_approval_status(layer, points)
    assert status["stale_stretches"] == 2 and status["stale_normal_points"] == 0
    assert status["stretch_current"] == [False, True, False]  # geometry 60..70, normals 10..20, normals 60..70
    normals = approved_fiber_line_normals(layer, points)
    assert normals[15][2] == 1 and normals[65][2] == 0
    assert not fiber_approval_status(layer, points[:-1])["line_normals_usable"]
    assert approved_fiber_line_normals(layer, points[:-1]) == []


def test_the_newest_generation_in_a_store_is_the_layer(db):
    fiber, other = str(uuid.uuid4()), str(uuid.uuid4())
    points = line()
    assert db.read_fiber_approval(fiber) is None  # no tables yet
    stretches = set_fiber_stretch_state([], "geometry", 10, 40, "approved", points)
    stretches = set_fiber_stretch_state(stretches, "normals", 12.5, 30, "broken", points)
    normal_points, _ = set_fiber_normal_point([], 20, [0, 0.5, 0.5], points)
    layer = dict(fiber_uuid=fiber, generation=1, fiber_version=3, author="alice", saved_at="2026-11-25T10:00:00Z",
                 host="host-a", baseline_run="run-1", stretches=stretches, normal_points=normal_points,
                 line_normals=[[0.0, 0.5, -0.25]] * len(points))  # float32-exact, as stored

    def write(value):
        rows = fiber_approval_rows(value)
        with db.write() as tx:
            for table in fiber_approval_tables():
                tx.create_extension_table(table)
            tx.append_extension_rows(FIBER_APPROVAL_TABLE, [rows["header"]])
            if rows["stretches"]:
                tx.append_extension_rows(FIBER_APPROVAL_STRETCH_TABLE, rows["stretches"])
            if rows["normal_points"]:
                tx.append_extension_rows(FIBER_APPROVAL_NORMAL_POINT_TABLE, rows["normal_points"])

    assert [t["name"] for t in fiber_approval_tables()] == [
        FIBER_APPROVAL_TABLE, FIBER_APPROVAL_STRETCH_TABLE, FIBER_APPROVAL_NORMAL_POINT_TABLE]
    rows = fiber_approval_rows(layer)
    assert len(rows["stretches"]) == 2 and len(rows["normal_points"]) == 1
    assert len(rows["header"][-1]) == 12 * len(points)
    write(layer)
    assert db.read_fiber_approval(fiber) == layer
    assert db.read_fiber_approval(other) is None
    second = dict(layer, generation=2, stretches=[], normal_points=[])
    write(second)
    write(dict(layer, fiber_uuid=other))
    assert db.read_fiber_approval(fiber) == second
    assert db.read_fiber_approval(other)["generation"] == 1
