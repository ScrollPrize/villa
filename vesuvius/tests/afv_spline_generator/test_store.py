import os

import numpy as np
import pytest
from vc_object_store import ObjectStore, RegionSlice, RegionOptions
from vc3d_fiber_format import parse_vc3d_fiber_format
from vesuvius.afv_spline_generator.store import Fiber, write_store

COORDINATES = {"volume_uuid": "20250101000000", "scale_factor": 1}


def fiber(name, count, family="V"):
    t = np.arange(count, dtype=np.float64)
    points = np.stack([0.5 * t + 0.1, np.sin(t / 10) + 100, 0.25 * t + 1000], axis=1)
    return Fiber(name=name, family=family, points=points, annotation={
        "hv_classification": {"manual_tag": family},
        "control_points": [points[0].tolist(), points[-1].tolist()],
        "inferred_gap_point_ranges": [[1, 2]],
    })


def test_round_trip(tmp_path):
    path = tmp_path / "fibers.vcobj"
    fibers = [fiber("long", 519), fiber("short", 2, "H")]
    info = write_store(path, fibers, coordinates=COORDINATES, metadata={"generator": {"name": "test"}})
    assert [p.name for p in tmp_path.iterdir()] == ["fibers.vcobj"]
    with ObjectStore.open(str(path), read_only=True) as db:
        assert info == {"uuid": db.uuid, "fiber_count": 2, "point_count": 521}
        records = {r["name"]: r for r in db.objects_by_kind("fiber")}
        with db.geometry_snapshot() as snapshot:
            for source in fibers:
                row = records[source.name]
                assert row["meta"]["coordinates"] == COORDINATES
                assert row["meta"]["generator"] == {"name": "test"}
                assert row["meta"]["source"]["inferred_gap_point_ranges"] == [[1, 2]]
                assert "control_points" not in row["meta"]["source"]
                assert row["fiber"]["control_point_count"] == 2
                assert row["fiber"]["line_point_count"] == len(source.points)
                assert row["fiber"]["family"] == source.family
                assert row["fiber"]["length"] == pytest.approx(np.linalg.norm(np.diff(source.points, axis=0), axis=1).sum())
                view = snapshot.fiber(row["uuid"])
                assert np.array_equal(view.line_points_array(), source.points)
                assert np.array_equal(view.control_points_array(), source.points[[0, -1]])
                # Check the editor's fiber-format contract as well as the
                # store round trip: segment modes are not fiber modes.
                parsed = parse_vc3d_fiber_format({
                    **row["meta"]["source"],
                    "optimization_mode": row["fiber"]["optimization_mode"],
                    "control_points": [
                        {"position": point.tolist()}
                        for point in view.control_points_array()
                    ],
                    "line_points": view.line_points_array().tolist(),
                })
                assert parsed.optimization_mode == "lasagna"
        # VC3D uses these volume-targeted slab queries at the displayed level.
        slab = RegionSlice()
        slab.local_bounds = [0, 45, 495, 150, 55, 570]
        options = RegionOptions()
        options.volume = COORDINATES["volume_uuid"]
        options.target_scale = 0.5
        options.min_length = 10
        options.family = "V"
        region = db.geometry_region(slab, options)
        assert region["complete"] and region["runs"]
        assert {r["object_uuid"] for r in region["runs"]} == {records["long"]["uuid"]}
        assert np.array_equal(region["runs"][0]["points"][0], fibers[0].points[0] / 2)


def test_never_replaces_an_existing_file(tmp_path):
    path = tmp_path / "fibers.vcobj"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        write_store(path, [fiber("a", 3)], coordinates=COORDINATES)
    assert path.read_bytes() == b"keep"
    assert [p.name for p in tmp_path.iterdir()] == ["fibers.vcobj"]


@pytest.mark.skipif(os.name == "nt", reason="Windows renames without replacing")
def test_refuses_to_publish_without_hard_links(tmp_path, monkeypatch):
    def no_hard_links(source, destination):
        raise PermissionError("no hard links")

    monkeypatch.setattr(os, "link", no_hard_links)
    with pytest.raises(OSError, match="hard links"):
        write_store(tmp_path / "fibers.vcobj", [fiber("a", 3)], coordinates=COORDINATES)
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
        write_store(tmp_path / "fibers.vcobj", [fiber("good", 3), bad], coordinates=COORDINATES)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"coordinates": {}},
        {"coordinates": {**COORDINATES, "scale_factor": 2}},
        {"metadata": {"coordinates": {}}},
        {"path": "fibers.json"},
    ],
)
def test_rejects_invalid_collections(tmp_path, arguments):
    arguments = {"coordinates": COORDINATES, **arguments}
    path = tmp_path / arguments.pop("path", "fibers.vcobj")
    with pytest.raises(ValueError):
        write_store(path, [fiber("a", 3)], **arguments)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not os.environ.get("VC_FIBER_TEST_SOURCE"), reason="Set VC_FIBER_TEST_SOURCE to real fiber JSONs")
def test_real_fiber_output(tmp_path):
    import json
    from pathlib import Path
    from vesuvius.afv_spline_generator.cli import afv_fibers

    paths = sorted(Path(os.environ["VC_FIBER_TEST_SOURCE"]).glob("*.json"))
    assert paths, "The real-data fixture contains no fiber JSONs"
    traces = []
    for path in paths:
        source = json.loads(path.read_text())
        points = np.asarray([p["position"] if isinstance(p, dict) else p for p in source["line_points"]], dtype=np.float64)
        if len(points) < 2:
            continue
        hv = source.get("hv_classification", {})
        family = hv.get("manual_tag") or hv.get("automatic_tag", "V")
        traces.append(dict(points=points / 2, family=int(family == "H")))
    assert traces
    fibers = list(afv_fibers(traces, native_scale=2, voxel_size=None))
    output = tmp_path / "real-fibers.vcobj"
    info = write_store(output, fibers, coordinates=COORDINATES)
    with ObjectStore.open(str(output), read_only=True) as store:
        rows = {r["name"]: r for r in store.objects_by_kind("fiber")}
        assert len(rows) == info["fiber_count"] == len(traces)
        with store.geometry_snapshot() as snapshot:
            for expected in fibers:
                record = rows[expected.name]
                view = snapshot.fiber(record["uuid"])
                assert np.array_equal(view.line_points_array(), expected.points)
                assert np.array_equal(view.control_points_array(), expected.points[[0, -1]])
                assert record["fiber"]["family"] == expected.family
    print(f"Real-data round trip: {len(traces)} fibers, {info['point_count']} points")
