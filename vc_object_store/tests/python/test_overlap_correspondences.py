"""Point pairs materialized from stored surface-overlap poses."""
import json

import numpy as np
import pytest

from object_helpers import VOLUME, array, edge_choice, frame, put

LOW = "00000000-0000-4000-8000-000000000001"
HIGH = "ffffffff-0000-4000-8000-000000000001"


def patch(db, uid, to_xyz, size=40, scale=1, hole=None):
    """A patch whose vertex (row, col) lies at volume XYZ to_xyz(row, col), stored at scale."""
    rows, cols = np.mgrid[:size, :size]
    xyz = np.stack(to_xyz(rows, cols), axis=-1).astype(np.float64) * scale
    if hole is not None:
        xyz[hole] = -1
    return put(db, "surface", [array("points", xyz, "<f4")], {"scale": [1 / scale, 1 / scale]},
               meta={"coordinates": frame(scale)}, **({"uuid": uid} if uid else {}))


def flat(dx=0):
    return lambda rows, cols: (cols + dx, rows, np.full_like(rows, 20))


def overlap_edge(db):
    report = db.link_surface_overlaps(volume_uuid=VOLUME, min_inliers=16)
    assert report["edges_created"] == 1 and not report["problems"], report
    return db.edges_by_kind("same-surface")[0]


def materialize(db, edge, **kwargs):
    [result] = db.overlap_correspondences([edge["uuid"]], VOLUME, **kwargs)
    return result


def pinned(db, edge):
    endpoints = {side: db.current_object_revision(edge[side]) for side in "ab"}
    return {"uuid": edge["uuid"], "whole_geometry": True, "meta": json.dumps(edge["meta"]),
            **{f"{side}_reference": {"object": revision["object_uuid"],
                                    **{key: revision[key] for key in ("store", "generation", "version")}}
               for side, revision in endpoints.items()}}


@pytest.mark.parametrize("history_enabled", [False, True])
def test_incremental_overlaps_and_current_pairs_with_optional_history(tmp_path, history_enabled):
    from vc_object_store import ObjectStore

    with ObjectStore.create(str(tmp_path / "overlaps.vcobj"), author="test",
                            history_enabled=history_enabled) as store:
        patch(store, LOW, flat())
        patch(store, HIGH, flat(10))
        report = store.link_surface_overlaps(volume_uuid=VOLUME, min_inliers=16,
                                             new_uuids=[HIGH])
        assert report["edges_created"] == 1 and not report["problems"], report
        edge = store.edges_by_kind("same-surface")[0]
        assert "generation" not in edge and edge["version"] >= 1
        result = materialize(store, edge)
        assert result["error"] is None and result["pairs"]
        repeat = store.link_surface_overlaps(volume_uuid=VOLUME, min_inliers=16,
                                             new_uuids=[HIGH])
        assert repeat["edges_skipped_existing"] == 1 and not repeat["problems"]


def test_incremental_overlaps_do_not_implicitly_build_omitted_spatial_index(tmp_path):
    from vc_object_store import ObjectStore

    with ObjectStore.create(str(tmp_path / "slim.vcobj"), author="test",
                            history_enabled=False, spatial_index=False) as store:
        patch(store, LOW, flat())
        patch(store, HIGH, flat(10))
        with pytest.raises(RuntimeError, match="spatial index is absent"):
            store.link_surface_overlaps(volume_uuid=VOLUME, new_uuids=[HIGH])
        assert not store.provisioning()["spatial_index"]
        assert not store.edges_by_kind("same-surface")
        # Full detection does not need a persisted spatial index.
        edge = overlap_edge(store)
        assert materialize(store, edge)["pairs"]


def assert_consistent(result, a_to_xyz, b_to_xyz):
    """Every pair names nearly the same volume point on both patches, at its grid position.

    The stored pose is the detector's fit, a fraction of a voxel from the exact
    transform; a reversed direction would put pairs about 20 voxels apart.
    """
    assert result["error"] is None and result["pairs"]
    for pair in result["pairs"]:
        np.testing.assert_allclose(pair["a_xyz"], pair["b_xyz"], atol=0.5)
        assert pair["distance"] < 0.5
        np.testing.assert_allclose(pair["a_xyz"], np.array(a_to_xyz(*pair["a_grid"]), float), atol=1e-3)
        np.testing.assert_allclose(pair["b_xyz"], np.array(b_to_xyz(*pair["b_grid"]), float), atol=1e-3)


@pytest.mark.parametrize("first,second", [(LOW, HIGH), (HIGH, LOW)])
def test_pairs_follow_stored_endpoint_order_whatever_the_pose_direction(db, first, second):
    # Endpoint identity/order uses UUID keys, independent of insertion order.
    patch(db, first, flat())
    patch(db, second, flat(10))
    edge = overlap_edge(db)
    assert (edge["a"], edge["b"]) == (LOW, HIGH)
    fit = edge["meta"]["fit"]
    assert (fit["pose_from"], fit["pose_to"]) == (HIGH, LOW)
    assert fit["grid_scales"] == {LOW: [1, 1], HIGH: [1, 1]}
    shifts = {first: 0, second: 10}
    result = materialize(db, edge)
    assert (result["a"], result["b"]) == (edge["a"], edge["b"])
    assert_consistent(result, flat(shifts[edge["a"]]), flat(shifts[edge["b"]]))
    for pair in result["pairs"]:
        assert pair["b_grid"][1] == pytest.approx(pair["a_grid"][1] + shifts[edge["a"]] - shifts[edge["b"]], abs=0.5)


def test_reflected_pose_and_mixed_scales(db):
    patch(db, LOW, flat(), scale=2)
    transposed = lambda rows, cols: (rows + 5, cols + 3, np.full_like(rows, 20))
    patch(db, HIGH, transposed, scale=0.5)
    edge = overlap_edge(db)
    assert edge["meta"]["fit"]["reflected"]
    result = materialize(db, edge)
    assert_consistent(result, flat() if edge["a"] == LOW else transposed,
                      transposed if edge["a"] == LOW else flat())


def test_invalid_vertices_never_anchor_pairs(db):
    patch(db, LOW, flat())
    hole = (slice(10, 20), slice(20, 30))
    patch(db, HIGH, flat(10), hole=hole)
    edge = overlap_edge(db)
    result = materialize(db, edge, spacing=1)
    assert_consistent(result, flat(), flat(10))
    holed = 0 if edge["a"] == HIGH else 1
    for pair in result["pairs"]:
        row, col = (pair["a_grid"], pair["b_grid"])[holed]
        # Every corner of the quad containing the point is valid.
        assert not (9 <= row < 20 and 19 <= col < 30)


def overlapping_pair(db):
    """LOW and HIGH ten voxels apart, linked by the detector; HIGH's geometry."""
    patch(db, LOW, flat())
    _, geometry = patch(db, HIGH, flat(10))
    return overlap_edge(db), geometry


def whole_edge(db, kind, meta=None, a=LOW, b=HIGH):
    whole = dict(whole_geometry=True)
    return db.add_edge(dict(a=a, b=b, kind=kind, a_anchor=whole, b_anchor=whole,
                            **({} if meta is None else {"meta": meta})))


def move_high_keeping_edges(db, geometry):
    """A new HIGH geometry revision; its edges stay bound to the recorded one."""
    geometry["arrays"] = patch(db, None, flat(11))[1]["arrays"]
    db.put_geometry(HIGH, geometry, expected_version=db.current_object_revision(HIGH)["version"],
                    edge_edits=[edge_choice(db, e["uuid"], "keep_recorded")
                                for e in db.affected_geometry_edges(HIGH, geometry)])


def test_deterministic_thinning(db):
    edge, _ = overlapping_pair(db)
    dense = materialize(db, edge, spacing=1, max_pairs=100000)
    thinned = materialize(db, edge, spacing=1, max_pairs=50)
    assert thinned == materialize(db, edge, spacing=1, max_pairs=50)
    assert len(thinned["pairs"]) == 50 and len(dense["pairs"]) > 500
    n = len(dense["pairs"])
    assert thinned["pairs"] == [dense["pairs"][i * n // 50] for i in range(50)]


def test_unusable_edges_report_errors_independently(db):
    edge, geometry = overlapping_pair(db)
    # Edges from before the pose's frame and direction were recorded must be re-detected.
    old_fit = {k: v for k, v in edge["meta"]["fit"].items() if k != "pose_from"}
    results = db.overlap_correspondences([
        whole_edge(db, "explicit", {"fit": edge["meta"]["fit"], "coordinate_frame": None}),
        whole_edge(db, "old", dict(edge["meta"], fit=old_fit)),
        edge["uuid"], whole_edge(db, "plain")], VOLUME)
    assert [r["error"] is None for r in results] == [False, False, True, False]
    assert all("rerun surface-overlap detection" in r["error"] for r in results[:2])
    assert results[3]["error"] == "edge has no overlap pose"
    assert all(r["pairs"] == [] for r in results if r["error"])
    # A changed frame changes the grid scale the pose was fitted in.
    db.set_meta(HIGH, "coordinates", frame(2))
    assert "grid scale differs" in materialize(db, edge)["error"]
    db.set_meta(HIGH, "coordinates", frame(volume="other"))
    assert "does not match target volume" in materialize(db, edge)["error"]
    db.set_meta(HIGH, "coordinates", frame(1))
    assert materialize(db, edge)["error"] is None
    # An edge kept bound to its recorded revision across a coordinate edit is stale.
    move_high_keeping_edges(db, geometry)
    assert db.edge_geometry_state(edge["uuid"])["stale"]
    assert materialize(db, edge)["error"] == "edge geometry is stale"


@pytest.mark.parametrize("radius,accepted", [(6.0, True), (np.nextafter(6.0, 0.0), False)])
def test_correspondence_agreement_boundary(db, radius, accepted):
    patch(db, LOW, flat())
    patch(db, HIGH, lambda rows, cols: (cols, rows, np.full_like(rows, 26)))
    report = db.link_surface_overlaps(volume_uuid=VOLUME, min_inliers=16, tolerance=7)
    assert report["edges_created"] == 1
    edge = db.edges_by_kind("same-surface")[0]
    assert edge["meta"]["fit"]["agreement_radius"] == 10  # the detector's radius is recorded
    # Exact pose isolates the distance gate from fitting error: every pair is
    # six voxels apart. Equality is accepted, a smaller radius rejects it.
    fit = dict(edge["meta"]["fit"], pose=[1, 0, 0, 0, 1, 0], agreement_radius=float(radius))
    [result] = db.overlap_correspondences(
        [whole_edge(db, "boundary", dict(edge["meta"], fit=fit), edge["a"], edge["b"])], VOLUME)
    assert bool(result["pairs"]) == accepted
    if accepted:
        assert result["error"] is None and all(pair["distance"] == 6 for pair in result["pairs"])
    else:
        assert result["error"] == "no agreeing correspondences under the stored pose"


def test_pairs_from_an_earlier_state_ignore_later_changes(db):
    edge, geometry = overlapping_pair(db)
    # The edge and its endpoints as a reader holding them now would pass them.
    then = pinned(db, edge)
    before = materialize(db, edge)
    assert db.overlap_correspondences_at([then], VOLUME) == [before]
    # Later HIGH moves and the edge is removed: the earlier state still reads
    # the same pairs, from its recorded pose and geometry.
    move_high_keeping_edges(db, geometry)
    db.remove_edge(edge["uuid"])
    assert db.overlap_correspondences([edge["uuid"]], VOLUME)[0]["error"] == "edge not found"
    assert db.overlap_correspondences_at([then], VOLUME) == [before]


def test_pinned_pairs_read_through_a_session_attachment(db, db_path, tmp_path):
    """overlap_correspondences_at over an ObjectStoreSession attachment reads
    the pinned revisions through its reader: a local attachment and, when a
    vc_object_server is at hand, a hosted one give the store's own pairs."""
    import os
    import re
    import subprocess
    import vc_object_store as vos
    from vc_object_store.locations import open_store

    edge, _ = overlapping_pair(db)
    then = pinned(db, edge)
    expected = db.overlap_correspondences_at([then], VOLUME)
    assert expected[0]["error"] is None and expected[0]["pairs"]
    db.close()

    session = vos.ObjectStoreSession()
    local = session.attach(db_path, "pytest", read_only=True)
    try:
        assert session.overlap_correspondences_at(local, [then], VOLUME) == expected
    finally:
        session.detach(local)

    server_binary = os.environ.get("VC_OBJECT_SERVER")
    if not server_binary:
        pytest.skip("set VC_OBJECT_SERVER to a vc_object_server binary for the hosted half")
    server = subprocess.Popen([server_binary, db_path, "--port", "0"], stderr=subprocess.PIPE, text=True)
    try:
        port = re.search(r":(\d+)\s*$", server.stderr.readline())
        assert port
        endpoint = f"http://127.0.0.1:{port.group(1)}"
        with open_store(endpoint, author="pytest", cache_directory=tmp_path / "cache") as hosted:
            assert hosted.overlap_correspondences_at([then], VOLUME) == expected
            # The same edge again is answered from the attachment's cache.
            assert hosted.overlap_correspondences_at([then], VOLUME) == expected
    finally:
        server.terminate()
        assert server.wait(timeout=30) == 0


def test_pinned_cache_distinguishes_metadata_only_coordinate_edits(db):
    edge, _ = overlapping_pair(db)
    before = pinned(db, edge)
    db.set_meta(HIGH, "coordinates", frame(2))
    after = pinned(db, edge)
    # Geometry bytes/revision are unchanged, but the selected coordinate frame
    # differs. Batching must not reuse the first endpoint's transform.
    assert before["b_reference"] != after["b_reference"]
    old = db.overlap_correspondences_at([before], VOLUME)[0]
    new = db.overlap_correspondences_at([after], VOLUME)[0]
    assert old["error"] is None and "grid scale differs" in new["error"]
    assert db.overlap_correspondences_at([before, after], VOLUME) == [old, new]
    assert db.overlap_correspondences_at([after, before], VOLUME) == [new, old]
