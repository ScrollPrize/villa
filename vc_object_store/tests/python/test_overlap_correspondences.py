"""Point pairs materialized from stored patch-overlap poses."""
import numpy as np
import pytest

from object_helpers import edge_choice
from test_coordinate_frames import VOLUME, frame
from test_object_links import array, put

LOW = "00000000-0000-4000-8000-000000000001"
HIGH = "ffffffff-0000-4000-8000-000000000001"


def patch(db, uid, to_xyz, size=40, scale=1, hole=None):
    """A patch whose vertex (row, col) lies at volume XYZ to_xyz(row, col), stored at scale."""
    rows, cols = np.mgrid[:size, :size]
    xyz = np.stack(to_xyz(rows, cols), axis=-1).astype(np.float64) * scale
    if hole is not None:
        xyz[hole] = -1
    return put(db, "patch", [array("points", xyz, "<f4")], {"scale": [1 / scale, 1 / scale]},
               meta={"coordinates": frame(scale)}, **({"uuid": uid} if uid else {}))


def flat(dx=0):
    return lambda rows, cols: (cols + dx, rows, np.full_like(rows, 20))


def overlap_edge(db):
    report = db.link_patch_overlaps(volume_uuid=VOLUME, min_inliers=16)
    assert report["edges_created"] == 1 and not report["failed"], report
    return db.edges_by_kind("same-surface")[0]


def materialize(db, edge, **kwargs):
    [result] = db.overlap_correspondences([edge["uuid"]], VOLUME, **kwargs)
    return result


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
    # Insertion order decides stored endpoint order; UUID order decided the detector's seed.
    patch(db, first, flat())
    patch(db, second, flat(10))
    edge = overlap_edge(db)
    assert edge["a"] == first
    fit = edge["meta"]["fit"]
    assert (fit["pose_from"], fit["pose_to"]) == (HIGH, LOW)
    assert fit["grid_scales"] == {LOW: [1, 1], HIGH: [1, 1]}
    shifts = {first: 0, second: 10}
    result = materialize(db, edge)
    assert (result["a"], result["b"]) == (first, second)
    assert_consistent(result, flat(shifts[first]), flat(shifts[second]))
    for pair in result["pairs"]:
        assert pair["b_grid"][1] == pytest.approx(pair["a_grid"][1] + shifts[first] - shifts[second], abs=0.5)


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


def test_deterministic_thinning(db):
    patch(db, LOW, flat())
    patch(db, HIGH, flat(10))
    edge = overlap_edge(db)
    dense = materialize(db, edge, spacing=1, max_pairs=100000)
    thinned = materialize(db, edge, spacing=1, max_pairs=50)
    assert thinned == materialize(db, edge, spacing=1, max_pairs=50)
    assert len(thinned["pairs"]) == 50 and len(dense["pairs"]) > 500
    n = len(dense["pairs"])
    assert thinned["pairs"] == [dense["pairs"][i * n // 50] for i in range(50)]


def test_agreement_radius_is_the_detectors(db):
    # Two planes six voxels apart, accepted under a wide tolerance.
    patch(db, LOW, flat())
    patch(db, HIGH, lambda rows, cols: (cols + 10, rows, np.full_like(rows, 26)))
    report = db.link_patch_overlaps(volume_uuid=VOLUME, min_inliers=16, tolerance=7)
    assert report["edges_created"] == 1, report
    edge = db.edges_by_kind("same-surface")[0]
    assert edge["meta"]["fit"]["agreement_radius"] == 10
    result = materialize(db, edge)
    assert result["error"] is None and result["pairs"]
    assert all(5.9 < pair["distance"] < 6.1 for pair in result["pairs"])


def test_edges_without_recorded_fit_fields_must_be_redetected(db):
    patch(db, LOW, flat())
    patch(db, HIGH, flat(10))
    edge = overlap_edge(db)
    whole = dict(whole_geometry=True)
    for key in ("pose_from", "grid_scales", "agreement_radius"):
        fit = dict(edge["meta"]["fit"])
        del fit[key]
        old = db.add_edge(dict(a=edge["a"], b=edge["b"], kind="old-" + key, a_anchor=whole, b_anchor=whole,
                               meta=dict(edge["meta"], fit=fit)))
        [result] = db.overlap_correspondences([old], VOLUME)
        assert "rerun patch-overlap detection" in result["error"] and result["pairs"] == []


def test_unusable_edges_report_errors_independently(db):
    patch(db, LOW, flat())
    _, geometry = patch(db, HIGH, flat(10))
    edge = overlap_edge(db)
    whole = dict(whole_geometry=True)
    explicit = db.add_edge(dict(a=LOW, b=HIGH, kind="explicit", a_anchor=whole, b_anchor=whole,
                                meta={"fit": dict(edge["meta"]["fit"]), "coordinate_frame": None}))
    plain = db.add_edge(dict(a=LOW, b=HIGH, kind="plain", a_anchor=whole, b_anchor=whole))
    results = db.overlap_correspondences([explicit, edge["uuid"], plain, "00000000-0000-4000-8000-0000000000ff"], VOLUME)
    assert [r["error"] is None for r in results] == [False, True, False, False]
    assert "rerun patch-overlap detection" in results[0]["error"]
    assert results[2]["error"] == "edge has no overlap pose"
    assert results[3]["error"] == "edge not found"
    assert all(r["pairs"] == [] for r in results if r["error"])
    [other] = db.overlap_correspondences([edge["uuid"]], "other")
    assert "not in volume other" in other["error"]
    # A changed frame changes the grid scale the pose was fitted in.
    db.set_meta(HIGH, "coordinates", frame(2))
    assert "grid scale differs" in materialize(db, edge)["error"]
    db.set_meta(HIGH, "coordinates", frame(1))
    assert materialize(db, edge)["error"] is None
    # An edge kept bound to its recorded revision across a coordinate edit is stale.
    _, moved = patch(db, None, flat(11))
    geometry["arrays"] = moved["arrays"]
    db.put_geometry(HIGH, geometry, expected_revision=db.current_geometry(HIGH)["uuid"],
                    edge_edits=[edge_choice(db, e["uuid"], "keep_recorded") for e in db.affected_geometry_edges(HIGH, geometry)])
    assert db.edge_geometry_state(edge["uuid"])["stale"]
    assert materialize(db, edge)["error"] == "edge geometry is stale"
    with pytest.raises(ValueError):
        db.overlap_correspondences([edge["uuid"]], "")
    with pytest.raises(ValueError):
        db.overlap_correspondences([edge["uuid"]], VOLUME, max_pairs=0)
