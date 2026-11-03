"""Shared coordinate resolver and mixed-scale detector regression coverage."""
import json
import uuid

import numpy as np
import pytest

from vc_object_store import (
    Fiber, Track, geometry_coordinates, geometry_scale_to_frame, geometry_scale_to_volume,
    merge_geometry_coordinates, resolve_geometry_coordinates,
)
from object_helpers import LINE_OPTIONS, VOLUME, frame, framed_collection as collection, framed_patch as patch


def test_shared_resolution_and_round_trip():
    assert geometry_coordinates({}) == {"volume_uuid": None, "scale_factor": None}
    assert merge_geometry_coordinates({"volume_uuid": VOLUME}, {"scale_factor": 2}) == frame(2)
    assert geometry_scale_to_volume({}, VOLUME) is None
    assert geometry_scale_to_volume({"coordinates": frame(volume="other")}, VOLUME) is None
    for scale in (0.25, 1, 2, 8):
        multiplier = geometry_scale_to_frame({"coordinates": frame(scale)}, VOLUME)
        xyz = np.array([100., 200., 300.])
        np.testing.assert_array_equal(xyz * scale * multiplier, xyz)
    assert resolve_geometry_coordinates({"coordinates": frame(2)}, VOLUME) == frame(2)
    # Only the object's own metadata is authoritative, never a source declaration.
    assert geometry_scale_to_frame({"coordinates": frame(4), "source": {"coordinates": frame(99)}}, VOLUME) == 0.25


@pytest.mark.parametrize("metadata,reason", [
    ({}, "unknown"),
    ({"coordinates": {"volume_uuid": VOLUME}, "source": {"coordinates": frame(2)}}, "unknown"),
    ({"coordinates": frame(volume="other")}, "does not match"),
    ({"coordinates": frame(0)}, "positive"),
    ({"coordinates": frame(-1)}, "positive"),
    ({"coordinates": {"scale_factor": 1, "typo": 1}}, "unknown geometry coordinate field"),
    ({"coordinates": frame(True)}, None),
    ({"coordinates": frame(float("nan"))}, None),
])
def test_invalid_frames(metadata, reason):
    with pytest.raises((ValueError, RuntimeError), match=reason):
        geometry_scale_to_frame(metadata, VOLUME)


def test_mixed_pcl_scales_selection_anchors_and_sentinels(db):
    a = patch(db, 2, hole=True)
    b = patch(db, 0.5, size=17)
    first, second = collection(db, 4), collection(db, 0.25)
    before = {uid: db.object(uid) for uid in (a, b, first, second)}
    report = db.link_point_collections_to_surfaces(volume_uuid=VOLUME)
    assert report["edges_created"] == 2
    for uid, scale in ((first, 4), (second, 0.25)):
        edge, = db.edges_of(uid)
        assert {edge["a"], edge["b"]} == {a, uid}  # physical area wins across scales
        anchor = edge["a_anchor"] if edge["a"] == uid else edge["b_anchor"]
        other = edge["b_anchor"] if edge["a"] == uid else edge["a_anchor"]
        assert anchor["position"] == (10 * scale, 10 * scale, 20 * scale)
        assert other["position"] == (20, 20, 40)  # source patch coordinates
        assert edge["meta"]["point_collection_surface"]["ij"] == [10, 10]
        assert edge["meta"]["settings"]["point_scale"] == 1 / scale
        assert edge["meta"]["coordinate_frame"]["objects"][uid] == {"coordinates": frame(scale)}
    assert db.link_point_collections_to_surfaces(volume_uuid=VOLUME)["edges_skipped_existing"] == 2
    assert {uid: db.object(uid) for uid in before} == before
    # The hole at grid (0,0) remains invalid after scaling (it must not become -0.5).
    hole = collection(db, points=((-0.5, -0.5, -0.5),))
    assert db.link_point_collections_to_surfaces(volume_uuid=VOLUME, collection_uuids=[hole],
                                               surface_uuids=[a], tolerance=0.1)["edges_created"] == 0
    assert len(db.edges_of(first)) == 1


def test_ingest_coordinate_context_fills_and_rejects_conflicts(db, tmp_path):
    source = tmp_path / "collections.json"

    def write(collections):
        source.write_text(json.dumps({"vc_pointcollections_json_version": "1", "collections": {
            str(i): {"uuid": str(uuid.uuid4()), "name": f"c{i}", "points": {str(i): {"p": [1.0, 2.0, 3.0]}}, **extra}
            for i, extra in enumerate(collections)}}))
    write([{}, {"coordinates": {"volume_uuid": VOLUME}}, {"scale_factor": 2}])
    report = db.ingest_geometry_source("point_collection", str(source), coordinate_context=frame(2))
    assert report["ingested"] == 3 and not report["problems"], report
    for record in db.select_objects():
        assert db.object(record["uuid"])["meta"]["coordinates"] == frame(2)
    before = db.journal_since()
    for declared in ({"scale_factor": 4}, {"volume_uuid": "other"}):
        write([{}, declared])
        report = db.ingest_geometry_source("point_collection", str(source), coordinate_context=frame(2))
        assert report["ingested"] == 0 and "conflict" in " ".join(report["problems"]), report
    assert db.journal_since() == before


def test_mixed_patch_overlap_parameterization_and_scope(db):
    a, b, c = patch(db, 2), patch(db, 0.5), patch(db, 4)
    report = db.link_surface_overlaps(volume_uuid=VOLUME)
    assert report["edges_created"] == 3, report
    for edge in db.edges_of(a):
        fit = edge["meta"]["fit"]
        np.testing.assert_allclose(fit["pose"], [1, 0, 0, 0, 1, 0], atol=1e-8)
        assert fit["grid_scales"] == {edge["a"]: [1, 1], edge["b"]: [1, 1]}
        assert fit["pose_units"] == "volume_voxels"
        assert {fit["pose_from"], fit["pose_to"]} == {edge["a"], edge["b"]}
    assert db.link_surface_overlaps(volume_uuid=VOLUME)["edges_skipped_existing"] == 3
    assert db.link_surface_overlaps(volume_uuid=VOLUME, surface_uuids=[a, b], min_inliers=10000)["stale_removed"] == 1
    assert len(db.edges_of(c)) == 2


def fiber(db, scale, xyz=None):
    xyz = np.array(xyz if xyz is not None else [[x, 96, 96] for x in range(72, 121)], dtype=float) * scale
    uid = db.upsert_object({"kind": "fiber", "meta": {"coordinates": frame(scale)}})
    db.put_geometry(uid, Fiber(xyz[[0, -1]].tolist(), xyz.tolist()))
    return uid


def u8_zarr(path, data):
    """Uncompressed single-chunk uint8 Zarr v2 array, no Python Zarr dependency."""
    path.mkdir(parents=True)
    (path / ".zarray").write_text(json.dumps(dict(zarr_format=2, shape=data.shape, chunks=data.shape,
        dtype="|u1", compressor=None, fill_value=0, order="C", filters=None)))
    (path / "0.0.0").write_bytes(np.ascontiguousarray(data, dtype=np.uint8).tobytes())


@pytest.mark.parametrize("kind", ["fiber", "track"])
@pytest.mark.parametrize("ct_downsample", [1, 2])
def test_mixed_lines_and_patch_with_ct(db, tmp_path, kind, ct_downsample):
    # Full tensor support.
    ct = tmp_path / "ct.zarr"
    shape = (192, 192, 192)
    z = np.arange(shape[0], dtype=float)
    intensity = (200 * np.exp(-((z - 96) / 3)**2)).astype(np.uint8)
    u8_zarr(ct, np.broadcast_to(intensity[:, None, None], shape))
    a, b = patch(db, 2, 65, (64, 64, 96)), patch(db, 0.5, 65, (64, 64, 96))
    def line(scale):
        if kind == "fiber":
            return fiber(db, scale)
        uid = db.upsert_object({"kind": "track", "meta": {"coordinates": frame(scale)},
            "track": {"dataset_ref": "fixture", "external_key": str(scale), "family": "horizontal"}})
        db.put_geometry(uid, Track([[int(x * scale), int(96 * scale), int(96 * scale)] for x in range(72, 121)]))
        return uid
    second_scale = 0.25 if kind == "fiber" else 2
    first, second = line(4), line(second_scale)
    for uid, scale in ((a, 2), (b, 0.5), (first, 4), (second, second_scale)):
        db.set_meta(uid, "coordinates", frame(scale / ct_downsample))
    link = db.link_fibers_to_surfaces if kind == "fiber" else db.link_tracks_to_surfaces
    options = dict(LINE_OPTIONS, ct_to_base_scale=ct_downsample)
    report = link(str(ct), **options)
    assert report["edges_created"] == 4, report
    for uid, scale in ((first, 4), (second, second_scale)):
        assert len(db.edges_of(uid)) == 2
        for edge in db.edges_of(uid):
            anchor = edge["a_anchor"] if edge["a"] == uid else edge["b_anchor"]
            assert anchor["position"] == (96 * scale, 96 * scale, 96 * scale)
            assert anchor["points"][0]["index"] == [24]
            assert edge["meta"][kind + "_surface"]["surface_grid"] == [32, 32]
            assert edge["meta"]["settings"][kind + "_scale"] == ct_downsample / scale
    assert link(str(ct), **options)["edges_skipped_existing"] == 4
    # Explicit-scale callers still use their supplied multipliers.
    explicit = {kind + "_uuids": [first], "surface_uuids": [a], kind + "_scale": ct_downsample / 4,
                "surface_scale": ct_downsample / 2, **{k: v for k, v in options.items() if k != "volume_uuid"}}
    assert link(str(ct), **explicit)["edges_updated"] == 1
    assert len(db.edges_of(first)) == 2
    selected = next(e for e in db.edges_of(first) if a in (e["a"], e["b"]))
    assert selected["meta"]["coordinate_frame"] is None


def test_fiber_patch_lasagna_normals(db, tmp_path):
    # Paired nx/ny group arrays, addressed by path (an array's only level is 0)
    # as the spiral fitter and .lasagna.json address them.
    def lasagna(value):
        for name in ("nx", "ny"):
            u8_zarr(tmp_path / str(value) / f"{name}.zarr" / "2", np.full((96,) * 3, value, np.uint8))
        return dict(normal_x_volume=str(tmp_path / str(value) / "nx.zarr" / "2"),
                    normal_y_volume=str(tmp_path / str(value) / "ny.zarr" / "2"), ct_level=0, ct_to_base_scale=2)
    a = patch(db, 1, 65, (64, 64, 96))
    uid = fiber(db, 1)
    # 0/0 encodes no valid normal: nothing links.
    report = db.link_fibers_to_surfaces(**LINE_OPTIONS, **lasagna(0))
    assert report["edges_created"] == 0 and db.edges_of(uid) == []
    assert any("no valid sheet normals" in problem for problem in report["problems"]), report
    # 128/128 decodes to nx = ny = 0, so the sheet normal is +z, as for the patch.
    normals = lasagna(128)
    report = db.link_fibers_to_surfaces(**LINE_OPTIONS, **normals)
    assert report["edges_created"] == 1, report
    edge = db.edges_of(uid)[0]
    assert a in (edge["a"], edge["b"])
    assert edge["meta"]["normal_method"] == "lasagna-nxny-u8-nearest-v1"
    assert edge["meta"]["normal_source"] == {"normal_x": normals["normal_x_volume"],
                                             "normal_y": normals["normal_y_volume"]}
    assert edge["meta"]["fiber_surface"]["surface_grid"] == [32, 32]


def test_line_patch_without_normals_uses_centerline(db):
    a = patch(db, 1, 65, (64, 64, 96))
    uid = fiber(db, 1)
    report = db.link_fibers_to_surfaces(**LINE_OPTIONS)
    assert report["edges_created"] == 1, report
    edge = db.edges_of(uid)[0]
    assert a in (edge["a"], edge["b"])
    assert edge["meta"]["normal_method"] == "centerline"
    assert edge["meta"]["normal_source"] is None
    assert edge["meta"]["fiber_surface"]["anchor"] is True
    assert edge["meta"]["fiber_surface"]["strip_same"] == 1


def test_centerline_gate_rejects_adjacent_sheet(db):
    # The fiber touches the patch only where a short dip reaches it; elsewhere it
    # runs one sheet away, so the centerline is mostly adjacent, not same-sheet.
    patch(db, 1, 65, (64, 64, 96))
    fiber(db, 1, [[x, 96, 96 if 94 <= x <= 106 else 104] for x in range(72, 129)])
    assert db.link_fibers_to_surfaces(**LINE_OPTIONS)["edges_created"] == 0
