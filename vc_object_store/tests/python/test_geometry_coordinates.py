"""One metadata-only coordinate contract for every geometry kind."""
import json
import struct
import uuid

import pytest
from vc_object_store import ObjectStore, read_geometry_source_coordinates
from object_helpers import _geometry, patch, sqlite_rows, track
from test_object_ingestion import _write_pcl_patch, fiber


KINDS = ("point_collection", "fiber", "patch", "track")
VOLUME = "abcdefab-1111-4111-8111-111111111111"
UNKNOWN = {"volume_uuid": None, "scale_factor": None}
LEGACY = {
    "vc_open_data_coordinate_space": "source-grid",
    "vc_open_data_source_path": "/unavailable/volume",
    "vc_open_data_source_coordinate_level": 2,
    "vc_open_data_source_coordinate_scale_factor": 4,
    "vc_open_data_source_original_resolution": 2.4,
}


def collection_document(path, **fields):
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    document = dict(vc_pointcollections_json_version="1", collections={
        str(i): dict(uuid=uid, points={str(i): {"p": [1.25, -2.5, 3.75]}})
        for i, uid in enumerate(ids)
    }, **fields)
    path.write_text(json.dumps(document))
    return ids, document


def heads(db, *ids):
    return {uid: db.current_object_revision(uid)["uuid"] for uid in ids}


def geometry(kind):
    if kind == "fiber":
        return _geometry()
    if kind == "patch":
        return patch(2, 2)
    if kind == "track":
        return track([(1, 2, 3), (4, 5, 6)])
    return dict(kind=kind, arrays=[
        dict(name="points", dtype="<f8", shape=[1, 3], data=struct.pack("<3d", 1.25, -2.5, 3.75)),
        dict(name="point_ids", dtype="<i8", shape=[1], data=struct.pack("<q", 42)),
    ])


def test_collection_import_maps_scale_discards_legacy_and_never_scales(db, tmp_path):
    source = tmp_path / "collections.json"
    ids, document = collection_document(source, **LEGACY)
    document["collections"]["1"].update(LEGACY)
    source.write_text(json.dumps(document))
    assert not db.ingest_geometry_source(
        "point_collection", source, coordinate_context={"volume_uuid": VOLUME.upper()})["problems"]
    for uid in ids:
        meta = db.object(uid)["meta"]
        assert meta["coordinates"] == {"volume_uuid": VOLUME, "scale_factor": 4}
        assert not (set(meta["source"]) & set(LEGACY))
        assert "coordinates" not in meta["source"]
        revision = db.current_geometry(uid)["uuid"]
        assert db.read_geometry_array(revision, "points") == struct.pack("<3d", 1.25, -2.5, 3.75)
    source.unlink()
    with ObjectStore.open(db.path, author="reopen") as reopened:
        assert reopened.object(ids[1])["meta"]["coordinates"] == db.object(ids[1])["meta"]["coordinates"]


@pytest.mark.parametrize("declarations,expected", [
    ({}, UNKNOWN),
    ({"vc_open_data_source_path": "/volume", "vc_open_data_coordinate_space": "volume",
      "vc_open_data_source_coordinate_level": 2}, UNKNOWN),
    ({"volume_uuid": VOLUME}, {**UNKNOWN, "volume_uuid": VOLUME}),
    ({"coordinates": {"scale_factor": 0.125}}, {**UNKNOWN, "scale_factor": 0.125}),
    ({"vc_open_data_source_coordinate_scale_factor": 2**60}, {**UNKNOWN, "scale_factor": 2**60}),
])
def test_unknown_partial_and_exact_integer_imports(db, tmp_path, declarations, expected):
    source = tmp_path / "collections.json"
    ids, document = collection_document(source)
    document["collections"]["0"].update(declarations)
    source.write_text(json.dumps(document))
    assert not db.ingest_geometry_source("point_collection", source)["problems"]
    assert db.object(ids[0])["meta"]["coordinates"] == expected
    assert db.object(ids[1])["meta"]["coordinates"] == UNKNOWN
    assert read_geometry_source_coordinates("point_collection", source, ids[0]) == expected


@pytest.mark.parametrize("bad", [
    {"volume_uuid": "not-a-uuid"}, {"volume_uuid": 17},
    {"scale_factor": 0}, {"scale_factor": -1}, {"scale_factor": True},
    {"scale_factor": "4"}, {"scale_factor": []}, {"scale_factor": float("inf")},
    {"scale_factor": float("nan")}, {"coordinate_space": "legacy"}, None, [],
])
def test_bad_coordinate_declarations_are_rejected(db, tmp_path, bad):
    source = tmp_path / "collections.json"
    ids, _ = collection_document(source, coordinates=bad)
    result = db.ingest_geometry_source("point_collection", source)
    assert result["problems"] and result["ingested"] == 0
    assert all(db.object(uid) is None for uid in ids)
    for kind in KINDS:
        with pytest.raises((RuntimeError, ValueError)):
            db.upsert_object(dict(kind=kind, meta={"coordinates": bad}))
        uid = db.upsert_object(dict(kind=kind))
        before = heads(db, uid)
        with pytest.raises((RuntimeError, ValueError)):
            db.set_meta(uid, "coordinates", bad)
        assert heads(db, uid) == before


@pytest.mark.parametrize("conflict", ["document-local", "document-context", "local-context", "legacy-modern", "volume"])
def test_import_declaration_conflicts_roll_back_all_collections(db, tmp_path, conflict):
    source = tmp_path / "collections.json"
    ids, document = collection_document(source)
    context = {}
    if conflict == "document-local":
        document["scale_factor"] = 4
        document["collections"]["1"]["scale_factor"] = 8
    elif conflict == "document-context":
        document["scale_factor"] = 4
        context["scale_factor"] = 8
    elif conflict == "local-context":
        document["collections"]["1"]["scale_factor"] = 4
        context["scale_factor"] = 8
    elif conflict == "legacy-modern":
        document.update(LEGACY, scale_factor=8)
    else:
        document["volume_uuid"] = VOLUME
        document["collections"]["1"]["volume_uuid"] = str(uuid.uuid4())
    source.write_text(json.dumps(document))
    result = db.ingest_geometry_source("point_collection", source, coordinate_context=context)
    assert "conflict" in result["problems"][0]
    assert result["ingested"] == 0 and all(db.object(uid) is None for uid in ids)


@pytest.mark.parametrize("kind", KINDS)
def test_direct_creation_revision_conflicts_and_geometry_reuse(db, kind):
    uid = db.upsert_object(dict(kind=kind, meta={"review": "keep"}))
    sibling = db.upsert_object(dict(kind=kind))
    assert db.object(uid)["meta"]["coordinates"] == UNKNOWN
    data = geometry(kind)
    revision = db.put_geometry(uid, data)
    before = heads(db, uid, sibling)
    counts = sqlite_rows(db.path, "SELECT (SELECT count(*) FROM geometry_revisions), "
                        "(SELECT count(*) FROM geometry_chunks)")
    with db.write(expected_revisions=before) as tx:
        tx.set_meta(uid, "coordinates", {"volume_uuid": VOLUME.upper(), "scale_factor": 0.5})
    assert db.object(uid)["meta"]["coordinates"] == {"volume_uuid": VOLUME, "scale_factor": 0.5}
    assert db.object_revision(before[uid])["record"]["meta"]["coordinates"] == UNKNOWN
    assert heads(db, sibling)[sibling] == before[sibling]
    assert db.current_object_revision(uid)["geometry_revision"] == revision
    assert sqlite_rows(db.path, "SELECT (SELECT count(*) FROM geometry_revisions), "
                       "(SELECT count(*) FROM geometry_chunks)") == counts
    for array in data["arrays"]:
        assert db.read_geometry_array(revision, array["name"]) == array["data"]
    with pytest.raises(RuntimeError, match="conflict"):
        with db.write(expected_revisions=before) as tx:
            tx.set_meta(sibling, "aborted", True)
    assert "aborted" not in db.object(sibling)["meta"]
    with ObjectStore.open(db.path, author="reopen") as reopened:
        assert reopened.object(uid)["meta"]["coordinates"]["scale_factor"] == 0.5


@pytest.mark.parametrize("kind", KINDS)
def test_coordinate_relationship_dependency_needs_explicit_decision(db, kind):
    a, b = [db.upsert_object(dict(kind=kind)) for _ in range(2)]
    for uid in (a, b):
        db.put_geometry(uid, geometry(kind))
    edge = db.add_edge(dict(kind="frame_link", a=a, b=b,
        a_anchor={"whole_geometry": True, "metadata_keys": ["coordinates"]},
        b_anchor={"whole_geometry": True, "metadata_keys": ["coordinates"]}))
    before = heads(db, a, b)
    with pytest.raises(RuntimeError, match="explicit"):
        with db.write(expected_revisions=before) as tx:
            tx.set_meta(a, "coordinates", {"scale_factor": 4})
    assert heads(db, a, b) == before
    with db.write(expected_revisions=before) as tx:
        tx.set_meta(a, "coordinates", {"scale_factor": 4})
        tx.keep_recorded_relationship(edge)
    assert db.relationships_at(before)[0]["applicable"]
    assert not db.relationships_at({a: None, b: None})[0]["applicable"]
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.retain_relationship(edge)
    assert db.relationships_at({a: None, b: None})[0]["applicable"]


@pytest.mark.parametrize("kind", KINDS)
def test_explicit_backfill_preserves_annotations_and_history(db, kind):
    uid = db.upsert_object(dict(kind=kind, meta={"review": "keep", "source": {**LEGACY, "note": "keep"}}))
    revision = db.put_geometry(uid, geometry(kind))
    before = heads(db, uid)
    with pytest.raises(RuntimeError, match="requires write"):
        with db.write() as tx:
            tx.backfill_geometry_coordinates(uid)
    with db.write(expected_revisions=before) as tx:
        tx.backfill_geometry_coordinates(uid, {"volume_uuid": VOLUME})
    assert db.object(uid)["meta"] == {
        "review": "keep", "source": {"note": "keep"},
        "coordinates": {"volume_uuid": VOLUME, "scale_factor": 4}}
    assert db.object_revision(before[uid])["record"]["meta"]["source"] == {**LEGACY, "note": "keep"}
    assert db.current_object_revision(uid)["geometry_revision"] == revision
    current = heads(db, uid)
    with db.write(expected_revisions=current) as tx:
        tx.backfill_geometry_coordinates(uid)
    assert heads(db, uid) == current
    with pytest.raises(RuntimeError, match="conflict"):
        with db.write(expected_revisions=current) as tx:
            tx.set_meta(uid, "review", "aborted")
            # Even a caught backfill conflict poisons the entire checked action.
            with pytest.raises(RuntimeError, match="conflict"):
                tx.backfill_geometry_coordinates(uid, {"scale_factor": 8})
    assert heads(db, uid) == current and db.object(uid)["meta"]["review"] == "keep"


def test_reingestion_does_not_refresh_but_source_backfill_does(db, tmp_path):
    source = tmp_path / "collections.json"
    ids, document = collection_document(source)
    assert not db.ingest_geometry_source("point_collection", source)["problems"]
    # Represent an older import with no coordinate metadata at all.
    for uid in ids:
        db.remove_meta(uid, "coordinates")
    before = heads(db, *ids)
    document.update(LEGACY, volume_uuid=VOLUME)
    source.write_text(json.dumps(document))
    assert db.ingest_geometry_source("point_collection", source)["unchanged"] == 2
    assert heads(db, *ids) == before
    declarations = read_geometry_source_coordinates("point_collection", source, ids[0])
    with db.write(expected_revisions=before) as tx:
        tx.backfill_geometry_coordinates(ids[0], declarations)
    assert db.object(ids[0])["meta"]["coordinates"] == {"volume_uuid": VOLUME, "scale_factor": 4}
    assert "coordinates" not in db.object(ids[1])["meta"]
    assert "coordinates" not in db.object_revision(before[ids[0]])["record"]["meta"]


@pytest.mark.parametrize("kind", ["fiber", "patch", "track"])
def test_other_source_formats_use_same_contract_without_rescaling(db, tmp_path, kind):
    uid = str(uuid.uuid4())
    if kind == "fiber":
        source = tmp_path / "fiber.json"
        source.write_text(json.dumps({**fiber(uid), **LEGACY}))
    elif kind == "patch":
        source = tmp_path / "surface"
        uid = _write_pcl_patch(source, 2, 3, 50, scale=(0.25, 0.5))
        metadata_path = source / "meta.json"
        metadata_path.write_text(json.dumps({**json.loads(metadata_path.read_text()), **LEGACY}))
    else:
        # Use the existing native exporter to build a packed-track fixture.
        original = db.upsert_object(dict(kind="track", track={"external_key": "42"}))
        db.put_geometry(original, geometry("track"))
        source = tmp_path / "tracks.vctracks"
        db.export_geometry(original, source)
    with ObjectStore.create(str(tmp_path / "baseline.vcobj"), author="baseline") as baseline:
        assert not baseline.ingest_geometry_source(kind, source)["problems"]
        reference = baseline.objects_by_kind(kind)[0]
        ref_geometry = baseline.current_geometry(reference["uuid"])
        assert not db.ingest_geometry_source(kind, source,
            coordinate_context={"volume_uuid": VOLUME, "scale_factor": 4})["problems"]
        if kind == "track":
            uid = next(o["uuid"] for o in db.objects_by_kind(kind) if o["uuid"] != original)
        current = db.current_geometry(uid)
        for array in current["arrays"]:
            assert db.read_geometry_array(current["uuid"], array["name"]) == baseline.read_geometry_array(
                ref_geometry["uuid"], array["name"])
    assert db.object(uid)["meta"]["coordinates"] == {"volume_uuid": VOLUME, "scale_factor": 4}
    assert not (set(db.object(uid)["meta"]["source"]) & set(LEGACY))
    if kind != "track":
        assert read_geometry_source_coordinates(kind, source, uid)["scale_factor"] == 4
    if kind == "patch":
        assert db.object(uid)["patch"]["scale"] == (0.25, 0.5)


def test_backfill_batch_coverage_conflicts_unknowns_and_unrelated_relationships(db):
    a, b = [db.upsert_object(dict(kind=kind)) for kind in ("point_collection", "patch")]
    edge = db.add_edge(dict(kind="annotation", a=a, b=b))
    before = heads(db, a, b)
    with db.write(expected_revisions=before) as tx:
        tx.backfill_geometry_coordinates(a)
    assert heads(db, a, b) == before
    with pytest.raises(RuntimeError, match="expected"):
        with db.write(expected_revisions={a: before[a]}) as tx:
            tx.backfill_geometry_coordinates(a, {"scale_factor": 4})
            tx.backfill_geometry_coordinates(b, {"scale_factor": 4})
    assert heads(db, a, b) == before
    with db.write(expected_revisions=before) as tx:
        tx.backfill_geometry_coordinates(a, {"scale_factor": 4})
        tx.backfill_geometry_coordinates(b, {"volume_uuid": VOLUME})
    assert db.relationships_at({a: None, b: None})[0]["applicable"]
    assert db.edge(edge)
    current = heads(db, a, b)
    with pytest.raises(RuntimeError, match="conflict"):
        with db.write(expected_revisions=current) as tx:
            tx.backfill_geometry_coordinates(b, {"scale_factor": 2})
            tx.backfill_geometry_coordinates(a, {"scale_factor": 8})
    assert heads(db, a, b) == current


@pytest.mark.parametrize("scale,context,conflict", [
    (4, 4.0, False),
    (2**60 + 1, float(2**60), True),
    (2**64 - 1, float(2**64), True),
])
def test_scale_conflicts_compare_exact_integers(db, tmp_path, scale, context, conflict):
    source = tmp_path / "collections.json"
    ids, _ = collection_document(source, scale_factor=scale)
    result = db.ingest_geometry_source("point_collection", source,
                                      coordinate_context={"scale_factor": context})
    if conflict:
        assert "conflict" in result["problems"][0]
        assert all(db.object(uid) is None for uid in ids)
    else:
        assert not result["problems"]
        assert db.object(ids[0])["meta"]["coordinates"]["scale_factor"] == scale


def test_source_reader_checks_identity_without_requiring_geometry(tmp_path):
    source = tmp_path / "collections.json"
    ids, document = collection_document(source, volume_uuid=VOLUME)
    del document["collections"]["0"]["points"]
    source.write_text(json.dumps(document))
    assert read_geometry_source_coordinates("point_collection", source, ids[0]) == {
        **UNKNOWN, "volume_uuid": VOLUME}
    with pytest.raises(RuntimeError, match="not found"):
        read_geometry_source_coordinates("point_collection", source, str(uuid.uuid4()))
    document["collections"]["duplicate"] = document["collections"]["0"]
    source.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="duplicate"):
        read_geometry_source_coordinates("point_collection", source, ids[0])
