"""The line-surface detectors' normal sources: CT, lasagna nx/ny arrays, a lasagna
resident-pool pack, or none (centerline-only linking), standalone and through
link_objects, locally and on a served store."""
import json
import os
import re
import subprocess
import uuid

import numpy as np
import pytest

import vc_object_store as vos
from vc_object_store import Fiber

SERVER = os.environ.get("VC_OBJECT_SERVER")
LINE_OPTIONS = dict(strip_spacing=2, strip_march_step=1, anchor_contact=10, contact_gap=10, strip_margin=0)
LASAGNA_METHOD = "lasagna-nxny-u8-nearest-v1"


def array(name, values, dtype):
    data = np.asarray(values, dtype=dtype)
    return {"name": name, "dtype": dtype, "shape": list(data.shape), "data": data.tobytes()}


def patch(db, size=65, offset=(64, 64, 96)):
    """A flat patch of `size`² unit cells at `offset`, normal +z."""
    y, x = np.mgrid[:size, :size]
    xyz = (np.stack([x, y, np.zeros_like(x)], axis=-1) + offset).astype(np.float32)
    uid = db.upsert_object({"kind": "surface", "storage_ref": "/missing/source", "name": "patch"})
    db.put_geometry(uid, {"kind": "surface", "attributes": {"scale": [1, 1]}, "arrays": [array("points", xyz, "<f4")]})
    return uid


def fiber(db, xyz=None):
    """A fiber along x through the patch's middle, lying on it unless `xyz` says otherwise."""
    xyz = np.array(xyz if xyz is not None else [[x, 96, 96] for x in range(72, 121)], dtype=float)
    uid = db.upsert_object({"kind": "fiber"})
    db.put_geometry(uid, Fiber(xyz[[0, -1]].tolist(), xyz.tolist()))
    return uid


def u8_zarr(path, data):
    """Uncompressed single-chunk uint8 Zarr v2 array, no Python Zarr dependency."""
    path.mkdir(parents=True)
    (path / ".zarray").write_text(json.dumps(dict(zarr_format=2, shape=data.shape, chunks=data.shape,
        dtype="|u1", compressor=None, fill_value=0, order="C", filters=None)))
    (path / "0.0.0").write_bytes(np.ascontiguousarray(data, dtype=np.uint8).tobytes())


def lasagna_arrays(root, value, shape=(96, 96, 96)):
    """Paired nx/ny group arrays holding `value`, addressed by path as the spiral fitter
    and .lasagna.json address them (an array's only level is 0). At ct_to_base_scale 2
    they cover working XYZ 0..192."""
    for name in ("nx", "ny"):
        u8_zarr(root / f"{name}.zarr" / "2", np.full(shape, value, np.uint8))
    return dict(normal_x_volume=str(root / "nx.zarr" / "2"), normal_y_volume=str(root / "ny.zarr" / "2"),
                ct_level=0, ct_to_base_scale=2)


def lasagna_pack(root, value=128, shape=(96, 96, 96), brick=(32, 32, 32), drop=(), identity="fixture"):
    """A resident-pool pack (spiral-fitting's pack_resident_pools.py, respool v2) whose
    present bricks all hold `value` in both channels; `drop` names absent bricks (grid ZYX)."""
    root.mkdir(parents=True)
    grid = tuple(s // b for s, b in zip(shape, brick))
    slots = [(z, y, x) for z in range(grid[0]) for y in range(grid[1]) for x in range(grid[2]) if (z, y, x) not in drop]
    table = np.zeros(grid, np.int32)
    for row, slot in enumerate(slots, 1):
        table[slot] = row
    np.save(root / "table.npy", table)
    rows = 1 + len(slots)
    voxels = int(np.prod(brick))
    for channel in range(2):
        pool = np.full((rows, voxels), value, np.uint8)
        pool[0] = 0
        (root / f"channel_{channel}.u8").write_bytes(pool.tobytes())
    (root / "meta.json").write_text(json.dumps(dict(
        format="respool", version=2, array_shape=list(shape), source_chunk_shape=list(brick),
        brick_shape=list(brick), grid_shape=list(grid), rows=rows, channels=["nx", "ny"],
        channel_names=["normals/nx", "normals/ny"], dtype="u1", lasagna_identity=identity, ct_mask=None)))
    return dict(normal_pool=str(root), ct_to_base_scale=2)


def ct_slab(root, shape=(192, 192, 192)):
    """A uint8 CT whose intensity is a gaussian slab at z = 96: structure-tensor normals are +z."""
    z = np.arange(shape[0], dtype=float)
    intensity = (200 * np.exp(-((z - 96) / 3) ** 2)).astype(np.uint8)
    u8_zarr(root / "ct.zarr", np.broadcast_to(intensity[:, None, None], shape))
    return str(root / "ct.zarr")


def only_edge(db, uid):
    edges = db.edges_of(uid)
    assert len(edges) == 1, edges
    return edges[0]


def test_ct_normals_link_a_fiber_on_its_patch(db, tmp_path):
    a, uid = patch(db), fiber(db)
    ct = ct_slab(tmp_path)
    report = db.link_fibers_to_surfaces(ct, **LINE_OPTIONS)
    assert report["edges_created"] == 1, report
    edge = only_edge(db, uid)
    assert a in (edge["a"], edge["b"])
    assert edge["meta"]["normal_method"].startswith("ct-")
    assert edge["meta"]["normal_source"] == ct
    assert edge["meta"]["fiber_surface"]["anchor"] is True


def test_lasagna_arrays_supply_strip_normals(db, tmp_path):
    a, uid = patch(db), fiber(db)
    # nx = ny = 0 encodes no valid normal: no strip can be built, nothing links.
    report = db.link_fibers_to_surfaces(**LINE_OPTIONS, **lasagna_arrays(tmp_path / "none", 0))
    assert report["edges_created"] == 0 and db.edges_of(uid) == []
    assert any("no valid sheet normals" in problem for problem in report["problems"]), report
    # 128/128 decodes to nx = ny = 0, so the sheet normal is +z, as for the patch.
    normals = lasagna_arrays(tmp_path / "flat", 128)
    report = db.link_fibers_to_surfaces(**LINE_OPTIONS, **normals)
    assert report["edges_created"] == 1, report
    edge = only_edge(db, uid)
    assert a in (edge["a"], edge["b"])
    assert edge["meta"]["normal_method"] == LASAGNA_METHOD
    assert edge["meta"]["normal_source"] == {"normal_x": normals["normal_x_volume"], "normal_y": normals["normal_y_volume"]}
    assert edge["meta"]["fiber_surface"]["surface_grid"] == [32, 32]
    assert edge["meta"]["fiber_surface"]["anchor"] is True
    # Only one of the lasagna volumes is a parameter error.
    with pytest.raises(ValueError, match="normal_x_volume and normal_y_volume"):
        db.link_fibers_to_surfaces(**LINE_OPTIONS, normal_x_volume=normals["normal_x_volume"])


def test_lasagna_pack_supplies_strip_normals(db, tmp_path):
    a, uid = patch(db), fiber(db)
    # The fiber lies in lasagna brick (1, 1, 1) at ct_to_base_scale 2; a pack
    # that dropped it has no normal along the fiber.
    report = db.link_fibers_to_surfaces(**LINE_OPTIONS, **lasagna_pack(tmp_path / "holed", drop=[(1, 1, 1)]))
    assert report["edges_created"] == 0 and db.edges_of(uid) == []
    assert any("no valid sheet normals" in problem for problem in report["problems"]), report
    pack = lasagna_pack(tmp_path / "full")
    report = db.link_fibers_to_surfaces(**LINE_OPTIONS, **pack)
    assert report["edges_created"] == 1, report
    edge = only_edge(db, uid)
    assert a in (edge["a"], edge["b"])
    assert edge["meta"]["normal_method"] == LASAGNA_METHOD
    assert edge["meta"]["normal_source"] == {"normal_pool": pack["normal_pool"]}
    assert edge["meta"]["fiber_surface"]["anchor"] is True
    # A rerun from the strip cache gives the same answer.
    cached = db.link_fibers_to_surfaces(**LINE_OPTIONS, **pack, cache_directory=str(tmp_path / "strips"))
    assert cached["edges_skipped_existing"] == 1, cached
    assert db.link_fibers_to_surfaces(**LINE_OPTIONS, **pack, cache_directory=str(tmp_path / "strips"))["edges_skipped_existing"] == 1
    assert list((tmp_path / "strips").rglob("*")), "the strip cache was written"


def test_tracks_take_the_same_normal_sources(db, tmp_path):
    a = patch(db)
    uid = db.upsert_object({"kind": "track", "storage_ref": "/missing/source",
                            "track": {"dataset_ref": "fixture", "external_key": "0", "family": "horizontal"}})
    db.put_geometry(uid, {"kind": "track", "attributes": {"family": "horizontal"},
                          "arrays": [array("points", [[x, 96, 96] for x in range(72, 121)], "<i4")]})
    pack = lasagna_pack(tmp_path / "pack")
    report = db.link_tracks_to_surfaces(**LINE_OPTIONS, **pack)
    assert report["edges_created"] == 1, report
    edge = only_edge(db, uid)
    assert a in (edge["a"], edge["b"])
    assert edge["meta"]["normal_method"] == LASAGNA_METHOD
    assert edge["meta"]["normal_source"] == {"normal_pool": pack["normal_pool"]}
    assert db.link_tracks_to_surfaces(**LINE_OPTIONS)["edges_updated"] == 1
    assert only_edge(db, uid)["meta"]["normal_method"] == "centerline"


def test_without_normals_the_centerline_gates_anchors(db):
    a, uid = patch(db), fiber(db)
    report = db.link_fibers_to_surfaces(**LINE_OPTIONS)
    assert report["edges_created"] == 1, report
    edge = only_edge(db, uid)
    assert a in (edge["a"], edge["b"])
    assert edge["meta"]["normal_method"] == "centerline"
    assert edge["meta"]["normal_source"] is None
    assert edge["meta"]["fiber_surface"]["anchor"] is True
    assert edge["meta"]["fiber_surface"]["strip_same"] == 1


def test_centerline_gate_rejects_an_adjacent_sheet(db):
    # The fiber touches the patch only where a short dip reaches it; elsewhere it
    # runs one sheet away, so the centerline is mostly adjacent, not same-sheet.
    patch(db)
    uid = fiber(db, [[x, 96, 96 if 94 <= x <= 106 else 104] for x in range(72, 129)])
    assert db.link_fibers_to_surfaces(**LINE_OPTIONS)["edges_created"] == 0
    assert db.edges_of(uid) == []


def test_several_sources_are_refused_before_any_work(db, tmp_path):
    patch(db), fiber(db)
    normals = lasagna_arrays(tmp_path / "flat", 128)
    pack = lasagna_pack(tmp_path / "pack")
    ct = ct_slab(tmp_path)
    before = db.status()["commit_sequence"]
    for conflicting in (dict(normals, normal_pool=pack["normal_pool"]),):
        with pytest.raises(ValueError, match="one normal source"):
            db.link_fibers_to_surfaces(**LINE_OPTIONS, **conflicting)
    with pytest.raises(ValueError, match="one normal source"):
        db.link_fibers_to_surfaces(ct, **LINE_OPTIONS, **pack)
    with pytest.raises(ValueError, match="one normal source"):
        db.link_fibers_to_surfaces(ct, **LINE_OPTIONS, **normals)
    assert db.status()["commit_sequence"] == before


def test_an_unavailable_source_fails_the_run(db, tmp_path):
    patch(db), fiber(db)
    before = db.status()["commit_sequence"]
    with pytest.raises(RuntimeError):
        db.link_fibers_to_surfaces(**LINE_OPTIONS, normal_pool=str(tmp_path / "no-such-pack"), ct_to_base_scale=2)
    with pytest.raises(FileNotFoundError):
        db.link_fibers_to_surfaces(**LINE_OPTIONS, normal_x_volume=str(tmp_path / "no-x"),
                                   normal_y_volume=str(tmp_path / "no-y"), ct_to_base_scale=2)
    assert db.status()["commit_sequence"] == before


def truncated_pack(root):
    """A pack whose table names a brick row its channel files do not hold: the
    source itself cannot be read where the fiber lies."""
    pack = lasagna_pack(root)
    rows = json.loads((root / "meta.json").read_text())["rows"]
    table = np.load(root / "table.npy")
    table[1, 1, 1] = rows - 1  # the fiber's brick -> the last row ...
    np.save(root / "table.npy", table)
    for channel in range(2):  # ... which the pools no longer hold
        voxels = 32 ** 3
        data = (root / f"channel_{channel}.u8").read_bytes()
        (root / f"channel_{channel}.u8").write_bytes(data[: (rows - 1) * voxels])
    meta = json.loads((root / "meta.json").read_text())
    meta["rows"] = rows - 1
    (root / "meta.json").write_text(json.dumps(meta))
    return pack


def test_a_source_that_fails_while_sampling_fails_the_run(db, tmp_path):
    # An unreadable brick is the source failing, not the fiber: no skipped
    # candidate, no commit.
    patch(db), fiber(db)
    before = db.status()["commit_sequence"]
    with pytest.raises(RuntimeError, match="lasagna pack brick"):
        db.link_fibers_to_surfaces(**LINE_OPTIONS, **truncated_pack(tmp_path / "truncated"))
    assert db.status()["commit_sequence"] == before


# --- link_objects: the same sources as plan parameters ---

def link_request(status, plan):
    return dict(command_version=1, request=str(uuid.uuid4()), store=status["store"], generation=status["generation"],
                author="py", expected_objects={}, expected_relationships={}, read_dependencies={},
                expected_commit_sequence=None, operations=[{"op": vos.LINK_OBJECTS_OPERATION, "plan": plan}])


def fiber_plan(ct_volume="", **params):
    step = {"detector": "fiber-surface", "params": dict(LINE_OPTIONS, **params)}
    if ct_volume:
        step["ct_volume"] = ct_volume
    return {"detectors": [step], "marker": {"key": "normals"}}


@pytest.mark.parametrize("source", ["ct", "lasagna", "pack", "none"])
def test_link_objects_takes_every_normal_source(db, tmp_path, source):
    a, uid = patch(db), fiber(db)
    plan = {"ct": lambda: fiber_plan(ct_slab(tmp_path)),
            "lasagna": lambda: fiber_plan(**lasagna_arrays(tmp_path / "flat", 128)),
            "pack": lambda: fiber_plan(**lasagna_pack(tmp_path / "pack")),
            "none": lambda: fiber_plan()}[source]()
    receipt = db.submit(link_request(db.status(), plan))
    assert receipt["commit_sequence"] == db.status()["commit_sequence"]
    edge = only_edge(db, uid)
    assert a in (edge["a"], edge["b"])
    expected_method = {"ct": "ct-", "lasagna": LASAGNA_METHOD, "pack": LASAGNA_METHOD, "none": "centerline"}[source]
    assert edge["meta"]["normal_method"].startswith(expected_method)
    marker = db.object(vos.link_marker_uuid())
    step = marker["meta"]["detectors"][0]
    assert step["params"] == plan["detectors"][0]["params"]
    assert step.get("ct_volume") == plan["detectors"][0].get("ct_volume")
    assert step["report"]["edges_created"] == 1


def test_link_objects_refuses_conflicting_sources_and_fails_atomically_on_a_missing_one(db, tmp_path):
    patch(db), fiber(db)
    db.submit(link_request(db.status(), fiber_plan()))
    before = db.status()["commit_sequence"]
    marker = db.object(vos.link_marker_uuid())
    conflicting = fiber_plan(ct_slab(tmp_path), **lasagna_pack(tmp_path / "pack"))
    with pytest.raises(ValueError, match="one normal source"):
        db.submit(link_request(db.status(), conflicting))
    missing = fiber_plan(normal_pool=str(tmp_path / "no-such-pack"), ct_to_base_scale=2)
    with pytest.raises(vos.PreparationFailed, match="fiber-surface"):
        db.submit(link_request(db.status(), missing))
    truncated = fiber_plan(**truncated_pack(tmp_path / "truncated"))
    with pytest.raises(vos.PreparationFailed, match="lasagna pack brick"):
        db.submit(link_request(db.status(), truncated))
    assert db.status()["commit_sequence"] == before
    assert db.object(vos.link_marker_uuid()) == marker


# --- served stores: sources resolve on the service's host ---

pytestmark_hosted = pytest.mark.skipif(not SERVER, reason="set VC_OBJECT_SERVER to a vc_object_server binary")


@pytest.fixture()
def hosted(tmp_path):
    path = tmp_path / "hosted.vcobj"
    with vos.ObjectStore.create(str(path), author="pytest", hostname="testhost") as db:
        a, uid = patch(db), fiber(db)
    server = subprocess.Popen([SERVER, str(path), "--port", "0", "--strip-cache", str(tmp_path / "server-strips")],
                              stderr=subprocess.PIPE, text=True)
    try:
        line = server.stderr.readline()
        port = re.search(r":(\d+)\s*$", line)
        assert port, line
        yield vos.HostedStore(f"http://127.0.0.1:{port.group(1)}", retries=1), path, a, uid
    finally:
        server.terminate()
        assert server.wait(timeout=30) == 0


def inspect(path):
    return vos.ObjectStore.open(str(path), read_only=True)


@pytestmark_hosted
@pytest.mark.parametrize("source", ["lasagna", "pack", "none"])
def test_hosted_link_objects_reads_the_source_on_the_service(hosted, tmp_path, source):
    store, path, a, uid = hosted
    plan = {"lasagna": lambda: fiber_plan(**lasagna_arrays(tmp_path / "flat", 128)),
            "pack": lambda: fiber_plan(**lasagna_pack(tmp_path / "pack")),
            "none": lambda: fiber_plan()}[source]()
    receipt = store.submit(link_request(store.status(), plan))
    assert receipt["commit_sequence"] == store.status()["commit_sequence"]
    with inspect(path) as db:
        edge = only_edge(db, uid)
        assert a in (edge["a"], edge["b"])
        assert edge["meta"]["normal_method"] == ("centerline" if source == "none" else LASAGNA_METHOD)
        assert db.object(vos.link_marker_uuid())["meta"]["detectors"][0]["report"]["edges_created"] == 1
    # The same answer from the service's strip cache.
    again = store.submit(link_request(store.status(), plan))
    assert again["commit_sequence"] == receipt["commit_sequence"] + 1
    with inspect(path) as db:
        assert db.object(vos.link_marker_uuid())["meta"]["detectors"][0]["report"]["edges_skipped_existing"] == 1


@pytestmark_hosted
def test_hosted_link_objects_fails_on_a_source_the_service_cannot_read(hosted, tmp_path):
    store, path, a, uid = hosted
    store.submit(link_request(store.status(), fiber_plan()))
    before = store.status()["commit_sequence"]
    with inspect(path) as db:
        marker = db.object(vos.link_marker_uuid())
    with pytest.raises(vos.PreparationFailed, match="fiber-surface"):
        store.submit(link_request(store.status(), fiber_plan(normal_pool=str(tmp_path / "no-such-pack"), ct_to_base_scale=2)))
    with pytest.raises(ValueError, match="one normal source"):
        store.submit(link_request(store.status(), fiber_plan(ct_slab(tmp_path), **lasagna_pack(tmp_path / "pack"))))
    assert store.status()["commit_sequence"] == before
    with inspect(path) as db:
        assert db.object(vos.link_marker_uuid()) == marker
