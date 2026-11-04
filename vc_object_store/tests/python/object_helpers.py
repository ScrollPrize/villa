import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


VOLUME = "20260411134726"
LINE_OPTIONS = dict(volume_uuid=VOLUME, strip_spacing=2, strip_march_step=1,
                    anchor_contact=10, contact_gap=10, strip_margin=0)


def raw(db, revision, name):
    """The stored bytes of one array of a geometry revision."""
    return db.read_geometry_arrays([revision], name).values.tobytes()


def export(db, uuid, output, revision=None, valid_only=False):
    """Export an object with the vc_object_export tool, as users do."""
    build = os.environ.get('VC_OBJECT_STORE_BUILD_PYTHON_PATH')
    tool = Path(build).resolve().parent / 'bin' / 'vc_object_export' if build else None
    if not tool or not tool.exists():
        pytest.skip('needs the built vc_object_export')
    command = [str(tool), '--db', db.path, '--uuid', uuid, '--output', str(output)]
    if revision:
        command += ['--revision', revision]
    if valid_only:
        command.append('--valid-only')
    return subprocess.run(command, capture_output=True, text=True)


def edge_choice(db, edge_uuid, action, **anchors):
    edge = db.edge(edge_uuid)
    revisions = [db.current_geometry(edge[side]) for side in ("a", "b")]
    return dict(edge_uuid=edge_uuid, expected_generation=edge["generation"], action=action,
                a_revision=revisions[0]["uuid"] if revisions[0] else None,
                b_revision=revisions[1]["uuid"] if revisions[1] else None, **anchors)


def array(name, values, dtype):
    data = np.asarray(values, dtype=dtype)
    return {"name": name, "dtype": dtype, "shape": list(data.shape), "data": data.tobytes()}


def put(db, kind, arrays, attributes=None, **record):
    uid = db.upsert_object({"kind": kind, "storage_ref": "/missing/source", **record})
    geometry = {"kind": kind, "attributes": attributes or {}, "arrays": arrays}
    db.put_geometry(uid, geometry)
    return uid, geometry


def plane(db, offset=0, name="surface"):
    y, x = np.mgrid[:33, :33]
    points = np.stack([x + offset, y, np.full_like(x, 20)], axis=-1)
    return put(db, "surface", [array("points", points, "<f4")], {"scale": [1, 1]}, name=name)


def frame(scale=1, volume=VOLUME):
    return {"volume_uuid": volume, "scale_factor": scale}


def framed_patch(db, scale=1, size=33, offset=(0, 0, 20), hole=False):
    """A flat patch stored at scale, its frame declared in its metadata."""
    y, x = np.mgrid[:size, :size]
    xyz = ((np.stack([x, y, np.zeros_like(x)], axis=-1) + offset) * scale).astype(np.float32)
    if hole:
        xyz[0, 0] = -1
    return put(db, "surface", [array("points", xyz, "<f4")],
               {"scale": [1 / scale, 1 / scale]}, meta={"coordinates": frame(scale)})[0]


def framed_collection(db, scale=1, points=((10, 10, 20),), **metadata):
    return put(db, "point_collection", [
        array("points", np.asarray(points) * scale, "<f8"),
        array("point_ids", np.arange(len(points)) * 9 + 7, "<i8"),
    ], meta={"coordinates": frame(scale), **metadata})[0]
