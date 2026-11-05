"""Object stores of the open project: importing and exporting objects."""

from __future__ import annotations

from typing import Any, Literal, Optional

from mcp.server.fastmcp import Context
from typing_extensions import TypedDict

from ..core import _call, _strip_none, _wait_for_job, mcp


class _Binding(TypedDict):
    scale: float


@mcp.tool()
async def vc3d_import_objects(
    kind: Literal["surface", "fiber", "point_collection"],
    path: str,
    tags: Optional[list[str]] = None,
    bind: Optional[_Binding] = None,
    uuid_namespace: Optional[str] = None,
    store: Optional[str] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Import source files into an object store of the open project, as new
    objects. Returns an "import" job; pass wait=true to get its terminal record.

    path is an absolute path on the filesystem where VC3D runs:
    - kind="surface": a tifxyz directory, or a folder of them;
    - kind="fiber": a vc3d_fiber JSON file, a fiber collection, or a folder of
      fiber JSON files;
    - kind="point_collection": a vc_pointcollections JSON file (every
      collection in it).
    Import copies the geometry into the .vcobj store; nothing references the
    source files afterwards. Large folders need room for that copy.

    tags are added to every imported object. Omitted, the last component of
    path is the tag (as in Stores > Import from Folder); [] adds none.

    bind={"scale": s} binds the objects to the current volume's annotation
    volume at s stored units per *current-volume* voxel (1 for segments traced
    on this volume). There is no default scale. Without bind, sources keep the
    coordinates they declare, and a source declaring none stays unbound (not
    shown on a volume). A binding that conflicts with a source's own
    declaration is reported as a problem of that source. bind needs an open
    volume.

    uuid_namespace: a UUID naming this dataset. Sources without canonical UUIDs
    (VC3D-era tifxyz directories, older fibers and point collections) get
    identities derived from it and their names; without it they are listed as
    problems. Use one fresh UUID per dataset and reuse it for every later
    import of the same files: a repeat then reports the objects as already
    existing instead of duplicating them.

    store: a store reference id from vc3d_get_state's "stores". Omitted, the
    store routed for this kind ("routes") receives the objects; it must be
    writable. Import never changes routing: when the receiving store is not the
    one routed for the kind, call vc3d_route_store(kind, store_id=<the job's
    result.store>) before listing, activating or editing the objects.

    The job's result is {store, ingested, edgesImported, problems, uuids,
    existing, cancelled, fatalError}. A job that traversed the whole source
    succeeds even when some sources were problems or already existed: inspect
    problems/existing. uuids are the new objects' UUIDs (segment ids for
    surfaces). A failure of the store itself stops the import: the job fails
    with fatalError set and the report of what was committed before it.
    vc3d_cancel_job stops an import before its next object; the job then fails
    with cancelled=true and the partial report. Fiber and point-collection
    sources commit all at once or not at all. Fails with -32004 while another
    import (or, for fibers, a fiber merge, split or link) runs.
    """
    result = await _call(
        "objects.import",
        _strip_none(
            {
                "kind": kind,
                "path": path,
                "tags": tags,
                "bind": bind,
                "uuidNamespace": uuid_namespace,
                "store": store,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_export_object(
    uuid: str,
    path: str,
    store: Optional[str] = None,
    overwrite: bool = False,
    valid_only: bool = False,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Export one surface, fiber or point collection of the open project's
    stores as native files. Returns an "export" job; pass wait=true to get its
    terminal record.

    uuid is the object's UUID: a segment id from vc3d_list_segments, the
    "uuid" of a fiber in vc3d_fiber_list or of a collection in
    vc3d_list_points. Only stores attached to the project are searched
    (read-only ones too); a catalog object is exported only from a store the
    project has attached. When the object is in more than one attached store,
    the call fails (-32602) with the candidate store ids: pass store to choose.

    path is an absolute destination whose parent directory must exist. A
    surface becomes a tifxyz directory, a fiber one vc3d_fiber JSON file, and
    a point collection a vc_pointcollections JSON file holding that collection.
    overwrite=true replaces an existing destination of the same type.
    valid_only (surfaces only) writes invalid vertices as missing and no
    mask.tif.

    Pending edits of the object are saved first (-32004 while an edit
    operation prevents it). The export is the stored frame: the files carry
    the object's uuid and coordinates {volume_uuid, scale_factor} (stored units
    per voxel of that volume; divide stored XYZ by scale_factor for that
    volume's voxels; unknown values stay null). Re-importing the files keeps
    the UUID and coordinates. The job's result is {path, uuid, kind, store,
    objectRevision, geometryRevision}. Export is not cancellable.
    """
    result = await _call(
        "objects.export",
        _strip_none(
            {
                "uuid": uuid,
                "path": path,
                "store": store,
                "overwrite": overwrite,
                "validOnly": valid_only,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)

