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


@mcp.tool()
async def vc3d_attach_store(path: str, read_only: bool = False) -> dict[str, Any]:
    """Attach an existing local .vcobj store to the open project (the Stores
    menu's Attach Store), recording it in the .vcproj. Attaching does not
    change routing: call vc3d_route_store to show or edit a kind's objects
    from it, or pass its id as `store` to vc3d_import_objects.

    path is an absolute path on the filesystem where VC3D runs. read_only
    attaches it for reading only. A store that is already attached returns its
    existing reference unchanged (read_only is then ignored). Returns
    {storeId, readOnly, path}: readOnly is the actual mode (a store this user
    cannot write is read-only).
    """
    return await _call("stores.attach", {"path": path, "readOnly": read_only})


@mcp.tool()
async def vc3d_create_store(path: str) -> dict[str, Any]:
    """Create a new, empty .vcobj store at `path` (absolute; .vcobj is added
    when it has no extension) and attach it to the open project. Refuses an
    existing file; the new file is removed again when it cannot be attached.
    Does not change routing (see vc3d_route_store). Returns {storeId,
    readOnly, path}.
    """
    return await _call("stores.create", {"path": path})


@mcp.tool()
async def vc3d_detach_store(store_id: str) -> dict[str, Any]:
    """Remove a store reference (an id from vc3d_get_state's "stores") from the
    open project, including one whose file is unavailable. The store file and
    its objects are untouched; kinds routed to it are then routed to no store.
    Pending edits (point collections, the open segment, idle fiber sessions)
    are saved first: -32004 when they cannot be, while growth or a line
    optimization runs, or while an import or export job uses the store.
    The last store of a project cannot be detached (-32005). Returns
    {storeId, routes}.
    """
    return await _call("stores.detach", {"storeId": store_id})


@mcp.tool()
async def vc3d_route_store(
    kind: Literal["surface", "fiber", "point_collection", "umbilicus", "volume_transform"],
    store_id: Optional[str],
) -> dict[str, Any]:
    """Choose the store the open project shows and edits one kind of object
    from, and where new objects of that kind go (the Stores menu's "Show ...
    From" menus). store_id is a reference id from vc3d_get_state's "stores";
    null routes no store, which stops using that kind without detaching
    anything. A read-only store can be routed for viewing.

    Edits of existing objects always go to the store that holds them; routing
    does not move objects. Pending edits are saved first (-32004 when they
    cannot be, or while growth or a line optimization runs); a refused route
    leaves the previous one. Persists in the .vcproj. Returns {kind, storeId,
    routes}.
    """
    return await _call("stores.route", {"kind": kind, "storeId": store_id})


@mcp.tool()
async def vc3d_save_project_as(path: str) -> dict[str, Any]:
    """Save the open project to a new .vcproj at `path` (absolute; .vcproj is
    added when it has no extension) and reopen it from there, keeping the
    current volume. Resource locations are rebased; stores are shared, not
    copied. Refuses an existing file (-32005). Pending edits are saved first
    (-32004 when they cannot be, or while growth or a line optimization runs).
    Returns {path, previousPath}.
    """
    return await _call("project.save_as", {"path": path})


@mcp.tool()
async def vc3d_attach_normal_grid(location: str) -> dict[str, Any]:
    """Attach a normal_grids root (an absolute local directory with xy/, xz/
    and yz/ subdirectories and metadata.json, or a folder of such roots) to
    the open project. Attaching the same location
    again is a success with alreadyAttached=true. An invalid location fails
    -32007. Remove it with vc3d_detach_project_entry. Returns {location,
    attached, alreadyAttached}.
    """
    return await _call("project.attach_normal_grid", {"location": location})


@mcp.tool()
async def vc3d_attach_lasagna_manifest(
    location: str,
    role: Literal["regular", "fiber"],
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Attach a Lasagna dataset by its .lasagna.json manifest (an absolute
    local path, or an s3://, http(s):// or sftp:// URL) to the open project.

    role is required and never inferred from the location: "regular" Lasagna
    data needs grad_mag, nx and ny normal channels; "fiber" attaches it as
    fiber inference (prediction traces). Returns a "manifest" job; loading
    and validating the manifest reads the network for remote data. pass
    wait=true to get its terminal record, whose result is {location, role,
    attached, alreadyAttached, volumeIds}. If another project is opened
    before it finishes, the job fails without attaching anything. Not
    cancellable. Remove it with vc3d_detach_project_entry.
    """
    result = await _call(
        "project.attach_lasagna_manifest", {"location": location, "role": role}
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_detach_project_entry(location: str) -> dict[str, Any]:
    """Remove a volume, normal grid or Lasagna dataset entry from the open
    project (the File menu's Detach), without a confirmation step. location is
    the exact "location" of an entry in vc3d_get_state's project.entries;
    -32007 when there is none. The source data is untouched. Detaching the
    current volume selects another one (or none). Pending edits are saved first
    (-32004 when they cannot be, or while growth or a line optimization runs).
    Returns {location, scope}: "everyone" (removed for everyone using the
    project) or "only you".
    """
    return await _call("project.detach_entry", {"location": location})


@mcp.tool()
async def vc3d_import_umbilicus(
    path: str,
    units_per_voxel: Optional[float] = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Import an umbilicus file (.json, .txt or .csv; absolute path) as the
    umbilicus of the current volume, in the store routed for umbilici. Needs a
    selected volume (-32001) and a writable umbilicus store (-32009).

    units_per_voxel is the file's units per current-volume voxel. When the
    file states its frame it is used and units_per_voxel may be omitted (a
    different value fails -32602). When the frame is only inferred from the
    points, or not stated at all, the call fails -32602 with the inference in
    error data {inferred, frame}: check it and call again with
    units_per_voxel. A file that does not fit the volume fails -32005. When
    the volume already has an umbilicus the call fails -32005 unless
    replace=true. Returns {uuid, volume, store, unitsPerVoxel, frame,
    replaced}; vc3d_get_state's "umbilicus" lists the current one.
    """
    return await _call(
        "umbilicus.import",
        _strip_none(
            {"path": path, "unitsPerVoxel": units_per_voxel, "replace": replace}
        ),
    )


@mcp.tool()
async def vc3d_remove_umbilicus() -> dict[str, Any]:
    """Remove the current volume's umbilicus by retiring its objects in the
    store routed for umbilici. This changes the store: other projects sharing
    that store see the removal too. To stop using umbilici in this project
    only, call vc3d_route_store(kind="umbilicus", store_id=None) instead.
    No umbilicus is a success with removed=[]; a read-only umbilicus store
    fails -32009. Returns {volume, store, removed}.
    """
    return await _call("umbilicus.remove")
