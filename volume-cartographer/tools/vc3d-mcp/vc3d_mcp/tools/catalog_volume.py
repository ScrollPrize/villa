"""Volume package + Open Data catalog tools."""

from __future__ import annotations

from typing import Any, Literal, Optional

from mcp.server.fastmcp import Context
from typing_extensions import TypedDict

from ..core import mcp, _call, _wait_for_job, _strip_none


class _CatalogResources(TypedDict, total=False):
    volumeIds: list[str]
    representationRefs: list[str]
    kinds: list[Literal["normal_grids", "lasagna", "prediction"]]


@mcp.tool()
async def vc3d_create_project(
    path: str,
    volume: str,
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a .volpkg.json project that references one local zarr volume or
    remote .zarr URL.
    This writes the project but does not open it; call vc3d_open_volume with the
    returned path when it should become the active project. Remote availability
    is checked when the project is opened.

    path must be an absolute output path on the filesystem where VC3D runs.
    volume must be an absolute local zarr path on that same filesystem, or a
    remote zarr URL. In the containerized development setup, use paths visible
    inside the container (typically /work/...), not host-only paths. Missing
    parent directories and the .volpkg.json suffix are added automatically.
    name and tags customize the project entry; overwrite must be true to
    replace an existing output file.
    """
    return await _call(
        "project.create",
        _strip_none(
            {
                "path": path,
                "volume": volume,
                "name": name,
                "tags": tags,
                "overwrite": overwrite,
            }
        ),
    )


@mcp.tool()
async def vc3d_open_volume(path: str, volume_id: Optional[str] = None) -> dict[str, Any]:
    """Open a volume package (.volpkg / .volpkg.json / zarr project) and
    optionally select a volume id."""
    return await _call("volume.open", _strip_none({"path": path, "volumeId": volume_id}))


@mcp.tool()
async def vc3d_open_catalog_sample(
    sample_id: str,
    resources: Optional[_CatalogResources] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Open an Open Data catalog sample by its manifest sample id. Async: a
    remote open is a multi-second-to-multi-minute network operation, so this
    returns a jobId immediately.

    resources: optional resource-selection filter to attach only a subset.
    Omit it to attach everything. Shape:
    {"volumeIds": [str],            # subset of the sample's volume ids
     "representationRefs": [str],   # "vi:ai" refs from vc3d_describe_catalog_sample
     "kinds": [str]}                # subset of "normal_grids"|"lasagna"|"prediction"
    An absent sub-field means no filter on that axis. A raw source volume is
    attached iff volumeIds is absent or lists its id; a derived representation
    must pass all provided axes.

    Returns {jobId, kind, source:"catalog", sampleId}. Poll job.status (or pass
    wait=true) for the terminal record; its "result" carries the opened project
    plus an "attached" block (volumes/segments/normalGrids/lasagnaDatasets),
    "vpkgPath", "volumeIds", and "messages".

    wait: if true (MCP-server-side only), block until the job finishes (30-minute
    cap) and return the terminal job.status inline.
    """
    result = await _call(
        "catalog.open_sample", _strip_none({"sampleId": sample_id, "resources": resources})
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_list_catalog_samples(refresh: bool = False) -> dict[str, Any]:
    """List Open Data catalog samples from the manifest (id, type, description,
    volume/segment/scan counts).

    refresh: force a fresh manifest fetch (up to 30 s) instead of serving the
    cached copy; also fetches automatically when nothing is cached yet.
    """
    return await _call("catalog.list_samples", {"refresh": refresh})


@mcp.tool()
async def vc3d_describe_catalog_sample(
    sample_id: str, refresh: bool = False
) -> dict[str, Any]:
    """Describe one Open Data catalog sample: its volumes (id, scanId, shape,
    pixel size, data format) and derived representations categorized by kind
    (normal_grids / lasagna / prediction), each with a stable "ref" ("vi:ai")
    usable in vc3d_open_catalog_sample's resources.representationRefs.

    refresh: force a fresh manifest fetch (up to 30 s) before describing.
    """
    return await _call(
        "catalog.describe_sample", {"sampleId": sample_id, "refresh": refresh}
    )


@mcp.tool()
async def vc3d_list_volumes() -> dict[str, Any]:
    """List the volumes attached to the open volume package, with the currently
    selected one.

    Use this to discover the volume ids you can pass to vc3d_select_volume (or
    other volume-scoped tools) without scraping vc3d_get_state. Requires a
    volume package to be open.

    Returns {"volumeIds": [str...], "currentVolumeId": str|null}, and may
    include a "volumes" array of {id, path, voxelSize} objects when that detail
    is cheap to gather. Errors: -32000 (no volume package loaded)."""
    return await _call("volume.list")


@mcp.tool()
async def vc3d_select_volume(volume_id: str) -> dict[str, Any]:
    """Switch the current volume among the already-attached volumes of the open
    package (the programmatic equivalent of picking one in the volume combo).
    Selecting the already-current volume is a no-op success. Returns
    {"volumeId", "previousVolumeId"}."""
    return await _call("volume.select", {"volumeId": volume_id})
