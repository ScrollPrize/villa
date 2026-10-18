"""Segment file operations: grid transforms, exports, neighbor copy, merge and
patch, backup restore, and moving traces to paths."""

from __future__ import annotations

from typing import Any, Literal, Optional

from mcp.server.fastmcp import Context

from ..core import mcp, _call, _strip_none, _wait_for_job

# Synchronous file operations save a whole segment before the bridge replies.
SEGMENT_FILE_OP_TIMEOUT_S = 120.0


@mcp.tool()
async def vc3d_transform_segment(
    segment_id: str,
    op: Literal["flip_u", "flip_v", "rotate90"],
    confirm: bool = False,
) -> dict[str, Any]:
    """Flip or rotate a segment's surface grid in place, the headless twin of the
    segment context menu's "Flip Surface" and "Rotate Surface 90° CW" entries.
    Synchronous: the transformed surface is saved before the call returns.

    segment_id: id of the segment to transform.
    op: "flip_u" reverses the rows (the menu's "V Flip"); "flip_v" reverses
    the columns (the menu's "H Flip", which also reverses the surface normals,
    so it is the menu's "Normals (H Flip)" too); "rotate90" rotates the grid by
    90 degrees clockwise.
    confirm: required (true) for "rotate90", which resamples the grid, so four
    rotations do not give back the original. Flips are exact and their own
    inverse and need no confirm.

    Not idempotent: a second flip undoes the first. If a call times out, read
    the segment before retrying. Requires a loaded volume package and current
    volume. Refused (-32004) for the segment being edited while segmentation
    editing is on, while a mask render runs, and while a source:"tool" job
    runs. Returns {"transformed": true, "segmentId", "op"}.
    """
    return await _call(
        "segment.transform",
        {"segmentId": segment_id, "op": op, "confirm": confirm},
        timeout=SEGMENT_FILE_OP_TIMEOUT_S,
    )


@mcp.tool()
async def vc3d_export_segment_obj(
    segment_id: str,
    output_path: Optional[str] = None,
    overwrite: bool = False,
    normalize_uv: bool = False,
    align_grid: bool = False,
    omp_threads: Optional[int] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Convert a segment's tifxyz grid to an OBJ mesh, the headless twin of the
    segment context menu's "Convert to OBJ". Runs vc_tifxyz2obj as a
    source:"tool" job; no dialog is shown.

    segment_id: the segment to convert.
    output_path: absolute path of the OBJ file to write, ending in ".obj"
    (default: <segment folder>/<segment id>.obj). Missing parent folders are
    created.
    overwrite: replace an existing file at that path. Without it an existing
    file is refused (-32005), so a repeated export needs overwrite=true.
    normalize_uv: scale UVs to [0,1]. align_grid: flatten Z per row.
    omp_threads: threads for the converter (>= 1; default: the runner's).
    wait: block until the job finishes (30-minute cap) and return the
    terminal job status inline; its outputPath is the OBJ file.

    Requires a current volume. Fails with -32006 when vc_tifxyz2obj is not
    installed beside VC3D and -32004 while another tool job runs. Returns
    {"jobId", "kind": "segment.export_obj", "source": "tool", "segmentId",
    "outputPath"}.
    """
    result = await _call(
        "segment.export_obj",
        _strip_none(
            {
                "segmentId": segment_id,
                "outputPath": output_path,
                "overwrite": overwrite,
                "normalizeUv": normalize_uv,
                "alignGrid": align_grid,
                "ompThreads": omp_threads,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_rasterize_segments(
    segment_ids: list[str],
    chunk_size: Optional[int] = None,
    z_min: Optional[int] = None,
    z_max: Optional[int] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Rasterize segments into a labels zarr, the headless twin of the segment
    context menu's "Rasterize". Runs vc_tifxyz2zarr_sparse as a source:"tool"
    job that ends once the zarr is moved into the project's volumes/ folder and
    its metadata written; no dialog is shown.

    segment_ids: one or more segments. Ids without tifxyz meshes are skipped
    and listed in "skippedIds"; at least one must remain.
    chunk_size: isotropic chunk size, 1 to 4096 (default 128).
    z_min: first level-0 slice, >= 0 (default 0).
    z_max: end slice (exclusive), or -1 for the volume end (default -1);
    must be greater than z_min unless -1.
    wait: block until the job finishes (30-minute cap) and return the
    terminal job status inline; its outputPath is the final zarr.

    The reference grid is the normal3d zarr chosen in the Segmentation panel,
    else the current volume. Returns {"jobId", "kind": "segment.rasterize",
    "source": "tool", "segmentIds", "skippedIds", "outputPath"}.
    """
    result = await _call(
        "segment.rasterize",
        _strip_none(
            {
                "segmentIds": segment_ids,
                "chunkSize": chunk_size,
                "zMin": z_min,
                "zMax": z_max,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)
