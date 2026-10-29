"""Segment operations on the object store: grid transforms, exports,
rasterization, neighbor copy, merge and patch."""

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
    the segment before retrying. Requires an open project and current
    volume. Refused (-32004) for the segment being edited while segmentation
    editing is on, and while a source:"tool" job runs. Returns {"transformed": true, "segmentId", "op"}.
    """
    return await _call(
        "segment.transform",
        {"segmentId": segment_id, "op": op, "confirm": confirm},
        timeout=SEGMENT_FILE_OP_TIMEOUT_S,
    )


@mcp.tool()
async def vc3d_export_segment_obj(
    segment_id: str,
    obj_name: Optional[str] = None,
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
    obj_name: file name of the OBJ, ending in ".obj" and without folders. The
    tool writes it into the surface's artifact folder in the object store
    (default name: the converter's).
    normalize_uv: scale UVs to [0,1]. align_grid: flatten Z per row.
    omp_threads: threads for the converter (>= 1; default: the runner's).
    wait: block until the job finishes (30-minute cap) and return the
    terminal job status inline; its outputPath is the OBJ file.

    Requires a current volume. Fails with -32006 when vc_tifxyz2obj is not
    installed beside VC3D and -32004 while another tool job runs. Returns
    {"jobId", "kind": "segment.export_obj", "source": "tool", "segmentId"}.
    """
    result = await _call(
        "segment.export_obj",
        _strip_none(
            {
                "segmentId": segment_id,
                "objName": obj_name,
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
    job; the tool publishes the labels zarr as a volume of the project's store.
    No dialog is shown.

    segment_ids: one or more segments. Ids without tifxyz meshes are skipped
    and listed in "skippedIds"; at least one must remain.
    chunk_size: isotropic chunk size, 1 to 4096 (default 128).
    z_min: first level-0 slice, >= 0 (default 0).
    z_max: end slice (exclusive), or -1 for the volume end (default -1);
    must be greater than z_min unless -1.
    wait: block until the job finishes (30-minute cap) and return the
    terminal job status inline; its outputPath is the final zarr.

    The reference grid is the current volume. Returns {"jobId", "kind": "segment.rasterize",
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


@mcp.tool()
async def vc3d_neighbor_copy_segment(
    segment_id: str,
    direction: Literal["out", "in"],
    volume_id: Optional[str] = None,
    output_tag: Optional[str] = None,
    neighbor_max_distance: Optional[int] = None,
    neighbor_min_clearance: Optional[int] = None,
    neighbor_fill: Optional[bool] = None,
    neighbor_interp_window: Optional[int] = None,
    generations: Optional[int] = None,
    neighbor_spike_window: Optional[int] = None,
    resume_local_opt_step: Optional[int] = None,
    resume_local_opt_radius: Optional[int] = None,
    resume_local_max_iters: Optional[int] = None,
    resume_local_dense_qr: Optional[bool] = None,
    refine_omp_threads: Optional[int] = None,
    refine_param_overrides: Optional[dict[str, Any]] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Grow a new segment on the neighboring sheet, the headless twin of the
    segment context menu's "Copy Surface > Out / In". Runs one
    vc_grow_seg_from_seed run as a source:"tool" job that generates the
    neighbor surface and then locally reoptimizes it. No dialog is shown.

    segment_id: the surface to copy from.
    direction: "out" or "in" (the neighboring wrap on that side).
    volume_id: volume to trace in; default the current volume.
    output_tag: tag given to the new surface in the store (default "paths").
    Neighbor generation (defaults as in the dialog): neighbor_max_distance 1-500 (200),
    neighbor_min_clearance 1-100 (4), neighbor_fill (true),
    neighbor_interp_window 1-50 (5), generations 1-10 (2),
    neighbor_spike_window 1-20 (2).
    Refinement: resume_local_opt_step 1-512 (20), resume_local_opt_radius
    1-2048 (40), resume_local_max_iters 1-10000 (1000), resume_local_dense_qr
    (false), refine_omp_threads 1-256 (1), refine_param_overrides (tracer
    params merged over the refinement's).
    wait: block until the run finishes (30-minute cap) and return the
    terminal job status inline. vc3d_cancel_job stops it.

    Returns {"jobId", "kind": "segment.neighbor_copy", "source": "tool",
    "segmentId", "direction", "outputTag"}.
    """
    result = await _call(
        "segment.neighbor_copy",
        _strip_none(
            {
                "segmentId": segment_id,
                "direction": direction,
                "volumeId": volume_id,
                "outputTag": output_tag,
                "neighborMaxDistance": neighbor_max_distance,
                "neighborMinClearance": neighbor_min_clearance,
                "neighborFill": neighbor_fill,
                "neighborInterpWindow": neighbor_interp_window,
                "generations": generations,
                "neighborSpikeWindow": neighbor_spike_window,
                "resumeLocalOptStep": resume_local_opt_step,
                "resumeLocalOptRadius": resume_local_opt_radius,
                "resumeLocalMaxIters": resume_local_max_iters,
                "resumeLocalDenseQr": resume_local_dense_qr,
                "refineOmpThreads": refine_omp_threads,
                "refineParamOverrides": refine_param_overrides,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_merge_segments(
    segment_ids: Optional[list[str]] = None,
    grid: Optional[list[list[Optional[str]]]] = None,
    ref_segment_id: Optional[str] = None,
    output_tag: Optional[str] = None,
    strip_cols: Optional[int] = None,
    ransac_iters: Optional[int] = None,
    ransac_min_thresh: Optional[float] = None,
    ransac_max_thresh: Optional[float] = None,
    ransac_mad_k: Optional[float] = None,
    ransac_seed: Optional[int] = None,
    anchor_cap: Optional[int] = None,
    omp_threads: Optional[int] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Merge tifxyz segments into a new segment, the headless twin of "Merge
    tifxyz..." (Actions menu and segment context menu). Runs vc_merge_tifxyz
    as a source:"tool" job; no dialog is shown.

    Pass exactly one of:
      grid: rows of segment ids laid out as they overlap, null for a hole;
      segment_ids: at least two ids, merged as a single row.
    At least two distinct segments are required; names with whitespace and a
    reference outside the grid are rejected (-32602); unknown ids are -32007.
    ref_segment_id: the reference surface (default: the one with the most
    valid cells). output_tag: tag given to the merged surface in the store
    (default "paths"). strip_cols: 0-1024 (0).
    RANSAC options (dialog defaults): ransac_iters (3000),
    ransac_min_thresh (5.0) < ransac_max_thresh (10.0), ransac_mad_k (3.0),
    ransac_seed (0), anchor_cap (0), omp_threads (runner default).
    wait: block until the job finishes (30-minute cap) and return the
    terminal job status inline. The merged surface is a new surface in the
    store.

    Returns {"jobId", "kind": "segments.merge", "source": "tool",
    "segmentIds" (distinct ids in grid order), "outputTag"}.
    """
    result = await _call(
        "segments.merge",
        _strip_none(
            {
                "segmentIds": segment_ids,
                "grid": grid,
                "refSegmentId": ref_segment_id,
                "outputTag": output_tag,
                "stripCols": strip_cols,
                "ransacIters": ransac_iters,
                "ransacMinThresh": ransac_min_thresh,
                "ransacMaxThresh": ransac_max_thresh,
                "ransacMadK": ransac_mad_k,
                "ransacSeed": ransac_seed,
                "anchorCap": anchor_cap,
                "ompThreads": omp_threads,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_patch_segment(
    segment_ids: list[str],
    parent_id: Optional[str] = None,
    border_cells: Optional[int] = None,
    blend_cells: Optional[int] = None,
    idw_k: Optional[int] = None,
    confirm: bool = False,
    ransac_iters: Optional[int] = None,
    ransac_min_thresh: Optional[float] = None,
    ransac_max_thresh: Optional[float] = None,
    ransac_mad_k: Optional[float] = None,
    ransac_seed: Optional[int] = None,
    anchor_cap: Optional[int] = None,
    omp_threads: Optional[int] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Patch one segment into another in place, the headless twin of "Patch
    tifxyz..." (Actions menu and segment context menu). Runs vc_merge_patch
    as a source:"tool" job; no dialog is shown.

    segment_ids: exactly two different segments.
    parent_id: which of them gets the patched grid as a new revision;
    omitted, the tool picks the one with more valid cells.
    confirm: must be true; patching replaces the parent's current grid.
    border_cells 1-256 (16), blend_cells 1-256 (6), idw_k 1-16 (4).
    RANSAC options as in vc3d_merge_segments; ransac_min_thresh must stay
    below ransac_max_thresh (-32602).
    wait: block until the job finishes (30-minute cap) and return the
    terminal job status inline.

    Refused (-32004) while either segment is being edited or another tool job
    runs. Returns {"jobId", "kind":
    "segments.patch", "source": "tool", "segmentIds", "parentId" (null when
    the tool picks)}.
    """
    result = await _call(
        "segments.patch",
        _strip_none(
            {
                "segmentIds": segment_ids,
                "parentId": parent_id,
                "borderCells": border_cells,
                "blendCells": blend_cells,
                "idwK": idw_k,
                "confirm": confirm,
                "ransacIters": ransac_iters,
                "ransacMinThresh": ransac_min_thresh,
                "ransacMaxThresh": ransac_max_thresh,
                "ransacMadK": ransac_mad_k,
                "ransacSeed": ransac_seed,
                "anchorCap": anchor_cap,
                "ompThreads": omp_threads,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)

