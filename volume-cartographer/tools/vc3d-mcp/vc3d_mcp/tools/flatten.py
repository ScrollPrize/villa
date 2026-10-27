"""Render + flatten/straighten output-producing tools."""

from __future__ import annotations

from typing import Any, Literal, Optional

from mcp.server.fastmcp import Context

from ..core import mcp, _call, _wait_for_job, _strip_none


@mcp.tool()
async def vc3d_render_tifxyz(
    segment_id: str,
    output_format: Literal["zarr", "tif_stack"],
    volume_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    scale: float = 1.0,
    group_idx: int = 0,
    num_slices: int = 1,
    voxel_size: Optional[float] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Render a segment's flattened surface with vc_render_tifxyz, the headless
    twin of the "Render" context-menu action. Asynchronous (a source:"tool" job
    -- poll vc3d_job_status).

    segment_id: id of the segment to render.
    output_format: "zarr" (OME-Zarr store) or "tif_stack" (per-slice TIFFs).
    This choice is the headline capability over the GUI, which only produces a
    TIFF stack.
    volume_id: vpkg volume id; default is the current volume.
    output_dir: absolute path, or relative to the project's directory. Default is the
    surface's artifact directory (output lands in <artifacts>/layers or
    <artifacts>/surface.zarr).
    scale: pixels per level-g voxel, > 0 (default 1.0).
    group_idx: OME-Zarr group index, >= 0 (default 0).
    num_slices: number of slices to render along the surface normal, >= 1
    (default 1).
    voxel_size: physical voxel size override in micrometers; omit to derive it
    from the volume metadata.
    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline.

    Returns {"jobId", "kind": "render.tifxyz", "source": "tool", "outputDir",
    "outputFormat", "volumeId"}. The surface is read in that volume through the
    object store (its binding or a registration). Advanced render options (crop/rotate/
    flip/in-render flatten/composite/alpha) are GUI-only and not exposed here.
    """
    result = await _call(
        "render.tifxyz",
        _strip_none(
            {
                "segmentId": segment_id,
                "outputFormat": output_format,
                "volumeId": volume_id,
                "outputDir": output_dir,
                "scale": scale,
                "groupIdx": group_idx,
                "numSlices": num_slices,
                "voxelSize": voxel_size,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_flatten_slim(
    segment_id: str,
    iterations: int = 50,
    tolerance: float = 0.0,
    energy_type: Literal["symmetric_dirichlet", "conformal"] = "symmetric_dirichlet",
    keep_percent: float = 100.0,
    inpaint_holes: bool = False,
    output_name: Optional[str] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Flatten a segment with SLIM/flatboi -- the production-recommended
    flattening method, the headless twin of the "SLIM flatten" context-menu
    action. Asynchronous (a source:"flatten" job -- poll vc3d_job_status).

    flatboi reads the stored surface and creates the flattened surface object
    (no OBJ files). No dialog is ever shown.

    segment_id: id of the segment to flatten.
    iterations: flatboi iterations, >= 1 (default 50).
    tolerance: convergence tolerance; 0 (default) runs all iterations.
    energy_type: "symmetric_dirichlet" (default) or "conformal".
    keep_percent: percentage of source grid points to keep, in (0, 100].
    Default 100 = full-resolution SLIM (no decimation), the production-
    recommended path. Below 100 decimates then lifts UVs back to full res.
    inpaint_holes: fill holes before flattening (default false).
    output_name: name of the new flattened surface. Default is <name>_flatboi.
    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline.

    Returns {"jobId", "kind": "flatten.slim", "source": "flatten", "outputName"};
    the new surface's UUID is the finished job's output.
    """
    result = await _call(
        "flatten.slim",
        _strip_none(
            {
                "segmentId": segment_id,
                "iterations": iterations,
                "tolerance": tolerance,
                "energyType": energy_type,
                "keepPercent": keep_percent,
                "inpaintHoles": inpaint_holes,
                "outputName": output_name,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_flatten_straighten(
    segment_id: str,
    unbend: bool = True,
    unbend_smooth_cols: float = 300.0,
    overlap_passes: int = 2,
    orthogonalize: bool = True,
    trim: bool = True,
    trim_max_edge: float = 100.0,
    output_name: Optional[str] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Straighten a segment with vc_straighten -- the headless twin of the
    "Straighten" context-menu action. Asynchronous (a source:"flatten" job --
    poll vc3d_job_status). No dialog is ever shown.

    segment_id: id of the segment to straighten.
    unbend: run the unbend (spine-straightening) stage (default true).
    unbend_smooth_cols: spine Gaussian sigma in columns (default 300); only
    used when unbend is true.
    overlap_passes: number of --overlap-pairs passes (default 2).
    orthogonalize: run the orthogonalize stage (default true).
    trim: run the trim stage (default true).
    trim_max_edge: max edge length for the trim stage (default 100); only used
    when trim is true.
    output_name: name of the new straightened surface. Default is
    <name>_straightened.
    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline.

    Returns {"jobId", "kind": "flatten.straighten", "source": "flatten",
    "outputName"}; the new surface's UUID is the finished job's output.
    """
    result = await _call(
        "flatten.straighten",
        _strip_none(
            {
                "segmentId": segment_id,
                "unbend": unbend,
                "unbendSmoothCols": unbend_smooth_cols,
                "overlapPasses": overlap_passes,
                "orthogonalize": orthogonalize,
                "trim": trim,
                "trimMaxEdge": trim_max_edge,
                "outputName": output_name,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)
