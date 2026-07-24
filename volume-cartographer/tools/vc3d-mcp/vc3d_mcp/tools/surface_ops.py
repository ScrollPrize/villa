"""Per-segment mesh operations (crop, area, reoptimize, refine, mask)."""

from __future__ import annotations

from typing import Any, Optional

from mcp.server.fastmcp import Context

from ..core import mcp, _call, _wait_for_job, _strip_none

# The bridge caps a mask render's deferred reply at 120s (AgentBridgeHandlers_
# surfaceops.cpp beginDeferred(120000)); the client's default request timeout is
# only 30s. Wait comfortably past the server cap so a slow-but-succeeding render
# resolves client-side instead of failing while the bridge is still writing.
MASK_RENDER_TIMEOUT_S = 130.0


@mcp.tool()
async def vc3d_crop_segment_bounds(segment_id: str) -> dict[str, Any]:
    """Crop a segment's surface grid to its tightest valid bounds, the headless
    twin of the "Crop to valid region" context-menu action. Synchronous: writes
    the cropped surface in place and refreshes its metrics before returning.

    segment_id: id of the segment to crop.

    Requires a loaded volume package and current volume. Returns
    {"cropped": true, "segmentId"}. Note: the underlying op does not distinguish
    "cropped" from "already at tightest bounds" -- both report cropped:true.
    """
    return await _call("segment.crop_bounds", {"segmentId": segment_id})


@mcp.tool()
async def vc3d_recalc_segment_area(segment_ids: list[str]) -> dict[str, Any]:
    """Recompute surface area for one or more segments (pure computation, no UI),
    the headless twin of the surface-panel area recalculation. Synchronous.

    segment_ids: non-empty list of segment ids to measure.

    Requires a loaded volume package and current volume. Returns
    {"results": [{"segmentId", "areaVx2", "areaCm2", "success", "errorReason"}]}.
    A bad/unknown id is reported in-band (success:false) rather than failing the
    whole call.
    """
    return await _call("segment.recalc_area", {"segmentIds": segment_ids})


@mcp.tool()
async def vc3d_reoptimize_segment(
    segment_id: str,
    volume_id: Optional[str] = None,
    omp_threads: int = 1,
    param_overrides: Optional[dict[str, Any]] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Resume-opt local reoptimization of a segment (vc_grow_seg_from_segments),
    the headless twin of the "Resume-opt Local (GrowPatch)" context-menu action.
    Asynchronous (a source:"tool" job -- poll vc3d_job_status). No dialog shown.

    segment_id: id of the segment to reoptimize.
    volume_id: vpkg volume id; default is the current volume.
    omp_threads: OMP_NUM_THREADS for the run, >= 0 (default 1; 0 => runner
    default).
    param_overrides: extra tracer-param fields merged over the fixed resume-local
    params (mirrors the dialog's JSON extra-params editor).
    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline.

    Returns {"jobId", "kind": "segment.reoptimize", "source": "tool",
    "outputDir", "volumeId"}.
    """
    result = await _call(
        "segment.reoptimize",
        _strip_none(
            {
                "segmentId": segment_id,
                "volumeId": volume_id,
                "ompThreads": omp_threads,
                "paramOverrides": param_overrides,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_refine_segment_alpha_comp(
    segment_id: str,
    refine: bool = True,
    start: float = -6.0,
    stop: float = 30.0,
    step: float = 2.0,
    low: int = 26,
    high: int = 255,
    border_off: float = 1.0,
    radius: int = 3,
    gen_vertex_color: bool = False,
    overwrite: bool = True,
    reader_scale: float = 0.5,
    scale_group: str = "1",
    omp_threads: Optional[int] = None,
    output_dir: Optional[str] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Alpha-composite refinement of a segment (vc_objrefine), the headless twin
    of the "Alpha-comp refine" context-menu action. Asynchronous (a source:"tool"
    job -- poll vc3d_job_status). No dialog is ever shown. Rejects remote volumes.

    segment_id: id of the segment to refine.
    refine: run the refine pass (default true).
    start/stop/step: alpha-sweep range (defaults -6 / 30 / 2).
    low/high: intensity clamp bounds (defaults 26 / 255).
    border_off: border offset (default 1.0).
    radius: neighborhood radius (default 3).
    gen_vertex_color: emit vertex colors (default false).
    overwrite: overwrite an existing output (default true).
    reader_scale: reader downscale factor (default 0.5).
    scale_group: OME-Zarr scale group (default "1").
    omp_threads: OMP_NUM_THREADS override; omit for the runner default.
    output_dir: absolute path, or relative to the volpkg root. Default is
    <segment>_refined.
    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline.

    Returns {"jobId", "kind": "segment.refine_alpha_comp", "source": "tool",
    "outputDir", "segmentId"}.
    """
    result = await _call(
        "segment.refine_alpha_comp",
        _strip_none(
            {
                "segmentId": segment_id,
                "refine": refine,
                "start": start,
                "stop": stop,
                "step": step,
                "low": low,
                "high": high,
                "borderOff": border_off,
                "radius": radius,
                "genVertexColor": gen_vertex_color,
                "overwrite": overwrite,
                "readerScale": reader_scale,
                "scaleGroup": scale_group,
                "ompThreads": omp_threads,
                "outputDir": output_dir,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_generate_segment_mask(segment_id: str) -> dict[str, Any]:
    """Render a segment's binary mask to <segment>/mask.tif, the headless twin of
    the "Edit mask" context-menu action. The bridge holds the reply until the
    render (a background worker) finishes, so this call resolves only on
    completion (no separate job to poll).

    segment_id: id of the segment to render a mask for.

    Requires a loaded volume package. If mask.tif already exists it is NOT
    regenerated (matching the GUI), returning {"generated": false,
    "alreadyExists": true, "maskPath", "segmentId"}. On a fresh render returns
    {"generated": true, "appended": false, "maskPath", "segmentId", "message"}.
    """
    return await _call("segment.generate_mask", {"segmentId": segment_id},
                       timeout=MASK_RENDER_TIMEOUT_S)


@mcp.tool()
async def vc3d_append_segment_mask(segment_id: str) -> dict[str, Any]:
    """Append a volume-image layer to a segment's mask.tif (creating it if
    absent), the headless twin of the "Append mask" context-menu action. Requires
    a current volume. The bridge holds the reply until the background render
    finishes, so this call resolves only on completion (no separate job to poll).

    segment_id: id of the segment whose mask to append to.

    Returns {"generated": true, "appended": true, "maskPath", "segmentId",
    "message"}.
    """
    return await _call("segment.append_mask", {"segmentId": segment_id},
                       timeout=MASK_RENDER_TIMEOUT_S)
