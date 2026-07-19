"""MCP server exposing the VC3D Agent Bridge RPCs as MCP tools.

One MCP tool per JSON-RPC method in apps/VC3D/agent_bridge/SPEC.md section 3,
per the tool surface table in SPEC.md section 5. Uses the official MCP Python
SDK (`mcp` package, `FastMCP`) over stdio -- see README.md "Implementation
notes" for why (the SDK installs cleanly in this environment, so we didn't
need to hand-roll the protocol).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from .bridge_client import BridgeClient, BridgeClientConfig, BridgeConnectionError, BridgeError

DEFAULT_WAIT_TIMEOUT_S = 30 * 60  # SPEC.md section 5: "the MCP server enforces a 30-minute cap"

mcp = FastMCP(
    name="vc3d-agent-bridge",
    instructions=(
        "Tools that drive a running VC3D instance through its Agent Bridge. "
        "Call vc3d_get_state first to see what's loaded (volume package, "
        "current volume, active segment, viewers, running job) before doing "
        "anything else."
    ),
)

# Set by main()/configure_client() before the stdio loop starts. A module
# global (rather than a class) because FastMCP tool functions are registered
# as free functions via the `@mcp.tool()` decorator.
_client: BridgeClient | None = None


def configure_client(socket: str, request_timeout: float = 30.0) -> BridgeClient:
    global _client
    _client = BridgeClient(BridgeClientConfig(socket=socket, request_timeout=request_timeout))
    return _client


def _get_client() -> BridgeClient:
    if _client is None:
        raise RuntimeError(
            "vc3d-mcp server was not configured with a bridge socket "
            "(configure_client() was never called before serving requests)"
        )
    return _client


async def _call(method: str, params: dict[str, Any] | None = None) -> Any:
    """Call a bridge RPC, translating transport failures into a clear message.

    JSON-RPC error replies (BridgeError) are left to propagate as-is -- its
    `__str__` is the `{"code", "message", "data"}` JSON object per SPEC.md
    section 2.5, which is what ends up in the MCP tool's error text, per the
    "RPC errors surface as MCP tool errors with code/message/data preserved"
    requirement in SPEC.md section 5.
    """
    client = _get_client()
    try:
        return await client.call(method, params)
    except BridgeConnectionError as exc:
        raise RuntimeError(str(exc)) from exc


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


async def _wait_for_job(job_id: str, wait: bool, initial_result: dict[str, Any]) -> dict[str, Any]:
    """Shared `wait: true` handling for the async grow.* tools.

    SPEC.md section 5: "forwards phase:'output' messages as tool progress
    ... when the caller opted into 'wait': true ... blocks the tool call
    until the job's finished notification and returns the terminal status
    inline. wait defaults to false; when true, the MCP server enforces a
    30-minute cap and returns the still-running jobId on timeout."
    """
    if not wait:
        return initial_result

    client = _get_client()
    record = await client.jobs.wait_finished(job_id, timeout=DEFAULT_WAIT_TIMEOUT_S)
    if record is None:
        return {**initial_result, "waitTimedOut": True}

    # job.status is the bridge's authoritative terminal record (SPEC.md 3.17);
    # prefer it over our locally-tracked notification data, falling back to
    # the local record if the RPC itself races with process shutdown.
    try:
        status = await _call("job.status", {"jobId": job_id})
    except Exception:
        status = record.as_dict()
    return {**initial_result, **status}


# ---------------------------------------------------------------------------
# Tools -- one per SPEC.md section 5 table row, same order.
# ---------------------------------------------------------------------------


@mcp.tool()
async def vc3d_ping() -> dict[str, Any]:
    """Check the VC3D bridge is alive; returns pid and app version."""
    return await _call("ping")


@mcp.tool()
async def vc3d_get_state() -> dict[str, Any]:
    """Snapshot of VC3D: open volume package, current volume, active segment,
    viewers (ids/names), editing mode, running job. Call this first."""
    return await _call("state.get")


@mcp.tool()
async def vc3d_list_segments(only_loaded: bool = False) -> dict[str, Any]:
    """List segments in the open volume package with loaded/active flags."""
    return await _call("segments.list", {"onlyLoaded": only_loaded})


@mcp.tool()
async def vc3d_screenshot(
    target: str = "window",
    file_path: Optional[str] = None,
    max_dim: Optional[int] = None,
) -> dict[str, Any]:
    """Capture a PNG of the whole VC3D window or one viewer pane. Returns
    base64 or writes to file_path.

    target: "window" for the whole app, or a viewer ref (a "vN" registry id
    or a surface-slot name like "segmentation"/"xy plane"/"seg xz"/"seg yz").
    file_path: absolute path; when set, the PNG is written to disk and the
    result's base64 field is null. Omit to get the PNG back as base64.
    max_dim: optional downscale, longest side in pixels, aspect preserved.
    """
    return await _call(
        "screenshot.capture",
        _strip_none({"target": target, "filePath": file_path, "maxDim": max_dim}),
    )


@mcp.tool()
async def vc3d_get_cursor_point(
    viewer: Optional[str] = None, scene: Optional[dict[str, float]] = None
) -> dict[str, Any]:
    """Resolve a viewer scene position (or the current cursor) to a 3D volume
    point + surface normal.

    viewer: viewer id ("v1") or surface-slot name; default "segmentation".
    scene: {"x", "y"} scene-space position; omit to use the viewer's last
    known cursor position.
    """
    return await _call(
        "canvas.get_cursor_volume_point", _strip_none({"viewer": viewer, "scene": scene})
    )


@mcp.tool()
async def vc3d_click(
    position: dict[str, float],
    viewer: Optional[str] = None,
    space: str = "volume",
    button: str = "left",
    modifiers: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Synthesize a mouse click in a viewer at a volume-space (or scene-space)
    position, with button and modifiers (e.g. modifiers=["shift"] to place a
    point / set focus).

    position: {"x","y","z"} in volume space (default), or {"x","y"} in scene
    space when space="scene".
    button: "left" | "right" | "middle".
    modifiers: any of "shift", "ctrl", "alt", "meta", "keypad".
    """
    return await _call(
        "canvas.click",
        _strip_none(
            {
                "viewer": viewer,
                "position": position,
                "space": space,
                "button": button,
                "modifiers": modifiers if modifiers is not None else [],
            }
        ),
    )


@mcp.tool()
async def vc3d_shift_click(
    position: dict[str, float],
    viewer: Optional[str] = None,
    space: str = "volume",
    button: str = "left",
    modifiers: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Shift+click convenience: the canonical place-point / set-focus gesture.
    Identical to vc3d_click with "shift" unioned into modifiers."""
    return await _call(
        "canvas.shift_click",
        _strip_none(
            {
                "viewer": viewer,
                "position": position,
                "space": space,
                "button": button,
                "modifiers": modifiers if modifiers is not None else [],
            }
        ),
    )


@mcp.tool()
async def vc3d_center_viewer(
    point: dict[str, float], viewer: Optional[str] = None, force_render: bool = True
) -> dict[str, Any]:
    """Center a viewer pane on a 3D volume point."""
    return await _call(
        "viewer.center_on_point",
        _strip_none({"viewer": viewer, "point": point, "forceRender": force_render}),
    )


@mcp.tool()
async def vc3d_zoom_viewer(factor: float, viewer: Optional[str] = None) -> dict[str, Any]:
    """Multiply a viewer's zoom by a factor (>1 zooms in). Returns the new
    scale."""
    return await _call("viewer.zoom", _strip_none({"viewer": viewer, "factor": factor}))


@mcp.tool()
async def vc3d_enable_editing(enabled: bool) -> dict[str, Any]:
    """Turn segmentation editing mode on/off for the active segment."""
    return await _call("segmentation.enable_editing", {"enabled": enabled})


@mcp.tool()
async def vc3d_grow_segment(
    steps: int,
    method: str = "tracer",
    direction: str = "all",
    inpaint_only: bool = False,
    wait: bool = False,
) -> dict[str, Any]:
    """Grow the active segmentation surface. Async: returns a jobId.

    method: "tracer" | "corrections" | "patch_tracer" | "manual_add".
    direction: "all" | "up" | "down" | "left" | "right" | "fill".
    steps: number of growth steps, >= 1.
    wait: if true (MCP-server-side only, not part of the underlying RPC),
    block until the job finishes (30-minute cap) and return the terminal
    job.status inline instead of just the jobId.
    """
    result = await _call(
        "segmentation.grow",
        {
            "method": method,
            "direction": direction,
            "steps": steps,
            "inpaintOnly": inpaint_only,
        },
    )
    return await _wait_for_job(result["jobId"], wait, result)


@mcp.tool()
async def vc3d_grow_patch_from_seed(
    seed: dict[str, float],
    volume_id: Optional[str] = None,
    iterations: int = 200,
    min_area_cm: float = 0.002,
    output_dir: Optional[str] = None,
    wait: bool = False,
) -> dict[str, Any]:
    """Create a brand-new segment by growing a patch from a 3D seed point
    (headless GrowPatch). Async: returns a jobId and outputDir.

    seed: volume-space {"x","y","z"} seed point.
    volume_id: vpkg volume id; default is the current volume.
    iterations: 1..100000, default 200 ("generations" in the underlying tool).
    min_area_cm: minimum patch area in cm^2, >= 0.
    output_dir: absolute path, or relative to the volpkg root.
    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline.
    """
    result = await _call(
        "segmentation.grow_patch_from_seed",
        _strip_none(
            {
                "seed": seed,
                "volumeId": volume_id,
                "iterations": iterations,
                "minAreaCm": min_area_cm,
                "outputDir": output_dir,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result)


@mcp.tool()
async def vc3d_commit_points(
    collection: str, points: list[dict[str, float]], winding: Optional[float] = None
) -> dict[str, Any]:
    """Add annotation points (volume space) to a named collection, optionally
    with a winding annotation. The collection is created if absent."""
    return await _call(
        "points.commit", _strip_none({"collection": collection, "points": points, "winding": winding})
    )


@mcp.tool()
async def vc3d_list_points(collection: Optional[str] = None) -> dict[str, Any]:
    """List point collections and their points. Omit collection to list all
    collections."""
    return await _call("points.list", _strip_none({"collection": collection}))


@mcp.tool()
async def vc3d_open_volume(path: str, volume_id: Optional[str] = None) -> dict[str, Any]:
    """Open a volume package (.volpkg / .volpkg.json / zarr project) and
    optionally select a volume id."""
    return await _call("volume.open", _strip_none({"path": path, "volumeId": volume_id}))


@mcp.tool()
async def vc3d_open_catalog_sample(sample_id: str) -> dict[str, Any]:
    """Open an Open Data catalog sample by its manifest sample id."""
    return await _call("catalog.open_sample", {"sampleId": sample_id})


@mcp.tool()
async def vc3d_job_status(job_id: Optional[str] = None) -> dict[str, Any]:
    """Poll a job by id (or the latest job): state, message, console tail."""
    return await _call("job.status", _strip_none({"jobId": job_id}))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vc3d-mcp",
        description=(
            "MCP server (stdio) that drives a running VC3D instance via its "
            "Agent Bridge local socket."
        ),
    )
    parser.add_argument(
        "--socket",
        default=os.environ.get("VC3D_AGENT_BRIDGE_SOCKET"),
        help=(
            "VC3D agent bridge socket: an explicit path (recommended, e.g. "
            "the 'path=' field from the VC3D-AGENT-BRIDGE stdout handshake "
            "line), or a bare QLocalServer name matching --agent-bridge-name. "
            "Defaults to the VC3D_AGENT_BRIDGE_SOCKET env var."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for a bridge RPC response before failing the tool call (default: 30).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.socket:
        print(
            "vc3d-mcp: no bridge socket configured. Pass --socket <name-or-path> "
            "or set VC3D_AGENT_BRIDGE_SOCKET.",
            file=sys.stderr,
        )
        return 2

    configure_client(args.socket, request_timeout=args.request_timeout)
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
