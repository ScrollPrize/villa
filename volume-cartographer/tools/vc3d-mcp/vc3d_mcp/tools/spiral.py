"""Thin MCP wrappers for VC3D's existing Spiral workspace service."""

from __future__ import annotations

from typing import Any, Literal, Optional

from ..core import mcp, _call, _strip_none

InputKind = Literal["patch", "fiber", "pcl"]


@mcp.tool()
async def vc3d_spiral_status() -> dict[str, Any]:
    """Return the current Spiral connection and session state."""
    return await _call("spiral.status", {})


@mcp.tool()
async def vc3d_spiral_list_profiles() -> dict[str, Any]:
    """List GUI-configured Spiral profiles without credential values."""
    return await _call("spiral.profiles", {})


@mcp.tool()
async def vc3d_spiral_connect(profile_id: str) -> dict[str, Any]:
    """Connect using a saved GUI profile. Poll vc3d_spiral_status for ready."""
    return await _call("spiral.connect", {"profileId": profile_id})


@mcp.tool()
async def vc3d_spiral_disconnect(force: bool = False) -> dict[str, Any]:
    """Disconnect; force is required when VC3D owns the local service."""
    return await _call("spiral.disconnect", {"force": force})


@mcp.tool()
async def vc3d_spiral_reconnect() -> dict[str, Any]:
    """Reconnect the current saved profile."""
    return await _call("spiral.reconnect", {})


@mcp.tool()
async def vc3d_spiral_get_dataset() -> dict[str, Any]:
    """Return the service-advertised dataset and checkpoints."""
    return await _call("spiral.dataset", {})


@mcp.tool()
async def vc3d_spiral_rebuild(
    confirm: bool,
    defaults: Optional[bool] = None,
    request: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Rebuild using defaults or the same request object accepted by the GUI."""
    return await _call(
        "spiral.rebuild",
        _strip_none({"confirm": confirm, "defaults": defaults, "request": request}),
    )


@mcp.tool()
async def vc3d_spiral_run(
    iterations: int,
    influence_config: Optional[dict[str, Any]] = None,
    run_config: Optional[dict[str, Any]] = None,
    inputs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Submit positive Spiral iterations using the GUI service client."""
    return await _call(
        "spiral.run",
        _strip_none({
            "iterations": iterations,
            "influenceConfig": influence_config,
            "runConfig": run_config,
            "inputs": inputs,
        }),
    )


@mcp.tool()
async def vc3d_spiral_stop() -> dict[str, Any]:
    """Request stop after the current iteration."""
    return await _call("spiral.stop", {})


@mcp.tool()
async def vc3d_spiral_export_preview() -> dict[str, Any]:
    """Request preview export; inspect the Spiral workspace for completion."""
    return await _call("spiral.preview_export", {})


@mcp.tool()
async def vc3d_spiral_save_checkpoint(name: str) -> dict[str, Any]:
    """Request a named service-host checkpoint."""
    return await _call("spiral.checkpoint_save", {"name": name})


@mcp.tool()
async def vc3d_spiral_download_checkpoint(local_path: str) -> dict[str, Any]:
    """Create and download a checkpoint to an absolute local path."""
    return await _call(
        "spiral.checkpoint_download", {"localPath": local_path}, timeout=610.0
    )


@mcp.tool()
async def vc3d_spiral_load_checkpoint(
    host_path: Optional[str] = None,
    local_path: Optional[str] = None,
    allow_rebuild: bool = False,
) -> dict[str, Any]:
    """Load exactly one advertised host checkpoint or local checkpoint."""
    return await _call(
        "spiral.checkpoint_load",
        _strip_none({
            "hostPath": host_path,
            "localPath": local_path,
            "allowRebuild": allow_rebuild,
        }),
        timeout=610.0,
    )


@mcp.tool()
async def vc3d_spiral_upload_input(
    kind: InputKind,
    local_path: str,
    input_id: str,
    role: Optional[str] = None,
) -> dict[str, Any]:
    """Upload a patch directory, fiber JSON, or PCL input."""
    return await _call(
        "spiral.input_upload",
        _strip_none({
            "kind": kind,
            "localPath": local_path,
            "inputId": input_id,
            "role": role,
        }),
        timeout=610.0,
    )


@mcp.tool()
async def vc3d_spiral_remove_input(
    kind: InputKind, input_id: str
) -> dict[str, Any]:
    """Remove an uncommitted input through the GUI service client."""
    return await _call(
        "spiral.input_remove", {"kind": kind, "inputId": input_id}
    )


@mcp.tool()
async def vc3d_spiral_commit_inputs() -> dict[str, Any]:
    """Commit all uploaded ephemeral inputs into the resident fit."""
    return await _call("spiral.inputs_commit", {}, timeout=250.0)
