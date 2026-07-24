"""Lasagna fit-service and optimization tools."""

from __future__ import annotations

from typing import Any, Literal, Optional

from mcp.server.fastmcp import Context
from typing_extensions import TypedDict

from ..core import mcp, _call, _wait_for_job, _strip_none


class _Vec3(TypedDict):
    x: float
    y: float
    z: float


@mcp.tool()
async def vc3d_lasagna_service_status() -> dict[str, Any]:
    """Report the Lasagna fit-service state without starting anything. Never
    errors.

    Returns {"running": bool, "external": bool, "host": str, "port": int,
    "lastError": str | null} -- external is true when attached to an
    externally-managed service (see vc3d_lasagna_ensure_service); lastError is
    null when no error has been recorded.
    """
    return await _call("lasagna.service_status", {})


@mcp.tool()
async def vc3d_lasagna_ensure_service(
    python_path: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> dict[str, Any]:
    """Ensure a Lasagna fit service is available, starting or connecting to one
    as needed. This service powers ONLY the vc3d_lasagna_* optimization panel;
    fiber tracing (vc3d_fiber_launch) and atlas creation do NOT use it -- they
    need a "lasagna"-kind dataset resolvable for the current volume, not this
    service. Two modes, selected by whether host+port are given:

    - Internal (default): omit both host and port. Launches a local service
    process (optionally using python_path as the interpreter) and blocks until
    it is up or fails -- returns synchronously.
    - External: give host AND port together (giving only one is -32602). Pings
    the external service's GET /health; the call is deferred (~15 s cap) and
    completes when the service answers, or fails -32005 with data.detail.

    python_path: interpreter path for the internal service; ignored in external
    mode.
    host / port: external service address; must be supplied together.

    Returns {"running": true, "external": bool, "host": str, "port": int}.
    Errors: -32005 (failed to start / connect, message in data.detail), -32602
    (host without port or vice versa). After this succeeds, vc3d_lasagna_list_datasets
    / vc3d_lasagna_jobs / vc3d_lasagna_start_optimization become usable.
    """
    return await _call(
        "lasagna.ensure_service",
        _strip_none({"pythonPath": python_path, "host": host, "port": port}),
    )


@mcp.tool()
async def vc3d_lasagna_list_datasets() -> dict[str, Any]:
    """List the datasets the Lasagna fit service knows about -- the optimization
    panel's server-side datasets, NOT the "lasagna"-kind dataset that
    vc3d_fiber_launch / atlas creation resolve for the current volume. Requires
    the service to be running (see vc3d_lasagna_ensure_service). Deferred
    (~10 s cap) while the service is queried.

    Returns {"datasets": [...]} -- the service's dataset objects passed through
    verbatim (the bridge does not reshape service JSON). Errors: -32005 (service
    not running, or the fetch timed out).
    """
    return await _call("lasagna.list_datasets", {})


@mcp.tool()
async def vc3d_lasagna_start_optimization(
    mode: Literal["reoptimize", "new_model", "offset", "atlas"],
    config_path: Optional[str] = None,
    seed: Optional[_Vec3] = None,
    atlas_path: Optional[str] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Start a Lasagna optimization on the open volume package. Async (a
    source:"lasagna" job -- poll vc3d_job_status, or pass wait=true). Requires
    the Lasagna service running and the Lasagna panel constructed.

    mode: "reoptimize" | "new_model" | "offset" | "atlas".
    config_path: path to the optimization config; default is the panel's
    selected config for the chosen mode. Resolving to nothing on disk fails
    -32007 (data.kind:"config").
    seed: optional volume-space {"x","y","z"} seed; its components are rounded to
    integers.
    atlas_path: required for mode "atlas" unless the panel already has an atlas
    selected; missing fails -32007 (data.kind:"atlas"). Ignored for other modes.
    wait: if true (MCP-server-side only, not part of the underlying RPC), block
    until the job finishes (30-minute cap) and return the terminal job.status
    inline instead of just the jobId; on timeout the jobId is returned with an
    extra "waitTimedOut": true.

    Returns {"jobId", "kind": "lasagna.optimize", "source": "lasagna"}. Errors:
    -32000 (no volume package), -32004 (data.source:"lasagna" -- the bridge
    allows one in-flight bridge-submitted optimization; use vc3d_lasagna_jobs for
    the service's own queue), -32005 (service not running / submission failed,
    data.detail), -32007 (config/atlas, see above), -32009 (Lasagna panel
    unavailable), -32602 (bad mode string or malformed seed).
    """
    result = await _call(
        "lasagna.start_optimization",
        _strip_none(
            {
                "mode": mode,
                "configPath": config_path,
                "seed": seed,
                "atlasPath": atlas_path,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_lasagna_jobs() -> dict[str, Any]:
    """List the Lasagna service's job queue (including the local-upload overlay
    entries the manager merges in). Requires the service to be running. Deferred
    (~10 s cap).

    Returns {"jobs": [...]} -- service job objects passed through verbatim.
    Errors: -32005 (service not running, or the fetch timed out).
    """
    return await _call("lasagna.jobs", {})


@mcp.tool()
async def vc3d_lasagna_cancel(job_id: Optional[str] = None) -> dict[str, Any]:
    """Cancel a Lasagna optimization. Requires the service to be running.

    job_id: a bridge job id ("job-<n>", resolved to its underlying service job
    id) or a raw service job id passed straight through. Omit to stop the active
    bridge-submitted optimization.

    Returns {"cancelRequested": true, "serviceJobId": str | null (null when the
    active optimization was stopped without a specific id)}. Errors: -32007
    (data.kind:"job" -- unknown bridge id, or omitted with no active lasagna
    job), -32005 (service not running).
    """
    return await _call("lasagna.cancel", _strip_none({"jobId": job_id}))


@mcp.tool()
async def vc3d_lasagna_select_output(name: str) -> dict[str, Any]:
    """Activate a Lasagna output segment by name -- the programmatic twin of
    picking a lasagna output in the panel. Requires a volume package to be open.

    name: the output segment id/name; must be non-empty (empty fails -32602).

    Returns {"selected": true, "name": str}. Errors: -32000 (no volume package),
    -32602 (empty name), -32007 (data.kind:"segment" -- unknown or unselectable
    segment).
    """
    return await _call("lasagna.select_output_segment", {"name": name})


@mcp.tool()
async def vc3d_lasagna_repeat_last(
    wait: bool = False, ctx: Optional[Context] = None
) -> dict[str, Any]:
    """Repeat the last Lasagna optimization -- relaunch the last-used mode with
    its settings. Async (a source:"lasagna" job -- poll vc3d_job_status, or pass
    wait=true); same job semantics as vc3d_lasagna_start_optimization. Requires
    a volume package open and the Lasagna panel constructed.

    wait: if true (MCP-server-side only), block until the job finishes (30-minute
    cap) and return the terminal job.status inline; on timeout the jobId is
    returned with an extra "waitTimedOut": true.

    Returns {"jobId", "kind": "lasagna.optimize", "source": "lasagna"}. Errors:
    -32000 (no volume package), -32004 (data.source:"lasagna" -- one already in
    flight), -32005 (nothing to repeat / launch failed, data.detail), -32009
    (Lasagna panel unavailable).
    """
    result = await _call("lasagna.repeat_last", {})
    return await _wait_for_job(result["jobId"], wait, result, ctx)
