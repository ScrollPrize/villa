"""Shared infrastructure for the vc3d-mcp tool modules.

This is the single source of the ``mcp`` :class:`FastMCP` instance that every
``vc3d_mcp.tools.*`` module registers its ``@mcp.tool()`` functions on, plus the
bridge-client plumbing (``configure_client`` / ``_get_client`` / ``_call``) and
the shared ``wait: true`` job handling. Split out of the original monolithic
``server.py`` so the domain tool modules can ``from vc3d_mcp.core import mcp,
_call, _wait_for_job`` without importing the entrypoint (and without a circular
import back through ``server.py``).
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .bridge_client import (
    BridgeClient,
    BridgeClientConfig,
    BridgeConnectionError,
    BridgeError,
)

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


def _is_placeholder_error(exc: Exception) -> bool:
    """True when a segments.activate failure is the 'unmaterialized open-data
    placeholder; fetch it first' refusal (SurfacePanelController::activateSurfaceById),
    so vc3d_activate_segment knows a fetch+retry can resolve it."""
    if not isinstance(exc, BridgeError):
        return False
    detail = ""
    if isinstance(exc.data, dict):
        detail = str(exc.data.get("detail", ""))
    return "placeholder" in detail.lower() or "placeholder" in str(exc.message).lower()


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
