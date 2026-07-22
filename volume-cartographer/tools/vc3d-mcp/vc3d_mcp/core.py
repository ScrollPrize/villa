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

import asyncio
from typing import Any, Optional

from mcp.server.fastmcp import Context, FastMCP

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


async def _report_progress(
    ctx: Optional[Context], seq: int, message: str, timeout: float | None = None
) -> None:
    """Best-effort MCP progress report. Never lets a reporting failure change
    the job's execution or the tool's terminal result (progress is purely
    observational), but never swallows task cancellation either.

    When ``timeout`` is given, delivery is bounded by it: a slow/blocked MCP
    progress sink is abandoned (best-effort, logged as a skipped report) rather
    than allowed to hold the tool past the wait deadline. A non-positive timeout
    skips the report entirely."""
    if ctx is None:
        return
    if timeout is not None and timeout <= 0:
        return
    try:
        coro = ctx.report_progress(progress=seq, total=None, message=message)
        if timeout is not None:
            await asyncio.wait_for(coro, timeout=timeout)
        else:
            await coro
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - progress is best-effort/observational
        # Includes asyncio.TimeoutError: a blocked sink must not fail the job or
        # delay terminal completion; drop this report and carry on.
        pass


async def _wait_for_job(
    job_id: str,
    wait: bool,
    initial_result: dict[str, Any],
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Shared `wait: true` handling for the async long-running tools.

    When ``wait`` is true, blocks the tool call until the job's finished
    notification and returns the terminal ``job.status`` inline. ``wait``
    defaults to false. When true, the MCP server enforces a 30-minute cap and
    returns the still-running ``jobId`` (with ``waitTimedOut: true``) on
    timeout, and fails promptly if the bridge disconnects.

    All MCP progress reporting is centralized here (tool wrappers only pass
    ``ctx`` through). Retained and newly-arriving ``phase:"output"`` lines are
    forwarded via ``ctx.report_progress`` with a monotonically increasing local
    sequence number. Forwarding is best-effort: it never affects the result, and
    an individual report that fails or exceeds the remaining wait deadline is
    dropped (so a slow/blocked sink can't hold the tool past the cap). Delivery
    of every line is therefore not guaranteed -- output beyond the tracker's
    bounded retention window may also have been evicted before a late waiter
    registers -- but ``job.status`` (returned on completion) remains authoritative.
    """
    if not wait:
        return initial_result

    client = _get_client()
    tracker = client.jobs
    record = tracker.register(job_id)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + DEFAULT_WAIT_TIMEOUT_S
    local_seq = 0          # monotonic progress index reported to the MCP client
    last_output_seq = 0    # highest job output sequence already forwarded
    seen_error = record.error  # baseline; a NEW latched error fails this wait
    timed_out = False

    async def _forward_new_outputs() -> None:
        # Forward retained/newly-arrived output lines, each bounded by the
        # REMAINING deadline so a slow/blocked MCP progress sink can never hold
        # the tool past the 30-minute cap. Progress is observational only: a
        # timed-out/failed report is dropped, never fails the job.
        nonlocal local_seq, last_output_seq
        for output_seq, line in record.outputs_after(last_output_seq):
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            last_output_seq = output_seq
            local_seq += 1
            await _report_progress(ctx, local_seq, line, timeout=remaining)

    while True:
        wake = record.wake  # capture BEFORE reading state (no lost wakeup)
        if record.error is not None and record.error is not seen_error:
            # Disconnect latched on our record while we waited; observe it even
            # if a racing reconnect already cleared the global error.
            raise RuntimeError(str(record.error))
        if tracker.error is not None:
            # Bridge disconnected: fail promptly rather than block to the cap.
            raise RuntimeError(str(tracker.error))

        await _forward_new_outputs()

        if record.finished_event.is_set():
            break

        remaining = deadline - loop.time()
        if remaining <= 0:
            timed_out = True
            break
        try:
            await asyncio.wait_for(wake.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            timed_out = True
            break

    if timed_out:
        return {**initial_result, "waitTimedOut": True}

    # Terminal state observed. Output notifications that landed together with (or
    # just before) `finished` -- possibly while a slow progress report was being
    # awaited -- would otherwise be dropped when we break above. Flush once more
    # so trailing lines are forwarded. Best-effort and deadline-bounded; this
    # never blocks terminal completion.
    await _forward_new_outputs()

    # job.status is the bridge's authoritative terminal record (SPEC.md 3.17);
    # prefer it over our locally-tracked notification data, falling back to
    # the local record if the RPC itself races with process shutdown.
    try:
        status = await _call("job.status", {"jobId": job_id})
    except Exception:
        status = record.as_dict()
    return {**initial_result, **status}
