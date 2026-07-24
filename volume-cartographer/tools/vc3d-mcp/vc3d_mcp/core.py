"""Shared infrastructure for the vc3d-mcp tool modules.

Holds the single ``mcp`` :class:`FastMCP` instance every ``vc3d_mcp.tools.*``
module registers its ``@mcp.tool()`` functions on, the bridge-client plumbing
(``configure_client`` / ``_get_client`` / ``_call``), and the shared
``wait: true`` job handling.
"""

from __future__ import annotations

import asyncio
import json
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
            "vc3d-mcp server was not configured with a bridge endpoint "
            "(configure_client() was never called before serving requests)"
        )
    return _client


async def _call(
    method: str,
    params: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> Any:
    """Call a bridge RPC, translating transport failures into a clear message.

    JSON-RPC error replies (BridgeError) are left to propagate as-is -- its
    `__str__` is the `{"code", "message", "data"}` JSON object per SPEC.md
    section 2.5, which is what ends up in the MCP tool's error text, per the
    "RPC errors surface as MCP tool errors with code/message/data preserved"
    requirement in SPEC.md section 5.

    `timeout` overrides the client's default request timeout for this one call
    (None => the configured default). Deferred RPCs whose server-side cap
    exceeds that default (e.g. the mask renders) pass a value comfortably above
    the cap so the client does not give up while the bridge is still working.
    """
    client = _get_client()
    try:
        return await client.call(method, params, timeout=timeout)
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


POLL_INTERVAL_S = 1.0
PROGRESS_REPORT_TIMEOUT_S = 1.0


def _new_tail_lines(prev: list[str], cur: list[str]) -> list[str]:
    """Lines in `cur` not already seen at the end of `prev`.

    consoleTail is a rolling window (last <=50 lines), so consecutive polls
    overlap: the longest suffix of `prev` that is a prefix of `cur` is old
    news; everything after it is new. Repeated identical lines can be
    under-reported -- acceptable, progress forwarding is best-effort.
    """
    for k in range(min(len(prev), len(cur)), 0, -1):
        if prev[len(prev) - k:] == cur[:k]:
            return cur[k:]
    return cur


async def _report_progress(
    ctx: Optional[Context], seq: int, message: str, timeout: float
) -> bool:
    """Report progress once, returning false when the sink is unavailable."""
    if ctx is None or timeout <= 0:
        return False
    try:
        await asyncio.wait_for(
            ctx.report_progress(progress=seq, total=None, message=message),
            timeout=timeout,
        )
        return True
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - progress is observational only
        return False


def _progress_message(update: dict[str, Any]) -> str:
    """Render output as text and terminal updates as compact structured data."""
    if update.get("phase") != "finished":
        return str(update.get("message") or update.get("phase") or "progress")
    fields = ("phase", "success", "message", "outputPath", "result")
    completion = {name: update[name] for name in fields if name in update}
    return json.dumps(completion, separators=(",", ":"), sort_keys=True)


async def _wait_for_job(
    job_id: str,
    wait: bool,
    initial_result: dict[str, Any],
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Shared `wait: true` handling for the long-running tools.

    Subscribes before reading status so buffered and live updates can be merged
    by sequence without a lost-wakeup window. A periodic status read remains as
    a best-effort delivery fallback and supplies the authoritative terminal
    result. Progress reporting is observational: one failing or stalled MCP
    sink disables it for this wait, while task cancellation still propagates.
    """
    if not wait:
        return initial_result

    loop = asyncio.get_running_loop()
    deadline = loop.time() + DEFAULT_WAIT_TIMEOUT_S
    seen_tail: list[str] = []
    last_seq = 0
    local_seq = 0
    reporting = ctx is not None

    async def forward(
        update: dict[str, Any],
        report_timeout: Optional[float] = None,
    ) -> None:
        nonlocal last_seq, local_seq, reporting
        raw_seq = update.get("seq")
        update_seq = int(raw_seq) if isinstance(raw_seq, (int, float)) else 0
        if update_seq and update_seq <= last_seq:
            return
        if update_seq:
            last_seq = update_seq
            local_seq = max(local_seq, update_seq)
        else:
            local_seq += 1
        if reporting:
            timeout = (
                PROGRESS_REPORT_TIMEOUT_S
                if report_timeout is None
                else report_timeout
            )
            reporting = await _report_progress(
                ctx, local_seq, _progress_message(update), timeout
            )

    client = _get_client()
    async with client.subscribe_job_progress(job_id) as updates:
        while True:
            status = await _call("job.status", {"jobId": job_id})
            history = status.get("progressHistory")
            if isinstance(history, list):
                # A slow but responsive sink must not turn a 64-item replay
                # into 64 independent timeout windows.
                replay_deadline = loop.time() + PROGRESS_REPORT_TIMEOUT_S
                for update in history:
                    if isinstance(update, dict):
                        await forward(update, replay_deadline - loop.time())
            else:
                # Compatibility with bridge versions that predate sequenced
                # progress history.
                tail = status.get("consoleTail") or []
                for line in _new_tail_lines(seen_tail, tail):
                    await forward({"phase": "output", "message": line})
                seen_tail = list(tail)

            if status.get("state") in ("succeeded", "failed"):
                return {**initial_result, **status}
            if loop.time() >= deadline:
                return {**initial_result, "waitTimedOut": True}

            next_status = min(loop.time() + POLL_INTERVAL_S, deadline)
            while True:
                remaining = next_status - loop.time()
                if remaining <= 0:
                    break
                try:
                    update = await asyncio.wait_for(
                        updates.get(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    break
                await forward(update)
                if update.get("phase") == "finished":
                    break
            # Re-read authoritative status after the fallback interval or a
            # terminal notification.
