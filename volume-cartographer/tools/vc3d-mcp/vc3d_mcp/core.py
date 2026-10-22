"""Shared infrastructure for the vc3d-mcp tool modules.

Holds the single ``mcp`` :class:`FastMCP` instance every ``vc3d_mcp.tools.*``
module registers its ``@mcp.tool()`` functions on, the bridge-client plumbing
(``configure_client`` / ``_get_client`` / ``_call``), and the shared
``wait: true`` job handling.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
from typing import Any, Literal, Optional

from mcp.server.fastmcp import Context, FastMCP, Image

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


# ``observe`` on the interaction tools: "state" diffs state.get around the
# call. MCP-only, like ``wait``. Screenshots come from vc3d_observe and
# vc3d_screenshot, so the interaction tools keep returning plain dicts.
ObserveMode = Literal["none", "state"]
OBSERVE_SCREENSHOT_MAX_DIM = 1024

# The last state.get snapshot an observation saw; vc3d_observe diffs against
# it. Like the bridge client's state.changed bookkeeping, it belongs to this
# server process, which serves one MCP client over stdio.
_last_observed_state: dict[str, Any] | None = None


def _json_pointer_token(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def state_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    """JSON Patch (RFC 6902) add/remove/replace operations turning ``before``
    into ``after``. Objects are compared key by key and equal-length arrays
    element by element; any other difference replaces the value at its path."""
    if isinstance(before, dict) and isinstance(after, dict):
        ops: list[dict[str, Any]] = []
        for key in sorted(before.keys() | after.keys()):
            child = f"{path}/{_json_pointer_token(str(key))}"
            if key not in after:
                ops.append({"op": "remove", "path": child})
            elif key not in before:
                ops.append({"op": "add", "path": child, "value": after[key]})
            else:
                ops.extend(state_diff(before[key], after[key], child))
        return ops
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        ops = []
        for index, (old, new) in enumerate(zip(before, after)):
            ops.extend(state_diff(old, new, f"{path}/{index}"))
        return ops
    if before == after and isinstance(before, bool) == isinstance(after, bool):
        return []
    return [{"op": "replace", "path": path, "value": after}]


def _observable_state(state: Any) -> dict[str, Any]:
    """state.get without the fields that change on every read: the revision
    (reported on its own) and running jobs' console and progress tails."""
    if not isinstance(state, dict):
        return {}
    observed = copy.deepcopy(state)
    observed.pop("revision", None)
    jobs = observed.get("jobs")
    if isinstance(jobs, list):
        for job in jobs:
            if isinstance(job, dict):
                job.pop("consoleTail", None)
                job.pop("progressHistory", None)
    return observed


async def _observation(
    mode: str,
    before: dict[str, Any] | None,
    screenshot_target: str,
) -> tuple[dict[str, Any], Image | None]:
    """Read state and/or a screenshot for an observation. ``before`` is the
    snapshot to diff against (None when there is no baseline)."""
    global _last_observed_state
    observation: dict[str, Any] = {}
    if mode in ("state", "both"):
        after = await _call("state.get")
        observation["revision"] = after.get("revision") if isinstance(after, dict) else None
        if before is None:
            observation["baseline"] = True
        else:
            observation["stateDiff"] = state_diff(
                _observable_state(before), _observable_state(after)
            )
        _last_observed_state = after if isinstance(after, dict) else None
    image: Image | None = None
    if mode in ("screenshot", "both"):
        try:
            shot = await _call(
                "screenshot.capture",
                {"target": screenshot_target, "maxDim": OBSERVE_SCREENSHOT_MAX_DIM},
            )
            data = shot.get("base64") if isinstance(shot, dict) else None
            if data:
                image = Image(data=base64.b64decode(data), format="png")
                observation["screenshot"] = {
                    "target": screenshot_target,
                    "width": shot.get("width"),
                    "height": shot.get("height"),
                }
        except BridgeError as exc:
            observation["screenshotError"] = exc.to_dict()
    # Taken after the last bridge call: the bridge sends a request's
    # state.changed before its response, so every change up to here is in.
    _, changed = _get_client().take_state_changes()
    observation["changed"] = changed
    return observation, image


async def _observed_call(
    method: str,
    params: dict[str, Any] | None,
    observe: str,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Call a bridge method, then observe its effect.

    ``observe="none"`` returns the bridge result unchanged. With "state" the
    result dict gains an ``observation`` object: ``stateDiff`` (JSON Patch
    operations between state.get just before and just after the call),
    ``revision``, and ``changed`` (state.changed fields seen since the last
    observation, including changes a person made).
    """
    if observe == "none":
        return await _call(method, params, timeout=timeout)
    before = await _call("state.get")
    result = await _call(method, params, timeout=timeout)
    observation, _ = await _observation("state", before, "window")
    out = dict(result) if isinstance(result, dict) else {"result": result}
    out["observation"] = observation
    return out


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
