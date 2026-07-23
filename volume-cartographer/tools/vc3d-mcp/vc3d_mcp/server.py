"""MCP stdio entrypoint for the VC3D Agent Bridge.

Resolves the bridge socket by priority (explicit ``--socket``/env, then the
discovery registry, then auto-launching VC3D), configures the client, and
runs the FastMCP stdio loop. The ``@mcp.tool()`` functions live in
:mod:`vc3d_mcp.tools`; importing that package registers them on the shared
``mcp`` instance from :mod:`vc3d_mcp.core`.
"""

from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import threading
import time
from collections import deque

from .bridge_client import (
    BridgeClient,
    BridgeConnectionError,
    discover_registry_socket,
)
from .core import configure_client, mcp
from . import tools  # noqa: F401  - imports every module, registering its @mcp.tool()s


# Handshake wait budget for the auto-launch path: VC3D can take a while to
# construct CWindow before AgentBridgeServer::listen() prints the handshake.
LAUNCH_HANDSHAKE_TIMEOUT_S = 30.0

# Kept alive for the process lifetime so a bridge we launched isn't reaped as a
# child zombie, and can be terminated on our own exit.
_launched_process: subprocess.Popen | None = None


def default_vc3d_binary() -> str:
    """The fallback VC3D binary path, resolved relative to the repo root.

    ``server.py`` lives at ``<repo>/tools/vc3d-mcp/vc3d_mcp/server.py``, so the
    repo root is three directories up.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir))
    return os.path.join(repo_root, "build-macos", "bin", "VC3D")


def resolve_launch_binary(explicit: str | None) -> str | None:
    """Pick the VC3D binary to auto-launch, or None if none is usable.

    Priority: explicit ``--launch`` > ``VC3D_BINARY`` env var > the repo-root
    default. Returns the path only if it names a real, executable file.
    """
    candidate = explicit or os.environ.get("VC3D_BINARY") or default_vc3d_binary()
    if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def _launch_command(binary: str, volpkg: str | None) -> list[str]:
    command = [binary, "--agent-bridge"]
    if volpkg:
        command += ["--volpkg", volpkg]
    return command


def launch_vc3d(
    binary: str,
    volpkg: str | None = None,
    timeout: float = LAUNCH_HANDSHAKE_TIMEOUT_S,
) -> str:
    """Spawn VC3D with the agent bridge enabled and return its socket path.

    A daemon thread drains the child's stdout for the whole process lifetime
    (with stderr merged in): it detects the ``VC3D-AGENT-BRIDGE: listening ...``
    handshake line (parsed by ``BridgeClient.socket_path_from_handshake``),
    signals a ``threading.Event`` with the authoritative ``path=`` value, and
    keeps reading so VC3D's stdout pipe keeps draining and never blocks VC3D.
    Non-handshake lines are forwarded to this process's STDERR only (stdout is
    reserved for MCP stdio); a bounded tail of recent lines is also retained
    for error reporting.

    Waits up to ``timeout`` seconds for the handshake. A silent-but-live child
    times out at the deadline; a child that exits before the handshake fails
    immediately with its captured log tail. Raises ``BridgeConnectionError`` in
    both failure cases. The process is retained (module global) so it keeps
    running for the MCP session and is terminated (escalating) on our exit.
    """
    global _launched_process

    proc = subprocess.Popen(
        _launch_command(binary, volpkg),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # keep diagnostics; drained below
        text=True,
        bufsize=1,  # line-buffered
    )
    _launched_process = proc
    atexit.register(_terminate_launched_process)

    handshake: dict[str, str] = {}
    found = threading.Event()
    tail: deque[str] = deque(maxlen=200)

    def _drain() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.rstrip("\n")
            tail.append(stripped)
            if not found.is_set():
                parsed = BridgeClient.socket_path_from_handshake(line)
                if parsed is not None:
                    handshake["path"] = parsed[1]
                    found.set()
                    continue
            try:
                print(stripped, file=sys.stderr)
            except Exception:  # noqa: BLE001 - stderr sink gone; keep draining
                pass
        found.set()  # EOF: unblock the waiter even if no handshake arrived

    threading.Thread(target=_drain, name="vc3d-stdout-drain", daemon=True).start()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if found.wait(timeout=0.1):
            if "path" in handshake:
                return handshake["path"]
            break  # EOF set the event with no handshake
        if proc.poll() is not None:
            break

    _terminate_launched_process()
    detail = "\n".join(tail)
    if proc.returncode is not None:
        raise BridgeConnectionError(
            f"VC3D ({binary!r}) exited with code {proc.returncode} before printing the "
            f"agent-bridge handshake line. Last output:\n{detail}"
        )
    raise BridgeConnectionError(
        f"timed out after {timeout:g}s waiting for VC3D ({binary!r}) to print its "
        f"agent-bridge handshake line. Last output:\n{detail}"
    )


def _reap_process(proc: subprocess.Popen) -> None:
    """Block until `proc` is reaped, so a slow-exiting child can't linger as a
    zombie after teardown gave up waiting. Runs on a daemon thread."""
    try:
        proc.wait()
    except Exception:  # noqa: BLE001 - already-reaped / OS error: nothing to do
        pass


def _terminate_launched_process() -> None:
    global _launched_process
    proc = _launched_process
    if proc is None:
        return
    _launched_process = None
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # kill() was delivered but the child hasn't been reaped within
                # the grace period. Reap it in the background so it can't linger
                # as a zombie (an unconditional final waitpid).
                threading.Thread(
                    target=_reap_process, args=(proc,), name="vc3d-reaper", daemon=True
                ).start()
    except OSError:
        pass


def resolve_connection(args: argparse.Namespace) -> tuple[str | None, str]:
    """Determine the bridge socket to use, per the documented priority order.

    Returns ``(socket_path_or_None, how)`` where ``how`` describes the source
    for diagnostics:

    (a) explicit ``--socket`` / ``VC3D_AGENT_BRIDGE_SOCKET`` (highest priority);
    (b) an already-running bridge found in the discovery registry;
    (c) auto-launch a new VC3D when ``--launch``/``VC3D_BINARY``/the default
        binary resolves to a real executable;
    (d) otherwise ``(None, ...)`` -- the caller prints the usage error.
    """
    if args.socket:
        return args.socket, "explicit --socket/env"

    discovered = discover_registry_socket()
    if discovered:
        return discovered, "discovery registry (already-running VC3D)"

    binary = resolve_launch_binary(args.launch)
    if binary:
        print(
            f"vc3d-mcp: no running bridge found; launching VC3D at {binary!r}...",
            file=sys.stderr,
        )
        path = launch_vc3d(binary, volpkg=args.volpkg)
        return path, f"auto-launched VC3D ({binary})"

    return None, "no socket, no running bridge, no launchable binary"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vc3d-mcp",
        description=(
            "MCP server (stdio) that drives a running VC3D instance via its "
            "Agent Bridge local socket. In the common case no socket needs to "
            "be passed: an already-running VC3D is found via the discovery "
            "registry (~/.vc3d/agent_bridge), and if none is running VC3D is "
            "auto-launched."
        ),
    )
    parser.add_argument(
        "--socket",
        default=os.environ.get("VC3D_AGENT_BRIDGE_SOCKET"),
        help=(
            "VC3D agent bridge socket: an explicit path (e.g. the 'path=' field "
            "from the VC3D-AGENT-BRIDGE stdout handshake line), or a bare "
            "QLocalServer name matching --agent-bridge-name. Highest priority "
            "when set. Defaults to the VC3D_AGENT_BRIDGE_SOCKET env var. When "
            "unset, an already-running bridge is auto-discovered, else VC3D is "
            "auto-launched (see --launch)."
        ),
    )
    parser.add_argument(
        "--launch",
        default=None,
        help=(
            "Path to a VC3D binary to auto-launch (with --agent-bridge) when no "
            "socket is given and no running bridge is discovered. Falls back to "
            "the VC3D_BINARY env var, then to the repo-root build "
            "(build-macos/bin/VC3D). Ignored when --socket or a running bridge "
            "is available."
        ),
    )
    parser.add_argument(
        "--volpkg",
        default=None,
        help=(
            "Optional volume-package path forwarded to an auto-launched VC3D as "
            "--volpkg, so the agent's first action need not be opening a "
            "volume. Only used on the auto-launch path."
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

    try:
        socket, how = resolve_connection(args)
    except BridgeConnectionError as exc:
        print(f"vc3d-mcp: auto-launch failed: {exc}", file=sys.stderr)
        return 2

    if not socket:
        print(
            "vc3d-mcp: could not connect to a VC3D agent bridge. Options, in "
            "priority order:\n"
            "  1. Pass --socket <name-or-path> (or set VC3D_AGENT_BRIDGE_SOCKET) "
            "to attach to a known bridge.\n"
            "  2. Start VC3D with --agent-bridge and it will be auto-discovered "
            "via ~/.vc3d/agent_bridge.\n"
            "  3. Pass --launch <path-to-VC3D> (or set VC3D_BINARY, or build "
            "build-macos/bin/VC3D) to have this server launch VC3D itself.",
            file=sys.stderr,
        )
        return 2

    print(f"vc3d-mcp: using bridge socket {socket!r} via {how}.", file=sys.stderr)
    configure_client(socket, request_timeout=args.request_timeout)
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
