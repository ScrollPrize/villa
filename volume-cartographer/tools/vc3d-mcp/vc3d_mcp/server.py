"""MCP server exposing the VC3D Agent Bridge RPCs as MCP tools.

One MCP tool per JSON-RPC method in apps/VC3D/agent_bridge/SPEC.md section 3,
per the tool surface table in SPEC.md section 5. Uses the official MCP Python
SDK (`mcp` package, `FastMCP`) over stdio -- see README.md "Implementation
notes" for why (the SDK installs cleanly in this environment, so we didn't
need to hand-roll the protocol).

This module is now a thin entrypoint: the ``mcp`` instance and shared helpers
live in :mod:`vc3d_mcp.core`, and the ~74 ``@mcp.tool()`` functions live in the
per-domain modules under :mod:`vc3d_mcp.tools`. Importing this module imports all
of them (registering every tool on the shared ``mcp``) and re-exports them at
module scope for backward compatibility (e.g. ``vc3d_mcp.server.vc3d_ping``).
"""

from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import time

from .bridge_client import (
    BridgeClient,
    BridgeConnectionError,
    discover_registry_socket,
)

# Re-export the shared instance + client plumbing from core so existing callers
# and the test suite can keep importing them from ``vc3d_mcp.server``.
from .core import (  # noqa: F401
    DEFAULT_WAIT_TIMEOUT_S,
    configure_client,
    mcp,
    _call,
    _get_client,
    _is_placeholder_error,
    _strip_none,
    _wait_for_job,
)

# Importing each domain module runs its @mcp.tool() decorators (registering the
# tools) and the star-imports below re-export the tool functions at this module
# scope, preserving ``vc3d_mcp.server.<tool>`` back-compat.
from .tools.atlas import *  # noqa: F401,F403
from .tools.catalog_volume import *  # noqa: F401,F403
from .tools.fiber import *  # noqa: F401,F403
from .tools.flatten import *  # noqa: F401,F403
from .tools.lasagna import *  # noqa: F401,F403
from .tools.manual_add import *  # noqa: F401,F403
from .tools.misc import *  # noqa: F401,F403
from .tools.points import *  # noqa: F401,F403
from .tools.seeding import *  # noqa: F401,F403
from .tools.segmentation import *  # noqa: F401,F403
from .tools.viewer import *  # noqa: F401,F403


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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


def launch_vc3d(
    binary: str,
    volpkg: str | None = None,
    timeout: float = LAUNCH_HANDSHAKE_TIMEOUT_S,
) -> str:
    """Spawn VC3D with the agent bridge enabled and return its socket path.

    Reads the child's stdout until the ``VC3D-AGENT-BRIDGE: listening ...``
    handshake line appears (parsed by ``BridgeClient.socket_path_from_handshake``),
    then returns the authoritative ``path=`` value. Raises
    ``BridgeConnectionError`` if VC3D exits or the handshake never arrives
    within ``timeout`` seconds. The process is retained (module global) so it
    keeps running for the MCP session and is terminated on our exit.
    """
    global _launched_process

    args = [binary, "--agent-bridge"]
    if volpkg:
        # SPEC: forwarded to VC3D as --load-first so the agent doesn't have to
        # open a volume as its first action.
        args += ["--load-first", volpkg]

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,  # line-buffered
    )
    _launched_process = proc
    atexit.register(_terminate_launched_process)

    deadline = time.monotonic() + timeout
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise BridgeConnectionError(
                f"VC3D ({binary!r}) exited with code {proc.returncode} "
                "before printing the agent-bridge handshake line"
            )
        line = proc.stdout.readline()
        if not line:
            # EOF without exit yet: brief pause, then re-check poll/deadline.
            time.sleep(0.05)
            continue
        parsed = BridgeClient.socket_path_from_handshake(line)
        if parsed is not None:
            _name, path = parsed
            return path

    _terminate_launched_process()
    raise BridgeConnectionError(
        f"timed out after {timeout:g}s waiting for VC3D ({binary!r}) to print "
        "its agent-bridge handshake line"
    )


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
            "--load-first, so the agent's first action need not be opening a "
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
