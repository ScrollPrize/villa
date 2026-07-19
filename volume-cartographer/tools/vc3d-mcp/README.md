# vc3d-mcp

A standalone MCP (Model Context Protocol) server that lets an AI agent (e.g.
Claude Code) drive a running VC3D instance through the **VC3D Agent Bridge**
— the in-process JSON-RPC-over-local-socket automation channel implemented
in `apps/VC3D/agent_bridge/` (see
[`SPEC.md`](../../apps/VC3D/agent_bridge/SPEC.md) for the binding contract).

This directory is a separate Python process. It does not modify or embed any
VC3D C++ code; it only speaks the wire protocol AgentBridgeServer exposes.

Status: the C++ bridge (`AgentBridgeServer`) is being built concurrently and
is not yet available to connect to. This package has been validated against
a **fake** stand-in server (see "Self-test" below); connecting to a real
VC3D instance is a later integration phase.

## Implementation notes

- **MCP SDK**: the official `mcp` Python package (the Model Context
  Protocol SDK) installs cleanly in this environment and is used directly
  (`mcp.server.fastmcp.FastMCP`, stdio transport). We did not need to
  hand-roll the MCP JSON-RPC protocol.
- **Bridge transport**: a small asyncio client (`vc3d_mcp/bridge_client.py`)
  that speaks the bridge's wire format directly over a Unix domain socket
  (`asyncio.open_unix_connection`) — no Qt dependency needed on the Python
  side, since `QLocalServer`/`QLocalSocket` are plain AF_UNIX sockets on the
  Unix platforms VC3D ships for.
- **Async jobs**: `job.progress` notifications (SPEC.md §3.18) are consumed
  by a background reader task and folded into a small `JobTracker`
  (`vc3d_mcp/jobs.py`), which backs both `vc3d_job_status` and the
  `wait: true` convenience on the two `vc3d_grow_*` tools.

## Layout

```
tools/vc3d-mcp/
  vc3d_mcp/
    bridge_client.py   # asyncio client: socket resolution, JSON-RPC framing,
                        # request/response correlation, notification dispatch
    jobs.py            # JobTracker: turns job.progress notifications into
                        # pollable/awaitable job records
    server.py           # FastMCP app: one @mcp.tool() per RPC, CLI entry point
    __main__.py         # `python -m vc3d_mcp` entry point
  test_mcp_bridge.py    # self-test: fake AF_UNIX JSON-RPC server + assertions
  pyproject.toml
  requirements.txt
  README.md
```

## Running it

```sh
cd tools/vc3d-mcp
python3 -m venv .venv
./.venv/bin/pip install -e .        # or: pip install -r requirements.txt

# Point at a running VC3D's bridge socket:
VC3D_AGENT_BRIDGE_SOCKET=/path/to/socket ./.venv/bin/python -m vc3d_mcp

# ...or equivalently:
./.venv/bin/python -m vc3d_mcp --socket /path/to/socket
```

The server speaks MCP over stdio, so it's meant to be launched by an MCP
client (e.g. registered as an MCP server in Claude Code's config), not run
interactively. `--socket`/`VC3D_AGENT_BRIDGE_SOCKET` is required; the process
exits with status 2 and a stderr message if neither is set.

### Socket discovery / naming convention

Per SPEC.md §1.1, VC3D is started with `--agent-bridge` (default socket name
`vc3d-agent-<pid>`) or `--agent-bridge-name <name>` (explicit, e.g. for
tests), and on success prints exactly one line to stdout:

```
VC3D-AGENT-BRIDGE: listening name=<serverName> path=<QLocalServer::fullServerName()>
```

This MCP server takes the same value via `--socket <name-or-path>` or the
`VC3D_AGENT_BRIDGE_SOCKET` env var (SPEC.md §5), and resolves it to an actual
filesystem path as follows (`BridgeClient.resolve_socket_path`):

1. If the value is a path that already exists on disk, connect to it
   directly as an AF_UNIX socket. **This is the recommended, unambiguous
   mode** — pass the exact `path=...` value from the handshake line.
2. Otherwise, treat the value as a bare `QLocalServer` name (matching
   `--agent-bridge-name`) and probe `$TMPDIR/<name>` then `/tmp/<name>`
   (the conventional places Qt places local-server sockets on macOS/Linux).

SPEC.md does not pin down exact `QLocalServer` filesystem conventions across
platforms/Qt versions, so step 2 is a documented best-effort assumption; step
1 sidesteps the ambiguity entirely. A helper,
`BridgeClient.socket_path_from_handshake(line)`, is also provided for a
future phase where this server spawns VC3D itself and parses its stdout
handshake line directly instead of guessing a path.

## Self-test (no real VC3D required)

`test_mcp_bridge.py` starts a trivial fake JSON-RPC server on a local
AF_UNIX socket (standing in for `AgentBridgeServer`, which speaks the same
newline-delimited JSON-RPC 2.0 framing per SPEC.md §1.2) and checks that:

- `BridgeClient` connects, sends a request, and parses both success and
  JSON-RPC error responses (error `code`/`message`/`data` preserved);
- unsolicited `job.progress` notifications are picked up by the background
  reader and folded into `JobTracker` (console tail, terminal state);
- the actual `@mcp.tool()` functions in `vc3d_mcp/server.py` — not just the
  transport layer — round-trip correctly, including the `wait: true`
  behavior on `vc3d_grow_segment`.

Run it:

```sh
cd tools/vc3d-mcp
./.venv/bin/python test_mcp_bridge.py -v
```

All 9 cases pass as of this writing.

## Tool list

One MCP tool per JSON-RPC method in SPEC.md §3, named per SPEC.md §5
(`vc3d_` prefix, snake_case). Coordinates are volume-space voxels unless a
tool takes an explicit `space` field. RPC errors surface as MCP tool errors
whose text is the JSON-encoded `{"code", "message", "data"}` error object.

| MCP tool | → RPC | Description |
|---|---|---|
| `vc3d_ping` | `ping` | Check the VC3D bridge is alive; returns pid and app version. |
| `vc3d_get_state` | `state.get` | Snapshot of VC3D: open volume package, current volume, active segment, viewers, editing mode, running job. Call this first. |
| `vc3d_list_segments` | `segments.list` | List segments in the open volume package with loaded/active flags. |
| `vc3d_screenshot` | `screenshot.capture` | Capture a PNG of the whole VC3D window or one viewer pane; base64 or write to a file. |
| `vc3d_get_cursor_point` | `canvas.get_cursor_volume_point` | Resolve a viewer scene position (or the current cursor) to a 3D volume point + normal. |
| `vc3d_click` | `canvas.click` | Synthesize a mouse click in a viewer at a volume- or scene-space position, with button/modifiers. |
| `vc3d_shift_click` | `canvas.shift_click` | Shift+click convenience: the canonical place-point / set-focus gesture. |
| `vc3d_center_viewer` | `viewer.center_on_point` | Center a viewer pane on a 3D volume point. |
| `vc3d_zoom_viewer` | `viewer.zoom` | Multiply a viewer's zoom by a factor (>1 zooms in). |
| `vc3d_enable_editing` | `segmentation.enable_editing` | Turn segmentation editing mode on/off for the active segment. |
| `vc3d_grow_segment` | `segmentation.grow` | Grow the active segmentation surface. Async: returns a jobId (or, with `wait: true`, blocks and returns the terminal status). |
| `vc3d_grow_patch_from_seed` | `segmentation.grow_patch_from_seed` | Create a new segment by growing a patch from a 3D seed point. Async, same `wait` convenience. |
| `vc3d_commit_points` | `points.commit` | Add annotation points (volume space) to a named collection, optionally with a winding annotation. |
| `vc3d_list_points` | `points.list` | List point collections and their points. |
| `vc3d_open_volume` | `volume.open` | Open a volume package (.volpkg / .volpkg.json / zarr project) and optionally select a volume id. |
| `vc3d_open_catalog_sample` | `catalog.open_sample` | Open an Open Data catalog sample by its manifest sample id. |
| `vc3d_job_status` | `job.status` | Poll a job by id (or the latest job): state, message, console tail. |

### The `wait` convenience

`vc3d_grow_segment` and `vc3d_grow_patch_from_seed` accept an MCP-only
`wait: bool = false` param (not part of the underlying RPC, per SPEC.md
§5). When `true`, the tool call blocks until a `job.progress` notification
with `phase:"finished"` arrives for that job (30-minute cap, per spec),
then returns the terminal `job.status` result inline instead of just the
`jobId`. On timeout it returns the original `{jobId, ...}` result with an
extra `"waitTimedOut": true` field so the caller can fall back to polling
`vc3d_job_status`.

## Ambiguities resolved while implementing (per task instructions)

- **Bridge socket filesystem path**: SPEC.md documents the `QLocalServer`
  *name* convention and the stdout handshake line, but not the concrete
  filesystem path such sockets land at across platforms. Resolved by
  supporting both an explicit path (preferred, unambiguous) and a
  best-effort name-based probe — see "Socket discovery" above.
- **Directory placement**: SPEC.md §6 lists `apps/VC3D/agent_bridge/mcp/`
  as the eventual placement for the MCP server; the task explicitly asked
  for a standalone `tools/vc3d-mcp/` directory instead, so that's what was
  built. Worth reconciling with SPEC.md §6 once this is merged.
- **`vc3d_click`/`vc3d_shift_click` `position` schema**: SPEC.md's RPC
  params allow `Vec3 | {"x","y"}` depending on `space`. Modeled as a plain
  `dict[str, float]` (rather than a strict union type) so the MCP-level
  JSON Schema doesn't over-constrain either shape — matching SPEC.md §5's
  "the MCP server performs no semantic validation beyond schema; the bridge
  is the authority."
