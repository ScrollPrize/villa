# vc3d-mcp

A standalone MCP (Model Context Protocol) server that lets an AI agent (e.g.
Claude Code) drive a running VC3D instance through the **VC3D Agent Bridge**
— the in-process JSON-RPC-over-local-socket automation channel implemented
in `apps/VC3D/agent_bridge/` (see
[`SPEC.md`](../../apps/VC3D/agent_bridge/SPEC.md) for the binding contract).

This directory is a separate Python process. It does not modify or embed any
VC3D C++ code; it only speaks the wire protocol AgentBridgeServer exposes.

Status: fully implemented and verified against a real, running VC3D instance
(offscreen and live/windowed), across every RPC group the bridge exposes —
navigation, segmentation growth/editing, remote catalog control, Lasagna,
Atlas, Line Annotation (fiber tracing), tags/seeding/push-pull, Run-Trace,
flattening, and rendering.

## Implementation notes

- **MCP SDK**: the official `mcp` Python package (the Model Context
  Protocol SDK) is used directly (`mcp.server.fastmcp.FastMCP`, stdio
  transport). No hand-rolled MCP JSON-RPC protocol.
- **Bridge transport**: a small asyncio client (`vc3d_mcp/bridge_client.py`)
  that speaks the bridge's wire format directly over a Unix domain socket
  (`asyncio.open_unix_connection`) — no Qt dependency needed on the Python
  side, since `QLocalServer`/`QLocalSocket` are plain AF_UNIX sockets on the
  Unix platforms VC3D ships for.
- **Async jobs**: `job.progress` notifications (SPEC.md §8.3/§3.18) are
  consumed by a background reader task and folded into a small `JobTracker`
  (`vc3d_mcp/jobs.py`), which backs `vc3d_job_status` and the `wait: true`
  convenience on the long-running grow/flatten/render/atlas-search tools.

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

In the common case you don't have to relay a socket path by hand — the server
finds or launches VC3D for you. It resolves a bridge to connect to in this
priority order:

1. **Explicit socket** — `--socket <name-or-path>` or the
   `VC3D_AGENT_BRIDGE_SOCKET` env var, when set. Highest priority; most
   explicit.
2. **Attach to an already-running VC3D** (zero config) — a VC3D started with
   `--agent-bridge` publishes a small JSON file under `~/.vc3d/agent_bridge/`
   (`<pid>.json` holding `{pid, name, path, startedAt}`). The server scans that
   directory, drops stale entries whose PID is dead (removing their files), and
   attaches to the newest live bridge. This mirrors the `~/.fit_services`
   discovery convention used elsewhere in VC3D
   (`LasagnaServiceManager::discoverServices`).
3. **Auto-launch VC3D** — if nothing is running, the server launches VC3D
   itself (with `--agent-bridge`), reads the handshake line off its stdout, and
   connects. The binary is `--launch <path>`, else the `VC3D_BINARY` env var,
   else the repo-root build `build-macos/bin/VC3D`; the fallback is used only if
   it names a real, executable file. Pass `--volpkg <path>` to have the launched
   VC3D preload a volume package (forwarded as `--load-first`) so the agent's
   first action need not be opening one.
4. Otherwise the server exits with status 2 and a stderr message explaining all
   three options above.

So the simplest invocations are:

```sh
cd tools/vc3d-mcp
python3 -m venv .venv
./.venv/bin/pip install -e .        # or: pip install -r requirements.txt

# (a) attach to whatever VC3D is already running (started with --agent-bridge):
./.venv/bin/python -m vc3d_mcp

# (b) auto-launch VC3D if none is running:
./.venv/bin/python -m vc3d_mcp --launch ./build-macos/bin/VC3D --volpkg /path/to/foo.volpkg

# (c) attach to an exact socket (unambiguous; e.g. across machines/containers):
./.venv/bin/python -m vc3d_mcp --socket /var/folders/.../T/vc3d-agent-<pid>
```

To start VC3D yourself for mode (a) or (c):

```sh
./build-macos/bin/VC3D --agent-bridge
# or, for a predictable socket name (handy for scripting/tests):
./build-macos/bin/VC3D --agent-bridge-name my-session
```

On success it prints exactly one line to stdout (and, for mode (a), publishes
its `~/.vc3d/agent_bridge/<pid>.json` registry entry):

```
VC3D-AGENT-BRIDGE: listening name=vc3d-agent-<pid> path=/var/folders/.../T/vc3d-agent-<pid>
```

The server speaks MCP over stdio, so it's meant to be launched by an MCP
client (e.g. registered as an MCP server in Claude Code's config), not run
interactively.

### Registering with Claude Code

```sh
# Seamless: no socket needed — attaches to a running VC3D, else auto-launches.
claude mcp add vc3d-bridge -- \
  /path/to/tools/vc3d-mcp/.venv/bin/python -m vc3d_mcp --launch /path/to/build-macos/bin/VC3D

# ...or pin an explicit socket if you prefer:
claude mcp add vc3d-bridge -e VC3D_AGENT_BRIDGE_SOCKET=/path/to/socket -- \
  /path/to/tools/vc3d-mcp/.venv/bin/python -m vc3d_mcp
```

(or the equivalent entry in `claude_desktop_config.json` / `.mcp.json` for
other MCP clients.)

### Socket discovery / naming convention

Per SPEC.md §1.1, VC3D is started with `--agent-bridge` (default socket name
`vc3d-agent-<pid>`) or `--agent-bridge-name <name>` (explicit, e.g. for
tests), and on success prints the handshake line shown above.

This MCP server takes the same value via `--socket <name-or-path>` or the
`VC3D_AGENT_BRIDGE_SOCKET` env var, and resolves it to an actual filesystem
path as follows (`BridgeClient.resolve_socket_path`):

1. If the value is a path that already exists on disk, connect to it
   directly as an AF_UNIX socket. **This is the recommended, unambiguous
   mode** — pass the exact `path=...` value from the handshake line.
2. Otherwise, treat the value as a bare `QLocalServer` name (matching
   `--agent-bridge-name`) and probe `$TMPDIR/<name>` then `/tmp/<name>`
   (the conventional places Qt places local-server sockets on macOS/Linux).

SPEC.md does not pin down exact `QLocalServer` filesystem conventions across
platforms/Qt versions, so step 2 is a documented best-effort assumption; step
1 sidesteps the ambiguity entirely.

When you don't pass a socket at all, resolution falls through to auto-discovery
and auto-launch (see "Running it" above). Auto-discovery reads
`~/.vc3d/agent_bridge/<pid>.json` registry files
(`discover_registry_socket`) and uses the authoritative `path` recorded there;
auto-launch parses the launched VC3D's handshake line with
`BridgeClient.socket_path_from_handshake(line)`. Neither path relies on the
best-effort name→path guessing of step 2.

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
  behavior on the long-running tools.

Run it:

```sh
cd tools/vc3d-mcp
./.venv/bin/python test_mcp_bridge.py -v
```

## Tool list

One MCP tool per JSON-RPC method documented in `apps/VC3D/agent_bridge/SPEC.md`,
named per its §5/§16 conventions (`vc3d_` prefix, snake_case). Coordinates are
volume-space voxels unless a tool takes an explicit `space` field. RPC errors
surface as MCP tool errors whose text is the JSON-encoded
`{"code", "message", "data"}` error object.

| Group | Tools |
|---|---|
| Core / navigation | `vc3d_ping`, `vc3d_get_state`, `vc3d_list_segments`, `vc3d_activate_segment`, `vc3d_screenshot`, `vc3d_get_cursor_point`, `vc3d_click`, `vc3d_shift_click`, `vc3d_center_viewer`, `vc3d_zoom_viewer` |
| Segmentation growth/editing | `vc3d_enable_editing`, `vc3d_grow_segment`, `vc3d_grow_patch_from_seed`, `vc3d_manual_add_begin`, `vc3d_manual_add_finish`, `vc3d_manual_add_set_line_mode`, `vc3d_manual_add_set_interpolation`, `vc3d_manual_add_undo_constraint`, `vc3d_corrections_set_point_mode`, `vc3d_push_pull_set_config`, `vc3d_push_pull_start`, `vc3d_push_pull_stop` |
| Points / tags | `vc3d_commit_points`, `vc3d_list_points`, `vc3d_set_segment_tag` |
| Volumes / catalog | `vc3d_open_volume`, `vc3d_open_catalog_sample`, `vc3d_list_catalog_samples`, `vc3d_describe_catalog_sample`, `vc3d_select_volume` |
| Atlas | `vc3d_atlas_open`, `vc3d_atlas_status`, `vc3d_atlas_search_start`, `vc3d_atlas_search_cancel`, `vc3d_atlas_search_results`, `vc3d_atlas_open_result`, `vc3d_atlas_remap`, `vc3d_atlas_optimize_snap_candidates` |
| Line Annotation (fiber tracing) | `vc3d_fiber_launch`, `vc3d_fiber_list`, `vc3d_fiber_open`, `vc3d_fiber_set_follow`, `vc3d_fiber_save`, `vc3d_fiber_delete`, `vc3d_fiber_set_tag`, `vc3d_fiber_create_atlas`, `vc3d_fiber_export`, `vc3d_fiber_import` |
| Seeding | `vc3d_seeding_set_winding_annotation_mode`, `vc3d_seeding_preview_rays`, `vc3d_seeding_cast_rays`, `vc3d_seeding_reset_points` |
| Tracer / flatten / render | `vc3d_run_trace`, `vc3d_render_tifxyz`, `vc3d_flatten_slim`, `vc3d_flatten_abf`, `vc3d_flatten_straighten` |
| Jobs | `vc3d_job_status` |

### The `wait` convenience

Long-running tools (`vc3d_grow_segment`, `vc3d_grow_patch_from_seed`,
`vc3d_run_trace`, `vc3d_render_tifxyz`, `vc3d_flatten_slim`/`_abf`/`_straighten`,
`vc3d_atlas_search_start`) accept an MCP-only `wait: bool = false` param (not
part of the underlying RPC, per SPEC.md §5). When `true`, the tool call
blocks until a `job.progress` notification with `phase:"finished"` arrives
for that job's source (30-minute cap), then returns the terminal
`job.status` result inline instead of just the `jobId`. On timeout it
returns the original `{jobId, ...}` result with an extra
`"waitTimedOut": true` field so the caller can fall back to polling
`vc3d_job_status`.

## Known gaps

- **No `vc3d_lasagna_*` tools yet.** The bridge's `lasagna.*` RPCs (service
  status/ensure, dataset listing, start/cancel optimization, output-segment
  selection, repeat-last, workspace switching) are fully implemented and
  tested in `AgentBridgeServer`/SPEC.md §11, but no MCP wrapper tools were
  added for them — an agent can only reach Lasagna today via a raw JSON-RPC
  call over the bridge socket directly, not through this MCP server. Worth
  closing as a follow-up.
- **Directory placement**: SPEC.md §6 originally described
  `apps/VC3D/agent_bridge/mcp/` as the eventual placement; a standalone
  `tools/vc3d-mcp/` directory was built instead per explicit instruction.
  Not reconciled with SPEC.md §6.
- **`vc3d_click`/`vc3d_shift_click` `position` schema**: SPEC.md's RPC
  params allow `Vec3 | {"x","y"}` depending on `space`. Modeled as a plain
  `dict[str, float]` (rather than a strict union type) so the MCP-level
  JSON Schema doesn't over-constrain either shape — matching SPEC.md §5's
  "the MCP server performs no semantic validation beyond schema; the bridge
  is the authority."
