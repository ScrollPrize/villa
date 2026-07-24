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
- **Async jobs**: the bridge is the authoritative source of job state —
  `vc3d_job_status` is answered by the bridge's own `job.status` RPC, not by
  anything in this process. A `wait: true` call subscribes to sequenced
  `job.progress` notifications, replays the server's bounded history, and
  periodically reads `job.status` as a lost-notification and terminal-state
  fallback.

## Layout

```
tools/vc3d-mcp/
  vc3d_mcp/
    bridge_client.py   # asyncio client: socket resolution, JSON-RPC framing,
                        # request/response correlation over one _Conn
    core.py            # shared FastMCP instance + _call / _wait_for_job /
                        # _strip_none used by every tool module
    tools/             # explicit per-domain @mcp.tool() modules; jobs and
                        # session/app-state tools are kept separate
    server.py          # entry/CLI: connection resolution + auto-launch, imports
                        # tools/ to register them on the shared mcp instance
    __main__.py        # `python -m vc3d_mcp` entry point
  tests/
    support/            # shared stateful and extensible AF_UNIX bridge fakes
    tools/              # focused tests for individual MCP tool domains
    test_bridge.py      # bridge transport and core tool round trips
    test_progress.py    # connection lifecycle, progress, waiting, schemas
    test_runtime.py     # discovery, auto-launch, teardown helpers
    test_contract.py    # descriptor snapshot, MCP schema, and SPEC alignment
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
   itself (with `--agent-bridge`) and connects. A daemon thread drains the
   child's stdout (with stderr merged in) for the whole process lifetime: it
   detects the handshake line, signals the launcher with the socket path, and
   keeps reading so VC3D's stdout pipe keeps draining and never blocks VC3D.
   Child log lines are handed to a bounded in-memory queue that a *separate*
   thread forwards to *this* process's stderr only — never stdout, which is
   reserved for the MCP stdio transport. That decoupling is deliberate: if the
   stderr sink stalls (e.g. an MCP host that stops reading the piped stderr),
   the forwarder blocks but the stdout drain does not, so VC3D never blocks on
   its stdout pipe; if the queue fills, the oldest overflow lines are dropped
   best-effort (a bounded tail is still retained for error reporting). The
   launcher waits up to a real wall-clock timeout (default 30 s): a
   silent-but-live child times out at the deadline; a child that exits before
   the handshake fails immediately with its captured log tail. On shutdown the
   child is torn down with escalation (`terminate()`, wait ~5 s, then `kill()`;
   if it still hasn't exited a background thread performs a final blocking
   `wait()` so it is always reaped and never lingers as a zombie), safe to
   invoke repeatedly. The binary is `--launch <path>`, else the `VC3D_BINARY`
   env var, `VC3D` on `PATH`, or a standard CMake preset build under `build/`;
   each candidate must be executable. Pass `--volpkg <path>` to have the
   launched VC3D preload a volume package so the agent's first action need not
   be opening one.
4. Otherwise the server exits with status 2 and a stderr message explaining all
   three options above.

### Zero-setup path: `run.sh`

Building VC3D via CMake only produces the C++ binary — it does nothing to
set up this directory's Python environment. `run.sh` is a self-bootstrapping
wrapper: the **first** time it's launched it creates `.venv` and installs
this package automatically (a few seconds, one-time); every launch after
that is a plain exec with no setup overhead. This means there is no manual
Python step at all between "finished building VC3D" and "an agent can
connect" — just point an MCP client at `run.sh` and it works, including on
a machine where nobody has ever run `pip install` for this package:

```sh
# (a) attach to whatever VC3D is already running (started with --agent-bridge):
./tools/vc3d-mcp/run.sh

# (b) auto-launch VC3D if none is running:
./tools/vc3d-mcp/run.sh --launch ./build-macos/bin/VC3D --volpkg /path/to/foo.volpkg

# (c) attach to an exact socket (unambiguous; e.g. across machines/containers):
./tools/vc3d-mcp/run.sh --socket /var/folders/.../T/vc3d-agent-<pid>
```

`run.sh` accepts the exact same arguments as `python -m vc3d_mcp` below —
it's a drop-in wrapper, not a different interface. Set `PYTHON=/path/to/python3`
if `python3` isn't the right interpreter to bootstrap with.

### Manual path (explicit venv)

If you'd rather manage the environment yourself (e.g. for a reproducible CI
setup, or to pin dependency versions explicitly):

```sh
cd tools/vc3d-mcp
python3 -m venv .venv
./.venv/bin/pip install -e .        # or: pip install -r requirements.txt

./.venv/bin/python -m vc3d_mcp                                    # (a) attach
./.venv/bin/python -m vc3d_mcp --launch ./build-macos/bin/VC3D     # (b) auto-launch
./.venv/bin/python -m vc3d_mcp --socket /var/folders/.../T/vc3d-agent-<pid>  # (c) exact socket
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
# Seamless: no socket needed, no manual venv setup needed — run.sh bootstraps
# its own Python environment on first launch, then attaches to a running
# VC3D or auto-launches one.
claude mcp add vc3d-bridge -- \
  /path/to/tools/vc3d-mcp/run.sh --launch /path/to/build-macos/bin/VC3D

# ...or pin an explicit socket if you prefer:
claude mcp add vc3d-bridge -e VC3D_AGENT_BRIDGE_SOCKET=/path/to/socket -- \
  /path/to/tools/vc3d-mcp/run.sh
```

(If you set up the venv manually per "Manual path" above, swap `run.sh` for
`.venv/bin/python -m vc3d_mcp` in either command.)

(or the equivalent entry in `claude_desktop_config.json` / `.mcp.json` for
other MCP clients.)

### Socket discovery / naming convention

VC3D is started with `--agent-bridge` (default socket name
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

### Socket and request-size restrictions

The bridge is a local-only automation channel. On the C++ side the
`QLocalServer` is created with user-only access, so only the same OS user can
connect. The server also bounds each request: a request line that grows past
the server's maximum without a terminating newline causes that one client to be
disconnected, and an oversized complete line is rejected — one misbehaving
client cannot affect others. Keep individual RPC payloads (e.g. base64
screenshots) within that bound. `vc3d_screenshot` returns the PNG as MCP **image
content** (a FastMCP `Image`, decoded from the RPC's base64) when `file_path` is
omitted, and a dict carrying the on-disk path when `file_path` is set. The base64
still crosses the socket in the image-content case and counts against this bound,
so prefer passing `file_path` for large captures.

## Self-test (no real VC3D required)

The test suite uses a fake JSON-RPC server on a local AF_UNIX socket (standing
in for `AgentBridgeServer`, which speaks the same newline-delimited JSON-RPC
2.0 framing) and checks that:

- `BridgeClient` connects, sends a request, and parses both success and
  JSON-RPC error responses (error `code`/`message`/`data` preserved);
- `wait: true` merges buffered and live `job.progress` updates in sequence,
  with `job.status` as the authoritative terminal fallback;
- the actual `@mcp.tool()` functions — not just the transport layer —
  round-trip correctly, including the `wait: true` behavior on the
  long-running tools.

Run it:

```sh
cd tools/vc3d-mcp
python3 -m unittest discover -v
```

The repository-local `.venv` is runtime convenience state created by
`run.sh`, not part of the source layout. For development and CI, use a clean
external environment such as `/tmp/vcmcp` so stale local environments cannot
affect test results.

## Tool list

One MCP tool per JSON-RPC method documented in `apps/VC3D/agent_bridge/SPEC.md`,
named with a `vc3d_` prefix and snake_case. Coordinates are
volume-space voxels unless a tool takes an explicit `space` field. RPC errors
surface as MCP tool errors whose text is the JSON-encoded
`{"code", "message", "data"}` error object.

| Group | Tools |
|---|---|
| Core / navigation | `vc3d_ping`, `vc3d_get_state`, `vc3d_list_segments`, `vc3d_activate_segment`, `vc3d_screenshot`, `vc3d_get_cursor_point`, `vc3d_click`, `vc3d_shift_click`, `vc3d_drag`, `vc3d_center_viewer`, `vc3d_zoom_viewer`, `vc3d_rotate_viewer`, `vc3d_set_axis_aligned_slices`, `vc3d_get_render_settings`, `vc3d_set_render_settings` |
| Segmentation growth/editing | `vc3d_fetch_segment`, `vc3d_enable_editing`, `vc3d_save_segment`, `vc3d_grow_segment`, `vc3d_grow_patch_from_seed`, `vc3d_delete_segment`, `vc3d_rename_segment`, `vc3d_manual_add_begin`, `vc3d_manual_add_finish`, `vc3d_manual_add_set_line_mode`, `vc3d_manual_add_set_interpolation`, `vc3d_manual_add_undo_constraint`, `vc3d_corrections_set_point_mode`, `vc3d_push_pull_set_config`, `vc3d_push_pull_start`, `vc3d_push_pull_stop` |
| Points / tags / review | `vc3d_commit_points`, `vc3d_list_points`, `vc3d_add_point_collection`, `vc3d_update_point`, `vc3d_remove_point`, `vc3d_clear_point_collection`, `vc3d_clear_all_points`, `vc3d_rename_point_collection`, `vc3d_set_point_collection_color`, `vc3d_set_point_collection_metadata`, `vc3d_set_point_collection_tag`, `vc3d_remove_point_collection_tag`, `vc3d_set_point_windings_linked`, `vc3d_auto_fill_windings`, `vc3d_set_auto_fill_mode`, `vc3d_reset_windings`, `vc3d_apply_anchor_offset`, `vc3d_save_points_json`, `vc3d_load_points_json`, `vc3d_save_points_segment_path`, `vc3d_load_points_segment_path`, `vc3d_set_segment_tag`, `vc3d_review_segments` |
| Wrap annotation | `vc3d_set_wrap_annotation_mode`, `vc3d_commit_wrap_annotation`, `vc3d_undo_wrap_annotation` |
| Volumes / catalog | `vc3d_open_volume`, `vc3d_list_attached_volumes`, `vc3d_open_catalog_sample`, `vc3d_list_catalog_samples`, `vc3d_describe_catalog_sample`, `vc3d_select_volume` |
| Lasagna / workspace | `vc3d_lasagna_service_status`, `vc3d_lasagna_ensure_service`, `vc3d_lasagna_list_datasets`, `vc3d_lasagna_start_optimization`, `vc3d_lasagna_jobs`, `vc3d_lasagna_cancel`, `vc3d_lasagna_select_output`, `vc3d_lasagna_repeat_last`, `vc3d_switch_workspace` |
| Atlas | `vc3d_atlas_open`, `vc3d_atlas_status`, `vc3d_atlas_search_start`, `vc3d_atlas_search_cancel`, `vc3d_atlas_search_results`, `vc3d_atlas_open_result`, `vc3d_atlas_remap`, `vc3d_atlas_optimize_snap_candidates` |
| Line Annotation (fiber tracing) | `vc3d_fiber_launch`, `vc3d_fiber_list`, `vc3d_fiber_open`, `vc3d_fiber_set_follow`, `vc3d_fiber_save`, `vc3d_fiber_delete`, `vc3d_fiber_set_tag`, `vc3d_fiber_create_atlas`, `vc3d_fiber_export`, `vc3d_fiber_import` |
| Seeding | `vc3d_seeding_set_winding_annotation_mode`, `vc3d_seeding_preview_rays`, `vc3d_seeding_cast_rays`, `vc3d_seeding_reset_points`, `vc3d_seeding_run`, `vc3d_seeding_expand`, `vc3d_seeding_cancel`, `vc3d_seeding_analyze_paths` |
| Tracer / flatten / render | `vc3d_run_trace`, `vc3d_render_tifxyz`, `vc3d_flatten_slim`, `vc3d_flatten_abf`, `vc3d_flatten_straighten` |
| Jobs | `vc3d_job_status`, `vc3d_wait_job`, `vc3d_cancel_job` |

### The `wait` convenience

The wait-capable tools accept an MCP-only `wait: bool` param that is not part of
the underlying RPC. The full set is:

- `vc3d_fetch_segment` (defaults to `wait=true`) and `vc3d_open_catalog_sample`,
- `vc3d_grow_segment`, `vc3d_grow_patch_from_seed`, `vc3d_run_trace`,
- `vc3d_atlas_search_start`,
- `vc3d_lasagna_start_optimization`, `vc3d_lasagna_repeat_last`,
- `vc3d_seeding_run`, `vc3d_seeding_expand`,
- `vc3d_render_tifxyz`, `vc3d_flatten_slim`, `vc3d_flatten_abf`,
  `vc3d_flatten_straighten`.

(`vc3d_save_segment` also takes `wait`, defaulting to `true`.) All other
wait-capable tools default to `wait=false`.

When `true`, the tool subscribes before its first status read, replays the
server's bounded progress history, and streams new notifications by sequence.
It periodically reads `job.status` to recover from notification loss and returns
that authoritative terminal result (30-minute cap). On timeout it returns the original
`{jobId, ...}` result with an extra `"waitTimedOut": true` field so the caller
can fall back to polling `vc3d_job_status`; on a bridge disconnect the next poll
raises and the call fails promptly rather than blocking to the cap.

Progress is **ordered and best-effort** for the lifetime of the wait call. The
bridge retains the latest 64 updates and the Python queue is bounded to the same
size. Output updates are forwarded as text; completion is forwarded as compact
structured data. Reporting is observational: an unsupported context is a no-op,
and a failing or stalled sink is abandoned after one bounded attempt without
changing execution or the terminal result. Task cancellation still propagates.
The injected `Context` is not part of any tool's input schema.

`vc3d_wait_job` is the standalone counterpart to `wait`: it has
no underlying RPC and blocks on an **already-running** job by id, using the same
notification/replay/status-fallback loop and returning the terminal `job.status`
inline — or the
last status with `"waitTimedOut": true` on the cap. Use it to wait on a job that
some earlier call started with `wait=false`, or on an externally-initiated job seen
in `vc3d_get_state`. To stop a running job, `vc3d_cancel_job` maps to
`job.cancel`: it is request-only (poll `vc3d_job_status` for the terminal state), and
only `tool`/`atlas`/`seeding`/`lasagna` jobs are cancellable — `growth`/`flatten`/
`catalog`/`autosave` jobs return an error.

## Known gaps

- **`vc3d_click`/`vc3d_shift_click` `position` schema**: SPEC.md's RPC
  params allow `Vec3 | {"x","y"}` depending on `space`. Modeled as a plain
  `dict[str, float]` (rather than a strict union type) so the MCP-level
  JSON Schema doesn't over-constrain either shape. The bridge remains the
  authority for semantic validation.
