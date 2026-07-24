# vc3d-mcp

`vc3d-mcp` lets an MCP-capable AI assistant work with VC3D: open projects,
inspect volumes and segments, navigate viewers, edit surfaces, run tracing and
flattening jobs, and capture screenshots.

The MCP server is a small Python process. It talks MCP over stdio to the
assistant and talks newline-delimited JSON-RPC to VC3D over a local endpoint.
The bridge is disabled unless VC3D is launched with `--agent-bridge`, so a
normal VC3D session is not exposed accidentally.

## Platform support

| Platform | VC3D connection | Launcher |
|---|---|---|
| Linux | Unix-domain socket | `run.sh` |
| macOS | Unix-domain socket under the user's temporary directory | `run.sh` |
| Windows 10/11 | Local named pipe | `run.ps1` |

All three use Qt's `QLocalServer` in VC3D. Windows support stays on the native
named pipe rather than opening a TCP port. Python's Windows event loop provides
the matching asynchronous pipe transport, so there is no platform-specific
runtime dependency.

The bridge is local to one machine. A Unix socket path or Windows pipe name
cannot be used to connect across machines. Container use requires the MCP
process and VC3D endpoint to be visible in the same environment.

## Quick start

You need:

- a built or installed VC3D;
- Python 3.10 or newer;
- an MCP client such as Claude Code, Claude Desktop, or another client that can
  launch a stdio server.

The launchers create an ignored `.venv` and install the Python package on their
first run. Later starts reuse it. Set `PYTHON` if the desired interpreter is
not the default one.

### Linux and macOS

Register the launcher using absolute paths:

```sh
claude mcp add vc3d-bridge -- \
  /path/to/villa/volume-cartographer/tools/vc3d-mcp/run.sh \
  --launch /path/to/VC3D
```

For a macOS application bundle, the binary is normally:

```text
/Applications/VC3D.app/Contents/MacOS/VC3D
```

### Windows

Run the PowerShell launcher from the MCP client:

```powershell
claude mcp add vc3d-bridge -- powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  C:\path\to\villa\volume-cartographer\tools\vc3d-mcp\run.ps1 `
  --launch C:\path\to\VC3D.exe
```

The same configuration in an MCP JSON file looks like:

```json
{
  "mcpServers": {
    "vc3d-bridge": {
      "command": "powershell.exe",
      "args": [
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "C:\\path\\to\\vc3d-mcp\\run.ps1",
        "--launch",
        "C:\\path\\to\\VC3D.exe"
      ]
    }
  }
}
```

Once the MCP server is connected, try:

```text
List the VC3D catalog samples that are available to open.
Open PHercParis4 and show me its volumes.
Take a screenshot of the current VC3D window.
```

## Connecting to VC3D

The MCP server chooses one connection in this order:

1. `--socket <endpoint>` or `VC3D_AGENT_BRIDGE_SOCKET`;
2. the newest compatible running VC3D in
   `~/.vc3d/agent_bridge/`;
3. a VC3D it can launch from `--launch`, `VC3D_BINARY`, `VC3D` on `PATH`, or a
   standard repository build.

This means the quick-start configuration can do both common jobs: it attaches
to an existing bridge when one is running, otherwise it launches VC3D.

To preload a project during auto-launch:

```sh
./tools/vc3d-mcp/run.sh \
  --launch /path/to/VC3D \
  --volpkg /data/example.volpkg
```

The PowerShell launcher accepts the same arguments:

```powershell
.\tools\vc3d-mcp\run.ps1 `
  --launch C:\path\to\VC3D.exe `
  --volpkg D:\data\example.volpkg
```

To start VC3D yourself:

```sh
VC3D --agent-bridge
```

or choose a predictable local name:

```sh
VC3D --agent-bridge-name my-session
```

On success, VC3D prints:

```text
VC3D-AGENT-BRIDGE: listening name=vc3d-agent-<pid> path=<native-endpoint>
```

On Unix, an explicit `--socket` value may be the full socket path from this
line. On Windows it may be a full `\\.\pipe\...` path or the bare server name.
Auto-discovery and auto-launch use the native endpoint directly, so most users
never need to copy it.

### Local access boundary

The bridge grants control of the running application. On Linux and Windows,
VC3D asks `QLocalServer` to restrict access to the current user. On macOS,
Unix-socket permission flags are not enforced by the operating system, so the
socket's per-user temporary directory is the access boundary; do not configure
an explicit endpoint in a shared directory.

Each newline-delimited request is limited to 1 MiB, and an oversized client is
disconnected without affecting other clients.

## Manual Python setup

If you prefer to manage the environment yourself:

```sh
cd tools/vc3d-mcp
python3 -m venv .venv
./.venv/bin/pip install -e .
./.venv/bin/python -m vc3d_mcp --launch /path/to/VC3D
```

On Windows:

```powershell
cd tools\vc3d-mcp
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m vc3d_mcp --launch C:\path\to\VC3D.exe
```

`python -m vc3d_mcp` is a stdio MCP server. Running it in a terminal will look
idle after it connects; normally the MCP client owns that process.

## Troubleshooting

### No bridge could be found

Either pass `--launch /path/to/VC3D`, or start VC3D with `--agent-bridge`.
A VC3D opened normally does not publish the automation endpoint.

### The launcher keeps rebuilding `.venv`

The environment is missing either this package or the MCP SDK. Remove
`tools/vc3d-mcp/.venv` and run the launcher once more. The first run needs
network access to install dependencies.

### Windows blocks `run.ps1`

Configure the MCP client to invoke:

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\...\run.ps1
```

This changes policy only for that PowerShell process.

### The wrong VC3D instance was selected

Pass the exact endpoint with `--socket`, or close the other bridge-enabled
instances. Discovery tries records newest-first and skips endpoints that do
not respond or use an incompatible protocol version. It leaves valid registry
records in place because a busy VC3D can time out temporarily.

### Large screenshots fail

Inline screenshots cross the local bridge as base64. For large captures, pass
`file_path` to `vc3d_screenshot` and let VC3D write the PNG directly.

## What the tools cover

Tool names use a `vc3d_` prefix and snake_case. The surface includes:

- project creation, attached volumes, and the Open Data catalog;
- viewer navigation, screenshots, render settings, and volume overlays;
- segment loading, growth, editing, masks, refinement, and review;
- points, tags, winding annotations, seeding, and push/pull;
- fiber annotation, Atlas search, and Lasagna workflows;
- tracing, flattening, rendering, jobs, progress, and cancellation.

The MCP client displays each installed tool with its current description and
input schema; that generated list is the useful reference and cannot drift
from the implementation. The C++ descriptors and
[`SPEC.md`](../../apps/VC3D/agent_bridge/SPEC.md) contain the full wire-level
reference.

RPC errors surface as MCP tool errors containing the bridge's JSON
`{"code", "message", "data"}` object. Coordinates are volume-space voxels
unless a tool takes an explicit `space`.

### Waiting for jobs

Long-running tools accept `wait`. With `wait=false`, they return a job id
immediately. With `wait=true`, the MCP server streams ordered, best-effort
progress and returns the bridge's authoritative terminal status.

The wait path is bounded to 30 minutes. On expiry it returns the latest result
with `"waitTimedOut": true`; use `vc3d_job_status` or `vc3d_wait_job` to
continue. Progress delivery is observational: a slow or failing MCP progress
sink cannot change the job's terminal result, and task cancellation still
propagates.

Only asynchronous tools expose `wait`; its presence in the generated input
schema is authoritative.

## Development

The C++ method descriptors are the bridge contract. `rpc.describe` exposes
them, the checked-in description snapshot records them, and the Python tests
compare FastMCP schemas and SPEC mappings against that snapshot. Do not add a
second hand-maintained contract.

The Python layout is deliberately explicit:

```text
vc3d_mcp/
  bridge_client.py   JSON-RPC requests, responses, and progress
  transport.py       platform-neutral local stream boundary
  windows_pipe.py    Windows proactor named-pipe connection
  core.py            shared MCP call and wait behavior
  tools/             typed tools grouped by domain
  server.py          discovery, auto-launch, and stdio entrypoint
tests/
  support/           shared fake bridge implementations
  tools/             focused domain tests
```

Run the host suite:

```sh
cd tools/vc3d-mcp
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -v
```

The Unix tests use AF_UNIX fake servers. The Windows suite checks the proactor
stream boundary and verifies that closing a live operating-system pipe
connection completes. Windows CI also launches the freshly built VC3D, calls
`ping` and `rpc.describe` through its Qt named pipe, and checks that descriptor
coverage is complete.

For the wire protocol, limits, method descriptions, and error model, see
[`SPEC.md`](../../apps/VC3D/agent_bridge/SPEC.md).

## Known schema exception

`vc3d_click` and `vc3d_shift_click` accept a `position` whose shape depends on
`space`: either a volume `Vec3` or scene `{"x", "y"}`. The MCP schema leaves it
as `dict[str, float]` so it does not reject either valid shape before the bridge
can perform semantic validation.
