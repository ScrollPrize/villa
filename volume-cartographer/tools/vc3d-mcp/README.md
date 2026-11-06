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

All three use Qt's `QLocalServer`. On Windows, VC3D keeps the native named
pipe instead of opening a TCP port. Python's Windows event loop can use that
pipe directly, so the MCP needs no Windows-only runtime dependency.

The bridge only connects processes on the same machine. If either process runs
in a container, both must still have access to the same socket or pipe.

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
  --project /data/example.vcproj
```

The PowerShell launcher accepts the same arguments:

```powershell
.\tools\vc3d-mcp\run.ps1 `
  --launch C:\path\to\VC3D.exe `
  --project D:\data\example.vcproj
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
instances. Discovery checks the newest records first and skips endpoints that
do not respond or use an incompatible protocol version. It leaves valid
registry records in place because a busy VC3D can time out temporarily.

### Large screenshots fail

Inline screenshots cross the local bridge as base64. For large captures, pass
`file_path` to `vc3d_screenshot` and let VC3D write the PNG directly.

## What the tools cover

Tool names use a `vc3d_` prefix and snake_case. The surface includes:

- project creation and Save As, attached volumes, normal grids, Lasagna
  datasets and umbilici, store attachment and routing, and the Open Data
  catalog;
- viewer navigation, mouse, keyboard and wheel input, workspaces,
  screenshots, render settings, volume overlays, and the ink detection
  overlay;
- segment loading, growth, editing, editing settings, refinement, transforms,
  exports, neighbor copy, merge and patch, and review;
- points, tags, winding annotations, seeding, and push/pull;
- fiber annotation and Lasagna workflows;
- importing tifxyz surfaces, fibers and point collections into the project's
  object stores, and exporting objects as native files;
- tracing, flattening, rendering, jobs, progress, and cancellation.

The MCP client displays each installed tool with its current description and
input schema; that generated list is the useful reference and cannot drift
from the implementation. The C++ descriptors and
[`SPEC.md`](../../apps/VC3D/agent_bridge/SPEC.md) contain the full wire-level
reference.

RPC errors surface as MCP tool errors containing the bridge's JSON
`{"code", "message", "data"}` object. Coordinates are volume-space voxels
unless a tool takes an explicit `space`.

### Importing and exporting objects

Surfaces, fibers and point collections live in the project's `.vcobj` stores.
`vc3d_get_state` lists the store references (`stores`) and the store each
kind is shown from and edited in (`routes`).

Load a folder of tifxyz segments onto the open volume:

```text
vc3d_import_objects(kind="surface", path="/data/paths", bind={"scale": 1}, wait=True)
vc3d_list_segments()                    # rows carry id (UUID), store, writable
vc3d_activate_segment(segment_id=<uuid>)
vc3d_screenshot()
```

- Import copies the geometry into the store; the source files are not
  referenced afterwards, so a large folder needs room for its copy.
- `bind.scale` is stored units per *current-volume* voxel. There is no default
  scale: without `bind`, sources keep the coordinates they declare, and a
  source that declares none stays unbound and is not shown on a volume.
- Import writes to the store routed for the kind unless `store` names another
  attached, writable store, and never changes routing. The job result's
  `store` says where the objects went; route that store for the kind before
  listing, activating or editing them through the kind's tools. Edits of an
  existing object always go to the store that holds it.
- The job result lists the new `uuids`, identities that were `existing`
  already, and `problems`. A finished import succeeds even when some sources
  were problems; check the lists. `vc3d_cancel_job` stops an import between
  objects, keeping what was committed.

Sources without canonical UUIDs (VC3D-era tifxyz directories named like
`auto_grown_...`, older fibers and point collections) need `uuid_namespace`:
one fresh UUID per dataset, recorded and reused for every later import of the
same files, so a retry reports the objects as `existing` instead of
duplicating them. There is no default namespace, because the names identities
are derived from repeat across datasets. A legacy `.volpkg`:

```text
vc3d_create_project(path="/work/p.vcproj", volume="<volpkg>/volumes/<volume>")
vc3d_open_volume(path="/work/p.vcproj")
vc3d_import_objects(kind="surface", path="<volpkg>/paths",
                    uuid_namespace="<one fresh UUID for this volpkg>",
                    bind={"scale": 1}, wait=True)
```

Export one object by UUID: a segment id from `vc3d_list_segments`, or the
`uuid` that `vc3d_fiber_list` and `vc3d_list_points` report next to the
numeric fiber and collection ids. Create or edit, list to learn the UUID and
store, then export:

```text
vc3d_fiber_list()                       # {fiberId, uuid, store, ...}
vc3d_export_object(uuid=<uuid>, path="/data/out/fiber.json", wait=True)
```

Export searches only stores attached to the project (read-only ones
included); when the UUID is in more than one, pass `store`. A surface becomes
a tifxyz directory, a fiber one fiber JSON, a point collection a
`vc_pointcollections` JSON holding it; the parent directory must exist.
Pending edits of the object are saved first. Files are written in the stored
frame with the object's `uuid` and `coordinates: {volume_uuid,
scale_factor}`: stored XYZ divided by `scale_factor` is voxels of that volume,
and unknown values stay null. Importing the files again (without `bind` or
`uuid_namespace`) keeps UUID and coordinates; an explicit `bind` that
conflicts with them is reported as a problem and imports nothing for that
source.

`vc3d_fiber_import` was removed. Use
`vc3d_import_objects(kind="fiber", path=...)`: fibers without a UUID need
`uuid_namespace` (the bridge no longer names them after their file), and a
fiber file that declares no coordinates stays unbound unless `bind` is given.
`vc3d_fiber_export` still writes every fiber of the fiber store as one
collection file; `vc3d_export_object` writes one fiber.

### Projects, stores and resources

A project can start without a volume:
`vc3d_create_project(path=...)` reports `"volume": null`, and segments are
listed and rendered once a volume is attached (`vc3d_attach_volume`).
`vc3d_get_state` reports the project's `stores`, `routes`, its detachable
`project.entries`, the current volume's `umbilicus`, and the `author` edits
are recorded under (`authorConfigured` says whether the username setting
names one).

Use another store for one kind of object:

```text
vc3d_attach_store(path="/data/team.vcobj")            # or vc3d_create_store(path=...)
vc3d_route_store(kind="surface", store_id=<storeId>)  # one kind per call
vc3d_route_store(kind="surface", store_id=None)       # stop showing surfaces
vc3d_detach_store(store_id=<storeId>)                 # drop the reference
```

Attaching and creating never change routing, and each route call persists
on its own. Routing chooses what is shown and where new objects go; edits of
an existing object stay in the store holding it. Route, detach and Save As
first save pending edits (point collections, the open segment, idle fiber
sessions) and are refused with -32004 while growth or a line optimization
runs or a save fails; nothing changes then.

Attach and detach project resources:

```text
vc3d_attach_normal_grid(location="/data/normal_grids")
vc3d_attach_lasagna_manifest(location="s3://.../x.lasagna.json", role="fiber", wait=True)
vc3d_get_state()                                       # project.entries: {kind, location, scope}
vc3d_detach_project_entry(location=<an entries location>)
vc3d_save_project_as(path="/work/copy.vcproj")         # shares the stores
```

Detaching an entry or a store removes the project's reference only: the
volume, grid, dataset or `.vcobj` and its objects stay where they are.
Removing an umbilicus is different:

```text
vc3d_import_umbilicus(path="/data/umbilicus.json")    # -32602 with data.inferred when the frame must be confirmed
vc3d_import_umbilicus(path="/data/umbilicus.json", units_per_voxel=1, replace=True)
vc3d_remove_umbilicus()                                # retires it in the store, for every project using it
vc3d_route_store(kind="umbilicus", store_id=None)      # hide umbilici in this project only
```

### Waiting for jobs

Long-running tools accept `wait`. With `wait=false`, they return a job id
immediately. With `wait=true`, the MCP server streams ordered, best-effort
progress and returns the bridge's authoritative terminal status.

The wait path is bounded to 30 minutes. On expiry it returns the latest result
with `"waitTimedOut": true`; use `vc3d_job_status` or `vc3d_wait_job` to
continue. Progress delivery is observational: a slow or failing MCP progress
sink cannot change the job's terminal result, and task cancellation still
propagates.

Only asynchronous tools expose `wait`; check the generated input schema to see
whether a tool supports it.

### Seeing the result of an action

The interaction tools (click, drag, key, wheel, center, zoom, rotate, and
workspace switch) accept `observe`: `state` adds a JSON Patch diff of
`vc3d_get_state` from just before to just after the call to the result. Their
results stay plain JSON objects with the same output schema; `observe`
defaults to `none`, which returns them unchanged. `vc3d_observe` shows what
changed since the last observation, including changes a person made in the
GUI, and with `mode` `screenshot` or `both` also returns an image.
Observation bookkeeping belongs to the server process, which serves one MCP
client over stdio.

## Development

The C++ method descriptors are the bridge contract. `rpc.describe` exposes
them, the checked-in description snapshot records them, and the Python tests
compare FastMCP schemas and SPEC mappings against that snapshot. Do not add a
second hand-maintained contract.

The Python layout is deliberately explicit:

```text
vc3d_mcp/
  bridge_client.py   JSON-RPC requests, responses, and progress
  transport.py       selects and opens the local endpoint
  windows_pipe.py    opens Windows named-pipe streams
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

The Unix tests use AF_UNIX fake servers. The Windows suite exercises the
named-pipe client and verifies that a live pipe connection closes cleanly.
Windows CI also launches the freshly built VC3D, calls `ping` and
`rpc.describe` through its Qt named pipe, and checks that descriptor coverage
is complete.

For the wire protocol, limits, method descriptions, and error model, see
[`SPEC.md`](../../apps/VC3D/agent_bridge/SPEC.md).

## Known schema exception

`vc3d_click` and `vc3d_shift_click` accept a `position` whose shape depends on
`space`: either a volume `Vec3` or scene `{"x", "y"}`. The MCP schema leaves it
as `dict[str, float]` so it does not reject either valid shape before the bridge
can perform semantic validation.
