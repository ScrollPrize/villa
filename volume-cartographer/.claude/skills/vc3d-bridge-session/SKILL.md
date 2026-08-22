---
name: vc3d-bridge-session
description: Start and operate a VC3D MCP session safely: identify the responding build, inspect live state, manage jobs, recognize bridge errors, satisfy editing preconditions, and leave persistent modes clean. Load before the first vc3d_* call in every session.
---

# VC3D bridge session

Use the exposed MCP tool names; per-call schemas belong to their tool
descriptions.

## Start

1. Call `vc3d_ping`; record pid, executable path, git SHA, application version,
   and protocol version when provenance matters.
2. Call `vc3d_get_state`; record package, volume, active segment, viewers,
   editing state, and jobs.
3. Load the workflow skill before mutating state.

Discovery records identify a process/socket, not its executable or revision.
Live `rpc.describe` describes the running binary; the checked-in
`rpc_description.json` is the repository snapshot.

## Route

- Projects/catalog: `vc3d-open-data`
- Navigation/capture: `vc3d-visual-evidence` and `vc3d-reading-the-image`
- Segments/editing/seeding: `vc3d-segment-lifecycle`,
  `vc3d-segmentation-editing`, `vc3d-seeding`
- Points/winding: `vc3d-winding-annotation`
- Fiber/Lasagna/Atlas/Spiral: their matching `vc3d-*` skills
- Rendering/flattening: `vc3d-rendering`, `vc3d-flattening`
- Unsupported GUI actions: `vc3d-capability-boundary`

## State and jobs

Viewer ids are process-local; re-read state after opening data or changing
workspaces. Open a package and select a volume before workspace-scoped calls.

Job sources are `tool`, `growth`, `lasagna`, `atlas`, `catalog`, `volume`,
`flatten`, `seeding`, and `autosave`. Retain the launch `jobId`; use
`vc3d_wait_job` or `vc3d_job_status` until terminal. A wait timeout does not
cancel work. Spiral operations are service calls, not VC3D jobs.

For segment editing: activate, enable editing, mutate, save, then disable.
There is no general undo.

## Errors

- `-32602`: correct the named parameter.
- `-32000` / `-32001`: open a package / select a volume.
- `-32002` / `-32003`: re-resolve the viewer / coordinate.
- `-32004`: poll the busy source.
- `-32005`: preserve service, launch, or I/O detail.
- `-32006`: report the missing executable.
- `-32007`: re-list the requested object.
- `-32008`: establish editing.
- `-32009`: check the capability boundary.
- `-32010`: preserve the internal detail.

## Finish

Save required state; stop transient modes; restore temporary workspace,
overlay, editing, and viewer settings. Never delete or clear without explicit
scope. Report build identity, data identity, mutations, terminal jobs, output
paths, and intentional residual state.

Read [`references/session-field-notes.md`](references/session-field-notes.md)
only when debugging discovery, naming, or job behavior.
