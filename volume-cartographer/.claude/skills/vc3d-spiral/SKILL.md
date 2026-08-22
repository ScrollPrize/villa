---
name: vc3d-spiral
description: Operate VC3D's existing Spiral service through MCP: use saved profiles, connect/reconnect, inspect the advertised dataset, upload and commit inputs, run/stop iterations, export previews, and save/download/load checkpoints. Load before vc3d_spiral_* calls.
---

# Spiral service

Assume `vc3d-bridge-session`. Spiral is an external authenticated service;
VC3D is its client.

## Establish the workspace and connection

1. Open or create a project. Spiral status is unavailable until VC3D has
   constructed the project-scoped Spiral workspace.
2. Call `vc3d_spiral_list_profiles`. MCP connects only through profiles already
   saved by the GUI; it does not create ephemeral profiles or accept endpoint
   credentials.
3. Call `vc3d_spiral_connect(profile_id=...)`, then poll
   `vc3d_spiral_status` until `ready` or `failed`.
4. For a direct profile, provide `SPIRAL_API_KEY` in VC3D's environment. Local
   and SSH profiles use their existing GUI configuration and transport.
5. Call `vc3d_spiral_get_dataset` and verify the service-advertised dataset
   before uploading or running anything.

`vc3d_spiral_disconnect(force=true)` is required when VC3D owns the local
service. `vc3d_spiral_reconnect` uses the current profile.

## Inputs and fitting

- Upload a real absolute path with `vc3d_spiral_upload_input`: a patch
  directory, fiber JSON, or PCL file. Use safe explicit input ids.
- Uploaded inputs are ephemeral until `vc3d_spiral_commit_inputs`.
  `vc3d_spiral_remove_input` removes an uncommitted input.
- `vc3d_spiral_rebuild` requires `confirm=true` and exactly one of service
  defaults or a GUI-compatible request object.
- `vc3d_spiral_run` requires positive iterations. `vc3d_spiral_stop` requests
  stop after the current iteration.
- These calls reuse `SpiralServiceManager`; they are not VC3D jobs and expose
  no operation id, event cursor, wait flag, or cancel-operation RPC. Observe
  connection/session state with `vc3d_spiral_status`, dataset state, and the
  Spiral workspace.

## Preview and checkpoints

- `vc3d_spiral_export_preview` requests the current service preview.
- Save a service-host checkpoint with `vc3d_spiral_save_checkpoint`.
- Download to an absolute local path whose parent exists with
  `vc3d_spiral_download_checkpoint`.
- Load exactly one advertised host checkpoint or existing absolute local file
  with `vc3d_spiral_load_checkpoint`; use `allow_rebuild` only when authorized.

Use `vc3d_switch_workspace(name="spiral")` for GUI evidence, then restore
`main`. For standalone checkpoint-to-TIFXYZ export, use
`vc3d-spiral-checkpoint-flattening`; it is not an MCP `flatten.*` job.
