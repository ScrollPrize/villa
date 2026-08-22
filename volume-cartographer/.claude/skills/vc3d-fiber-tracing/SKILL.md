---
name: vc3d-fiber-tracing
description: Trace, inspect, save, import/export, tag, and delete Fiber3D fibers through VC3D MCP. Load before vc3d_fiber_* calls; combine with vc3d-open-data and vc3d-reading-the-image for remote prediction-backed tracing.
---

# Fiber tracing

Assume `vc3d-bridge-session`. A fiber is one tubular papyrus cell, not a sheet
segmentation.

## Prepare and launch

1. Open a project and select the volume that owns a compatible attached
   Lasagna dataset. For Open Data, select representations by metadata and keep
   the base volume and Lasagna coordinate identity aligned.
2. Derive an L0 seed from base CT and prediction evidence. Do not pass a
   display-level coordinate unchanged.
3. Call `vc3d_fiber_launch` with an explicit viewer and seed. Re-read
   `vc3d_get_state`: line-annotation panes are created asynchronously and their
   viewer ids may change after edits.
4. Add controls with plain `vc3d_click` on a generated line strip
   (`*_line_surface` or `*_line_side_slice`); that click selects an along-line
   position and places the control. A cut-pane click instead uses the already
   current line position. `vc3d_shift_click` invokes a different predicted-snap
   gesture. The current
   contract has no direct control-point add, move, remove, configuration, or
   in-progress workspace-state RPC; use visual evidence and save/list readback.

Use `vc3d_fiber_set_follow` only after a line-annotation workspace is open.
Restore it if it was enabled only for evidence.

## Persist and inspect

- Call `vc3d_fiber_save`, then `vc3d_fiber_list`. Saved fibers are the
  authoritative readback for ids, point counts, spans, trace state, and tags.
- Use `vc3d_fiber_open` with at most one of control-point, line-point, or span
  selectors. Re-read viewer state after opening.
- Use explicit saved ids for `vc3d_fiber_set_tag`,
  `vc3d_fiber_create_atlas`, and `vc3d_fiber_delete`. Deletion is destructive.
- `vc3d_fiber_export` and `vc3d_fiber_import` require paths visible to VC3D;
  pass an explicit scale when it is not 1.0.

The native trace metric is a standalone `vc_fiber_trace_metric` CLI workflow,
not an MCP method. Do not invent `vc3d_fiber_trace_metric` or other absent
fiber calls. For image-based drift checks and metric caveats, read
[`references/drift-and-metrics.md`](references/drift-and-metrics.md).
