---
name: vc3d-capability-boundary
description: What the VC3D agent bridge cannot do. Load before promising outcomes involving GUI-only operations, undo, keyboard shortcuts, brushes, approval masks, direction fields, neural tracing, fiber control editing, or generic CLI execution.
---

# VC3D capability boundary

The generated contract is authoritative. If a named tool is absent, do not
invent a plausible call or substitute an adjacent outcome.

## Common boundaries

| Requested action | Reachable alternative |
|---|---|
| Merge or patch surfaces | Use the repository `vc_tifxyz_*` CLI outside MCP. |
| General undo/revert | None. Make a scoped backup before destructive edits. |
| Directly add/move/remove fiber controls | Use ordinary line-annotation pane clicks; no structured control-edit RPC. |
| Paint approval masks | None. |
| Record an Atlas intersection verdict | Search and report candidates as unadjudicated. |
| Close a workspace | Switch to `main`; no close RPC. |
| Run an arbitrary command | Use the shell; there is no generic command RPC. |

A first refusal is usually a missing precondition, not a capability boundary.
Inspect the structured error and state, retry across the relevant viewer,
coordinate, or selection, and only then report a blocker.

## Input and editing gaps

There is no keyboard-input RPC. Keyboard-only undo, composite view, brush
modes, z-offset, and shortcut gestures remain unreachable unless a dedicated
tool exists.

Segmentation supports the registered editing, growth, manual-add,
corrections, push/pull, save, mask, crop, area, reoptimization, tracing,
rendering, and flattening calls. It does not expose:

- approval-mask editing;
- direction-field management;
- the neural-tracer panel;
- smoothing controls;
- draw-mask brush configuration;
- the full growth-panel parameter set.

There is no general undo. Disabling editing does not roll back growth or
corrections.

## Fiber boundary

MCP can launch/open workspaces, set follow, save/list/tag/delete fibers,
create an Atlas, and import/export. It cannot read in-progress workspace
geometry, set tracing configuration, set an arbitrary along-line position,
show a saved fiber in Fiber Slice, set fiber overlay controls, or directly
add/move/remove a control point.

The native trace metric is the standalone `vc_fiber_trace_metric` CLI, not an
MCP method.

## Viewer and workspace boundary

`vc3d_switch_workspace` accepts `main`, `lasagna`, `spiral`, and
`fiber_slice`. Atlas, Intersections, and Line Annotation open as side effects
of their workflow calls. No RPC closes a tab.

Overlay compositing is not the GUI's base composite-surface view. There is no
absolute zoom-to-fit/reset-view RPC, transforms panel, Ink Detection controls,
or navigation-sensitivity control.

## Data and decisions

No RPC exists for Save Project As, detach, Attach Normal Grid, recent projects,
Settings, reload surfaces, cache-folder UI, or standalone normal-grid repair.
Open Data discovery/open and bounded representation selection are supported.

Atlas candidate adjudication and reserved fiber review verdicts stay human.
Segment review tags are supported.

Standalone programs under `apps/src` and
`scripts/spiral/flatten_spiral_checkpoint.py` are shell workflows. The latter
has the separate `vc3d-spiral-checkpoint-flattening` skill and is not a fourth
MCP flatten method.

When documentation and tools disagree, compare `vc3d_ping`, live
`rpc.describe`, and the checked-in snapshot, then restart a stale MCP process
before declaring a method unavailable.
