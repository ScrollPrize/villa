---
name: vc3d-segmentation-editing
description: Edit a VC3D surface through MCP: establish the active editable segment, grow it, perform manual-add or correction operations, run push/pull safely, verify geometry changes, save, and clean up modes. Load before segmentation editing tools.
---

# Edit a segmentation surface

Assume `vc3d-bridge-session`; use `vc3d-capability-boundary` for GUI-only
operations.

## Establish the target

1. `vc3d_list_segments` and choose by identity, not list position.
2. Materialize an Open Data placeholder with `vc3d_fetch_segment(...,
   wait=true)` before activation.
3. `vc3d_activate_segment`, then `vc3d_enable_editing(enabled=true)`.
4. Re-read `vc3d_get_state` and verify the active id and editing flag.

Changing the segment invalidates the editing context; repeat activation and
enablement. Use L0 volume coordinates and an explicit viewer for gestures.

## Grow and verify

- Launch `vc3d_grow_segment` with an explicit method, direction, and bounded
  step count. Retain and poll its `growth` job.
- `vc3d_grow_patch_from_seed` creates a new segment rather than editing the
  active one. Validate the seed in the selected volume first.
- Before and after growth, record segment identity, grid bounds/dimensions,
  valid-point count or another available geometry measure, and screenshots at
  the same view. A succeeded job alone does not prove useful growth.

## Manual add, corrections, and push/pull

- Manual add: begin, configure line/interpolation mode, place constraints with
  the intended viewer gesture, then finish with `apply=true` or abort with
  `apply=false`. Do not leave the mode active.
- Corrections: enable correction-point mode, place/drag the intended points,
  execute corrections growth, then disable the mode. A click and a drag have
  different semantics; verify the resulting collection.
- Push/pull: set the configuration, start at an on-surface point, allow only a
  bounded interval, and always call `vc3d_push_pull_stop` even after an error.

## Persist and clean up

1. Call `vc3d_save_segment(wait=true)` when the active segment is dirty.
2. Disable any editing or interaction mode enabled for the task.
3. Re-read state and verify the intended segment remains active and no related
   job is running.
4. Report the geometry delta, save result, and limitations. Do not claim that
   an unobservable GUI-only action occurred.

Detailed surface-point derivation, growth measurements, and editing edge cases
are preserved in
[`references/surface-growth-verification.md`](references/surface-growth-verification.md).
