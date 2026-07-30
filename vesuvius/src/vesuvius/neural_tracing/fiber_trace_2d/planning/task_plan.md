# VC3D Fiber-Global Tracing Mode Plan

## 1. Persisted Mode Contract

- Add a small `FiberOptimizationMode` enum with canonical JSON values
  `lasagna` and `native_fiber_trace3d` in the existing line-annotation fiber
  state helper.
- Store the mode on live sessions and `StoredFiber`, copy it through every
  session/snapshot/import/export path, and write it as top-level
  `optimization_mode` in `vc3d_fiber` JSON.
- Read a missing field as `lasagna` for existing version-1/version-2 files and
  reject unknown/non-string values.
- Keep per-span `segment_to_next` metadata unchanged: it records successful
  native interpolation, while the fiber-wide mode records the requested default
  algorithm for future edits/rebuilds.

## 2. Dialog Controls

- Add a compact mode combo with `Lasagna` and `Fiber model` options, a setter
  used while opening a stored fiber, and a change signal to the controller.
- Add an `Extrapolation` base-voxel spin box backed by `QSettings`; default it to
  half of the existing 2400-voxel initial line length so current tail extent is
  preserved.
- Disable both controls while an optimization task runs. Keep the existing
  initial line length control for new single-seed construction.

## 3. Shared Native Extrapolation

- Add a focused shared helper in `vc_fiber_tracer` that traces from an endpoint
  in an outward direction to a plane at the requested working-voxel distance.
- Build one named target plane perpendicular to the initial outward tangent,
  use the existing one-way beam tracer and normal-aware scoring, and return the
  crossing-snapped line on success.
- Reject invalid start/direction/distance explicitly. Do not add an alternate
  stepping implementation.

## 4. Mixed Whole-Fiber Task

- Extract the current GUI native segment trace body into a reusable task-local
  helper that returns replacement geometry and metadata without touching GUI
  state.
- Add one mixed optimization task that operates on a snapshot:
  1. Resolve adjacent CP spans in line order.
  2. Preserve already valid native spans for ordinary edits, or clear/retrace
     every span for a full fiber-mode rebuild.
  3. Trace each required span independently with the shared native segment
     tracer and retain successful geometry/metadata.
  4. Stitch successful native replacements with existing geometry.
  5. Run the shared Lasagna full reinitializer once, marking native spans as
     protected. This rebuilds only failed/untraced spans and naturally passes
     protected native endpoint directions into neighboring Lasagna candidates.
  6. Replace each Lasagna open tail with bounded native extrapolation when that
     tail succeeds; retain the Lasagna tail on native failure.
- Return one atomic result with per-span success/fallback reporting. Never
  partially apply task output.

## 5. Controller Integration

- On a mode combo change, block if another task is active, prepare both required
  datasets for native mode, and launch a full rebuild. Lasagna mode clears all
  native span metadata before the existing full reinitializer; fiber mode runs
  the mixed task with every span marked for retracing.
- Make the existing full-reoptimization button dispatch by fiber mode.
- After CP insertion/move/deletion in fiber mode, run the mixed task without
  clearing unrelated `segment_to_next` records; only invalidated adjacent spans
  are retraced and individually fall back.
- Keep manual Ctrl-right-click trace/revert actions available and ensure their
  results do not silently change the fiber-global mode.
- Revert the combo/session mode if a mode-change task fails; save mode only with
  successfully applied geometry.

## 6. Tests

- Add helper tests for mode JSON values, missing-field default, and invalid
  values.
- Add native extrapolation tests for requested distance, exact start,
  crossing-snapped endpoint, and invalid input.
- Add mixed-orchestration seams/tests covering all-native success, one native
  failure with Lasagna fallback, successful-neighbor protection/direction
  continuation, preservation of unrelated native spans, and tail fallback.
- Add dialog tests for mode selection, signal/setter behavior, extrapolation
  setting, and busy-state disabling where the existing Qt test harness permits.
- Build affected targets and VC3D with `-j32`; run `test_fiber_trace3d`,
  `test_lasagna_line_optimizer`, `test_line_annotation_generated_views`, and
  relevant Python fiber JSON/sync/merge regressions.

## 7. Spec Update

- Define the persisted fiber-global optimization mode and default.
- Define full rebuild behavior on mode switch, invalid-span retracing on edits,
  per-span native-to-Lasagna fallback, and native-tail-to-Lasagna fallback.
- State that protected native spans supply endpoint continuation directions to
  neighboring Lasagna spans through the shared reinitializer.
- Distinguish base-voxel extrapolation distance from trace-coordinate stepping.

## 8. Docs Updates

- Update `docs/code_structure.md` with the dialog/controller mode flow, mixed
  task, per-span metadata relationship, and open-tail handling.
- Update the VC3D fiber JSON documentation with `optimization_mode`.

## 9. Changelog

- Add a 2026-07-30 entry for fiber-global Lasagna/native modes, per-span
  fallback, trained-neighbor continuation, and configurable extrapolation.

## 10. Review And Risks

- Directly review against `specs.md`, current CP metadata invalidation rules,
  and the shared tracer/reinitializer. Independent-agent review remains
  unavailable because delegation was not requested.
- Primary risks are stale CP indices after stitching, loss of protected-span
  metadata, mode/UI rollback on task failure, and mixing base/trace distances.
  Resolve CP anchors after every reconstruction and cover coordinate conversion
  and transaction behavior in focused tests.
