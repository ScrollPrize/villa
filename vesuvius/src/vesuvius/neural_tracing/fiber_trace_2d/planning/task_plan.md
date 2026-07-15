# 3D Test Visualization And Raw Index Cleanup Plan

## Current Findings

- `_make_train_sample_3d_sheet(...)` displays `presence_target`, so CP-only
  JSON/test samples show only the supervised CP neighborhood in the target
  presence column even though `target_segment_*` line context is available.
- The previous loader patch exposed two public ways to drive augmentation
  seeding: implicit single-index behavior and an explicit
  `augmentation_start_sample_index`. That is unnecessarily ambiguous for the
  deterministic raw-stream requirement.
- `_write_3d_sample_sheet(...)` only renders the first sample in the batch.
- Dense 3D tests currently default to one batch when `test_control_points` is
  omitted, and the test visualization path still slices the vis batch to one
  sample.

## Implementation

1. Make 3D loader sample-index semantics single-path.
   - Treat public `sample_index` as the raw/global deterministic stream index.
   - Derive bounded data sample index internally via `sample_index_limit`.
   - Seed 3D augmentation from the raw `sample_index`.
   - Remove the alternate public augmentation-index arguments.

2. Fix TensorBoard target/context presence visualization.
   - Build a display-only line-presence volume from `target_segment_*` metadata.
   - Composite it with max-pooled `presence_target` for the visualization
     target/context presence panel.
   - Do not feed this display-only raster into loss materialization.

3. Add multi-sample train/test visualization.
   - Add `training.sample_vis_count` / `training.train_sample_vis_count`
     defaulting to `4`.
   - Add `training.test_sample_vis_count`, defaulting to the train count.
   - Concatenate up to that many batch sample sheets side by side for train and
     test.
   - Remove the hard one-sample slice from the test visualization path.

4. Make dense 3D tests full-set by default.
   - Resolve omitted `training.test_control_points` the same as explicit `0`:
     all held-out CPs in flat order from zero.
   - Keep positive `test_control_points` as an explicit deterministic random
     debug cap.

5. Add tests.
   - Verify CP-only loss targets stay CP-only while the sheet displays line
     context in the target/context presence panel.
   - Verify multi-sample contact sheets concatenate more than one sample
     horizontally.
   - Verify omitted dense-test count resolves to full flat-order evaluation.
   - Keep existing bounded-data/unbounded-augmentation regression coverage.

## Spec Update

- Document that the 3D target/context presence visualization includes
  visualization-only line segments for CP-only JSON/test samples.
- Document `sample_vis_count` and `test_sample_vis_count`.
- Document that dense 3D tests default to all held-out CPs.
- Clarify that 3D public loader `sample_index` is the raw stream index and
  `sample_index_limit` derives bounded data selection.

## Docs Updates

- Update `docs/code_structure.md` for the 3D sample-sheet layout and
  multi-sample visualization controls.

## Changelog

- Add a 2026-07-15 entry for 3D visualization target/context presence and raw
  sample-index cleanup.

## Deviations Or Deferrals

- None planned.
