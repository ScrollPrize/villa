# Trace2CP Shared Strip Builder And Tighter Reset Threshold Plan

## Implementation

- Extract the current common strip-source construction body from
  `build_trace2cp_segment_source(...)` into a focused helper, e.g.
  `_build_trace2cp_source_from_points(...)`.
- The shared helper input should be explicit data, not implicit original-strip
  state:
  - `record` / `record_index`;
  - `line_points_xyz`;
  - `anchor/local start index`;
  - `target/local target index`;
  - start/target CP indices for metadata only;
  - source shape, anchor column, center strip-z offset, and optional row-axis
    sign-alignment reference.
- The shared helper should construct the `FiberStripLineWindow`, compute
  `line_xy`, start/target XY, sample/align Lasagna normals, and call the same
  `build_side_strip_patch_grid_tensor_from_line_window(...)` used today.
- Keep `build_trace2cp_segment_source(...)` as the original-fiber wrapper:
  - resolve sample/record/start/target CPs;
  - compute distance, width, margins, and original `side_strip_segment_line_window`;
  - trim against compact valid intervals for the original fiber line;
  - call the shared helper with original `record.fiber.line_points_xyz` window
    and original line indices.
- Replace `build_trace2cp_refined_segment_source(...)` old-grid dependency with
  a traced-line wrapper:
  - accept actual volume-space `trace_xyz` line points;
  - filter only malformed/non-finite/degenerate trace points, not points outside
    the original strip;
  - optionally extend/trim failed traces in volume-line space before calling the
    helper;
  - call the shared helper with synthetic line-local indices.
- Fresh traced-line normal handling:
  - sample Lasagna normals at the traced 3D points using the existing normal
    decoder path;
  - do not use `source.grid.coords_xyz`, `source.grid.offset_axis_xyz`, or
    `source.grid.side_axis_xyz` to recover the regenerated line;
  - align normal signs along the traced line and optionally to the original
    start row-axis reference so regenerated rows are stable.
- Native 3D rendering integration:
  - for regenerated/fused rows, pass the native fused/restart-delimited trace
    as volume coordinates directly to the traced-line wrapper;
  - keep original-strip overlays clipped only for original rows;
  - do not fail or truncate regenerated rows because the trace left the
    original source strip.
- Lower the native whole-fiber reset/error threshold:
  - change `NativeTrace2CpConfig.whole_fiber_error_threshold_voxels` default
    from `100.0` to `10.0`;
  - change CLI `--whole-fiber-error-threshold-voxels` default from `100.0` to
    `10.0`;
  - update tests and docs that assert/report the default.

## Spec Update

- Update `planning/specs.md` so regenerated/refined Trace2CP strip construction
  is specified as fresh strip construction from explicit 3D line points using
  the same low-level strip builder as original CP-pair strips.
- Update specs to clarify that original-strip clipping is a display overlay
  concern only, not an input constraint for regenerated/fused strips.
- Update native whole-fiber default error threshold from 100 to 10 selected-scale
  voxels.

## Docs Updates

- Update `docs/` only if existing Trace2CP docs mention regenerated strip
  construction or the 100-voxel threshold.
- Update planning changelog with the shared-strip-builder refactor and threshold
  change.

## Tests

- Add/adjust loader tests so original CP-pair source and traced-line source both
  call the shared builder path and produce compatible source metadata.
- Add a regression where the traced/refined line leaves the original strip but
  still builds regenerated side/top strips from its volume coordinates.
- Add a regression that regenerated traced-line construction does not sample the
  original source grid to recover line points.
- Add a regression for Lasagna normal sampling/sign stability on traced-line
  regenerated sources.
- Update native 3D Trace2CP config/CLI default tests for 10-voxel whole-fiber
  threshold.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'trace2cp_refined or trace2cp_segment_source'`
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run `git diff --check` over touched files.

## Non-Goals

- Do not change the VC3D/Lasagna-equivalent strip geometry algorithm.
- Do not make regenerated strips from planar approximations.
- Do not change native 3D tracing candidate scoring, beam search, or fusion
  selection in this task.
