# Task Plan: Trace2CP Metric Command-Line Tool

## Scope

Implement a runner command-line inspection tool for the `trace2cp` metric
described in `planning/plan.md`.

This task is intentionally limited to an export/inspection command. It will not
wire the metric into training, TensorBoard, snapshots, or evaluation loops.

## User-Facing Behavior

- Add a runner flag, tentatively `--trace2cp-vis`, used like the existing
  `--line-trace-vis`.
- Required args:
  - `config`
  - `--checkpoint <snapshot>`
  - `--export-dir <dir>`
  - `--sample-index <idx>`
- Default CP pair:
  - `--sample-index` resolves through the same deterministic sample ordering as
    training, prefetch, augment-vis, dir-vis, and line-trace-vis.
  - The start CP is that sample's control point.
  - The target CP defaults to the next control point in the same fiber
    (`start_cp_index + 1`).
- Add optional CP-pair controls:
  - `--trace2cp-target-offset`, default `1`, for adjacent/future CP selection.
  - `--trace2cp-target-cp-index`, optional absolute CP index override.
  - Reject target indices outside the same fiber or equal to the start CP.
- Add metric controls:
  - Reuse `--line-trace-step`.
  - Reuse `--line-trace-rf-margin`.
  - Reuse `--line-trace-tta-count` if TTA visualization is enabled for this
    tool.
- Outputs:
  - Print one concise score line to stdout, including sample index, fiber path,
    start/target CP indices, score, raw y-error pixels, target column, and
    traced status.
  - Write `trace2cp_vis.jpg`.
  - Write `trace2cp_summary.txt` with the same metadata plus checkpoint path,
    checkpoint step, strip dimensions, CP coordinates, step size, RF margin, and
    trace termination reason.

## Metric Semantics

- Load a side-strip segment spanning the two CPs with enough x-margin on both
  sides for the trace receptive-field margin and visualization context.
- Run the existing direction model on the strip and decode the Lasagna
  ambiguous two-cos-channel direction output through the existing direction
  helpers.
- Trace starts at the transformed start CP.
- Initial direction is aligned toward the transformed target CP.
- The trace runs only in the start-to-target direction, not both directions.
- Stop when:
  - the trace reaches/crosses the target CP x-column;
  - the next point would enter the RF margin;
  - bilinear direction sampling fails;
  - image validity around the bilinear sample is insufficient;
  - a configured maximum step cap is reached.
- If the trace crosses the target CP x-column, linearly interpolate the trace
  y-coordinate at that x-column between the bracketing trace points.
- Raw error is `abs(trace_y_at_target_x - target_cp_y)`.
- Normalized score is clamped to `0..1`.
- `0.0` means exact y-hit at the target CP column.
- `1.0` means strip-edge error or failure to reach the target column.
- Normalization denominator:
  - use the distance from target CP y to the nearest usable vertical strip edge
    after RF-margin exclusion;
  - if that denominator is non-positive, fail clearly because the segment strip
    is unusable for trace2cp scoring.

## TTA Behavior

- Implement trace2cp TTA and reuse existing line-trace TTA infrastructure where
  possible.
- `planning/plan.md` says trace2cp TTA should leave out y-shift and scale
  because those are hard to handle for long strips.
- Therefore trace2cp TTA variants should include only geometric transforms that
  can be unambiguously mapped back to the reference segment strip for this
  metric:
  - horizontal flip only if start/target semantics are handled explicitly;
  - rotations that preserve target-column interpretation only if mapped back
    before scoring;
  - x-shift and shear/smooth offset only if the target-column crossing is
    evaluated after inverse mapping back to reference strip coordinates.
- Score each TTA trace only after inverse-mapping the trace back into reference
  strip coordinates.
- Do not include y-shift or scale in trace2cp TTA.
- Visualization shows the base trace, TTA flock overlay, and if `--med-tta` is
  set a median-direction TTA trace, all in reference strip coordinates.

## Loader/Data Path

- Add a loader method for CP-pair strip segment construction, rather than using
  the fixed CP-local center patch shape.
- It must keep the existing hard requirements:
  - Lasagna manifest path only for normals;
  - VC3D side-strip coordinate semantics;
  - no neural-tracing 3D crop loader;
  - same configured `augment_device` for torch coordinate generation except
    prefetch;
  - explicit NumPy boundary only at VC3D coordinate sampling and export;
  - no fabricated normals.
- The segment loader should:
  - resolve `(record, record_index, start_cp_index)` from `sample_index` using
    `descriptor_for_sample_index(..., sample_mode="random")`;
  - validate target CP is in the same record/fiber;
  - compute the CP-pair line window and side-strip coordinates covering the
    start-to-target segment plus margins;
  - use selected-level pixel spacing semantics from `base_volume_scale`;
  - build transformed `line_xy`, `start_cp_xy`, and `target_cp_xy` in output
    strip pixels;
  - sample one center strip-z image for now, unless the current loader's
    center-strip conventions naturally expose the same shape with z offset `0`.
- Prefer extending existing strip geometry helpers over adding a second
  incompatible coordinate generator.
- Any new helper should be reusable later for the planned segment refinement
  tool.

## Runner Implementation Plan

1. Refactor small reusable pieces from `--line-trace-vis` if needed:
   - single-direction trace helper;
   - target-column interpolation/scoring helper;
   - overlay helper for CP-pair traces and score text.
2. Add `_export_trace2cp_vis(...)` in `runner.py`.
3. Add CLI args:
   - `--trace2cp-vis`
   - `--trace2cp-target-offset`
   - `--trace2cp-target-cp-index`
4. Keep errors explicit for unsupported CP pairs, missing checkpoint, invalid
   segment geometry, and invalid target column.
5. Export a visualization:
   - base strip image;
   - original transformed fiber centerline at fixed opacity;
   - start CP and target CP markers;
   - traced line from start toward target;
   - target x-column marker or small target crosshair;
   - score text in a label band or non-overlapping overlay area.

## Tests

Add focused tests under `vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
or a new runner-focused test file if that keeps concerns cleaner.

Minimum tests:

- Scoring helper:
  - exact hit at target column gives score `0.0`;
  - y-error equal to usable edge distance gives score `1.0`;
  - failure to reach target column gives score `1.0`;
  - crossing target x interpolates y correctly.
- CLI/export wiring with monkeypatched loader/model helpers:
  - `--trace2cp-vis` requires `--checkpoint` and `--export-dir`;
  - successful export writes `trace2cp_vis.jpg` and `trace2cp_summary.txt`;
  - stdout includes the numeric score.
- Loader CP-pair validation:
  - target CP outside fiber is rejected;
  - target CP equal to start CP is rejected.

Validation commands:

```bash
python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

If a runner-specific test file is added, include it in the pytest command.

Manual smoke command after implementation, using local checkout conventions from
`planning/local_development.md`:

```bash
PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.runner $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --sample-index 0 --trace2cp-vis --checkpoint /path/to/current.pt --export-dir /tmp/fiber_trace_2d_trace2cp
```

## Spec Update

Update `planning/specs.md` to add:

- `--trace2cp-vis` runner tool behavior and required args.
- CP-pair default selection from deterministic sample order.
- Trace2CP score definition:
  - trace from first CP toward second CP;
  - evaluate y-error at target CP x-column;
  - normalize/clamp to `0..1`;
  - failed target-column reach scores `1.0`.
- CP-pair strip loading uses VC3D/Lasagna side-strip semantics, not fixed center
  patch shortcuts.
- Trace2CP TTA, if enabled, excludes y-shift and scale and scores only after
  inverse mapping traces back into reference strip coordinates.

## Docs Updates

Update `docs/code_structure.md` to document:

- runner CLI usage for `--trace2cp-vis`;
- output files;
- how the score is computed;
- relationship to existing `--line-trace-vis`.

## Changelog Update

Add a dated one-line entry after implementation:

- Added trace2cp runner visualization and score export for CP-pair segment
  tracing.

## Plan Review Notes

- Matches `planning/plan.md` trace2cp requirements: side strip between two CPs,
  trace from first toward second, y-error at target column, `0..1` score.
- Preserves current `planning/specs.md` requirements: deterministic sample
  ordering, Lasagna normals, VC3D side-strip coordinate semantics, configured
  augmentation device, and existing runner/checkpoint direction tooling.
- Keeps out-of-scope training/evaluation integration out of this task.
