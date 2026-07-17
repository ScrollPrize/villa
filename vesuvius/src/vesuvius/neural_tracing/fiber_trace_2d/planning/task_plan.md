# Native 3D Whole-Fiber Trace2CP Plan

## Scope

Implement whole-fiber native 3D Trace2CP for `fiber_trace_3d.trace2cp_tool`.
The default behavior changes only when `--fiber-json` is supplied without
explicit CP indices. Explicit `--fiber-json --start-cp-index A
--target-cp-index B` remains a single-segment debug mode.

No training loop migration is part of this task. The native 3D whole-fiber
metric remains tool-local unless a later task explicitly wires it into
checkpoint selection.

## Behavior

1. Add whole-fiber selection mode.
   - If `--fiber-json` is set and neither `--start-cp-index` nor
     `--target-cp-index` is set, load exactly that fiber and trace consecutive
     CP pairs from CP `0` through CP `N-1`.
   - If both explicit CP indices are supplied, keep current single-pair mode.
   - If no `--fiber-json` is supplied, keep current `--sample-index` based
     single-pair behavior.
   - Reject mixed partial arguments loudly, as the current single-pair resolver
     already does.

2. Refactor the one-way tracer into reusable segment stepping.
   - Keep candidate selection semantics unchanged:
     `dot(current_dir, step_dir) * dot(candidate_dir, step_dir) * presence`.
   - Keep selected-level ZYX coordinates internally.
   - Split the current `trace_native_3d_one_way(...)` so the segment loop can
     be called with:
     - current point;
     - current direction seed;
     - segment target CP;
     - target plane normal;
     - segment-local step budget.
   - First segment starts from the CP-local fiber tangent toward the next CP.
   - Successful later segments continue from the reached target-plane crossing
     and carry the accepted trace direction forward. A restart re-seeds from
     the CP-local fiber tangent at the restart CP.

3. Define segment target planes and budgets.
   - For segment `i -> i+1`, use the plane through CP `i+1`.
   - Plane normal is the local CP-to-CP direction in selected-level coordinates
     for that segment, matching current native target-plane semantics.
   - Segment step budget is
     `ceil(max_step_factor * segment_cp_distance / step_voxels)`, optionally
     capped by `--max-steps` and `--trace-step-limit` as today.
   - Add `--whole-fiber-error-threshold-voxels`, default `100.0`, in selected
     voxels.

4. Segment success and restart policy.
   - A segment succeeds only if the trace intersects the segment's target CP
     plane within budget and the target-plane in-plane error is at most the
     threshold.
   - In-plane error is the norm of `(crossing - target_cp)` after removing the
     component along the target-plane normal.
   - If the trace fails to reach the plane or reaches it with too much in-plane
     error, count one restart for that segment.
   - Record the reference fiber arc-length distance up to the last successful
     CP plane encountered before that restart.
   - Continue from the failed target CP with a fresh tangent seed toward the
     next CP so one bad segment does not abort the whole fiber.
   - If the last segment fails, record the failure and finish.

5. Whole-fiber metric and output.
   - Print a clear single-line metric:
     `native_trace2cp_fiber_restart_rate=... restarts=... segments=...`.
   - Write a JSON summary containing:
     - fiber path;
     - CP count and segment count;
     - restart count and restart rate;
     - per-segment status, reason, in-plane error, reached-plane flag, step
       count, reference arc distance at last success, and restart point;
     - total inferred block count;
     - exported image path.
   - Keep existing single-pair output fields unchanged.

6. Whole-fiber visualization.
   - Build a stitched view over the reference fiber CP sequence rather than a
     separate image per pair.
   - Keep the four useful panel types:
     - side volume with trace overlay;
     - side 3D presence with trace overlay;
     - top volume with trace overlay;
     - top 3D presence with trace overlay.
   - Use the existing 2D Trace2CP geometry loader/source builders for strip
     coordinates and the existing native inference cache for presence sampling.
   - Render the source/reference strips and project the native trace points into
     those strip coordinate systems as the current single-pair visualization
     does.
   - For failed segments, truncate the displayed failed trace before it would
     overlap the next CP region. Then continue the displayed trace from the
     restart CP. This should leave an intentional gap/cut rather than drawing a
     misleading off-track overlap.
   - Overwrite `trace2cp_native_3d_vis.jpg` after every completed segment, not
     only after whole render stages. The partial image should show all segments
     traced/rendered so far, including visible cut/restart gaps for failed
     segments, so long whole-fiber runs have useful visual progress.
   - Keep the existing stage-level progressive overwrites too; segment-level
     writes are the minimum progress granularity for whole-fiber mode.

7. Progress and diagnostics.
   - Add whole-fiber progress rows reporting CP segment index, success/failure,
     current restart count, current restart rate, ETA, and inferred block
     count.
   - Keep per-direction progress for single-pair mode.
   - For whole-fiber mode, avoid noisy per-step output unless existing progress
     throttling already emits it.

## Spec Update

Update `planning/specs.md` to state:

- `fiber_trace_3d.trace2cp_tool --fiber-json` defaults to whole-fiber tracing
  unless explicit CP indices are supplied.
- Whole-fiber native tracing is continuous across successful CP planes and
  restarts only after segment failure.
- Segment success requires reaching the next CP plane within budget and staying
  below `--whole-fiber-error-threshold-voxels` in in-plane selected voxels.
- The native whole-fiber error is restart probability:
  `restart_count / segment_count`.
- Whole-fiber visualization uses four panel types and cuts failed trace spans
  before restart stitching.
- Whole-fiber visualization progressively overwrites the regular JPG after
  each segment so partial trace progress is visible while the command runs.
- Single-pair native debug metrics remain distinct from this whole-fiber
  restart metric.

## Docs Updates

Update `docs/code_structure.md` to describe:

- native 3D Trace2CP modes:
  - sample-index single-pair;
  - explicit fiber CP-pair;
  - default fiber-json whole-fiber;
- the whole-fiber restart metric and summary JSON;
- the four-panel whole-fiber visualization and failure cut/stitch behavior;
- the segment-by-segment progressive update behavior for
  `trace2cp_native_3d_vis.jpg`;
- the threshold/coordinate convention: selected-level voxels.

## Tests

Add focused unit tests in `vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
or a new trace2cp-specific test file.

1. Segment error math:
   - Plane crossing exactly at target CP gives zero in-plane error.
   - Crossing offset within the plane gives the expected in-plane distance.

2. Whole-fiber state machine:
   - Mock tracer where all segments succeed: restart rate is `0`.
   - Mock tracer with one plane-miss failure: restart count increments and the
     next segment restarts from the failed CP.
   - Mock tracer with one too-large in-plane error: same restart behavior.
   - Last-segment failure records failure and finishes without trying to trace
     past the fiber.

3. CLI/selection behavior:
   - `--fiber-json` without CP indices routes to whole-fiber mode.
   - `--fiber-json --start-cp-index --target-cp-index` routes to single-pair
     mode.
   - no `--fiber-json` keeps sample-index single-pair mode.

4. Visualization trace trimming:
   - Given synthetic trace fragments and one failed segment, overlay generation
     omits the failed overlap region and restarts at the next CP.

Validation command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py
```

If the new tests are split into a new file, run that file too.

## Changelog Update

Add one durable changelog entry after implementation:

- Native 3D Trace2CP `--fiber-json` now defaults to whole-fiber continuous
  tracing with restart-rate reporting and four-panel stitched visualization.

## Non-Goals

- Do not replace the 3D training `test/trace2cp_error` or best-checkpoint
  metric.
- Do not change model inference normalization, candidate scoring, cone-grid
  generation, strict blocking sampler checks, or selected-level coordinate
  conventions.
- Do not remove explicit single-pair native debug mode.
- Do not add a new DP tracer or revive removed embedding/image scoring paths.
