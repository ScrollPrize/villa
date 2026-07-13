# Trace2CP Regular Trace With Side-View Z Experiment Plan

## Interpretation

- Yes, the proposed experiment is coherent: it keeps the proven regular
  side-strip tracer as the baseline, then adds a second opt-in diagnostic that
  augments that stepwise trace with lateral/top-view state.
- The experiment should not use the recent side DP path as the default. DP can
  remain available, but this task should route normal Trace2CP back through the
  regular tracer unless the user explicitly selects a DP mode.
- The new experiment should treat side-strip y motion as primary and z/lateral
  motion as secondary. The z state is updated from top-view inference and then
  used to pick/correct side z layers for the trace and visualization.

## Implementation

- Add a CLI mode/flag for the experimental trace, with a name that clearly
  indicates regular stepwise tracing plus top-informed z/lateral motion.
- Keep existing regular Trace2CP command behavior and output unchanged unless
  the new experiment flag is set.
- Reuse existing side z-layer cache/prediction machinery for side-strip
  direction and presence reads.
- Add a small per-step top-view source builder that can sample a top patch
  around the current traced 3D point without rebuilding unrelated fiber/global
  state.
- Derive the top-patch slicing normal/orientation from the sampled side-view
  direction at the current trace step.
- Keep the top correction one-dimensional relative to the side-view normal for
  now. Do not introduce roll optimization or arbitrary rotation around the
  fiber line in this experiment.
- Run the top-view model on that patch for each step where z/lateral inference
  is needed.
- Compute the top direction estimate by:
  - sampling predicted top directions over a radius-20 normally weighted area;
  - aligning every ambiguous direction to the current/reference horizontal
    motion before aggregation;
  - taking a weighted median or equivalent robust component-wise median and
    normalizing the result.
- Convert the top-view direction into the side-strip z/lateral state update for
  the next step.
- Generate both left-to-right and right-to-left traces.
- Store per-step state as at least side xy, side z offset/layer, top direction,
  and reason for termination.
- Reconstruct z-corrected side and top visualization inputs from the produced
  trace state rather than from the original straight segment.
- Export a separate diagnostic image/set of rows containing:
  - left and right experimental traces;
  - top views for the resulting traces;
  - side views with the resulting paths;
  - z-corrected variants derived from the stepwise top state.

## Spec Update

- Add a Trace2CP experimental mode spec: default remains regular side tracing;
  the new mode is opt-in and combines regular side step tracing with a
  top-view-derived z/lateral state.
- Specify that top-view directions for this experiment are re-inferred per step
  on a local patch and aggregated with ambiguity-aligned weighted median over a
  normal neighborhood, default radius 20 px.
- Specify that top patch orientation is derived from the sampled side-view
  direction, and that this experiment only corrects angle relative to the
  side-view normal rather than optimizing rotation around the line.
- Specify that side z-layer reuse is acceptable for this experiment because the
  lateral motion is assumed small.
- Specify that the experiment exports separate diagnostics and must not silently
  replace default Trace2CP outputs or training metrics.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP section with the new experimental
  regular-plus-top-z tracer entry points, how it relates to the regular and DP
  tracers, and what outputs it writes.
- Document the CLI flag and a minimal invocation example once implemented.

## Tests

- Add focused tests around the top direction aggregation:
  - ambiguous directions align before aggregation;
  - opposite-sign equivalents do not cancel;
  - weighted median returns the expected dominant direction in a synthetic
    field.
- Add a runner-level smoke/regression test for the new flag using mocked or
  minimal prediction fields where possible.
- Run:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
- Run focused loader/runner tests covering Trace2CP paths.
- Run the full existing fiber_trace_2d loader test file if the implementation
  touches shared loader or visualization code.

## Changelog

- Add a dated changelog note when the experimental regular-plus-top-z Trace2CP
  mode is implemented.

## Open Implementation Notes

- The top-view per-step inference will be more expensive than reusing a cached
  long strip. That is acceptable for an experiment, but timings should be
  printed or captured so we can decide whether to batch/cache later.
- The z/lateral update must define units explicitly: use selected-scale voxel
  offsets for side-strip z state and convert consistently when rendering top
  and side corrected views.
