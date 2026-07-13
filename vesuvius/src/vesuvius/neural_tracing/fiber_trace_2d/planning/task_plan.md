# Trace2CP Regular Trace With Side-View Z Experiment Plan

## Interpretation

- Yes, the proposed experiment is coherent: it keeps the proven regular
  side-strip tracer as the baseline, then adds a second opt-in diagnostic that
  augments that stepwise trace with lateral/top-view state.
- The experiment and plain z-search should not use the recent side DP path as
  the default. DP can remain available, but Trace2CP must route through the
  regular stepwise tracer unless the user explicitly selects a DP mode.
- `--trace2cp-side-top-z-experiment` is exclusive: it should run the side/top-z
  diagnostic only, not the full normal Trace2CP visualization first.
- The new experiment should treat regular side-strip candidate scoring as
  authoritative for x/y motion and z/lateral motion as secondary. The z state
  selects which side z-layer prediction is sampled for the current/candidate
  points; top inference updates that z state only after a side candidate has
  already been selected.

## Implementation

- Add a CLI mode/flag for the experimental trace, with a name that clearly
  indicates regular stepwise tracing plus top-informed z/lateral motion.
- Add an explicit DP flag, `--trace2cp-dp`, and gate all side/side-z DP routing
  on that flag. Do not treat `--trace2cp-z-search` as a DP request.
- Keep existing regular Trace2CP command behavior and output unchanged unless
  the new experiment flag is set.
- When the experiment flag is set, branch before `_evaluate_trace2cp_refinement_chain`
  so the command does not write `trace2cp_vis.jpg` or `trace2cp_summary.txt`.
- Use the same side candidate fan structure as the regular forward/backward
  combined tracer for every side x/y step: score candidates by interpolated
  current and candidate side direction agreement, and by optional presence when
  the CLI presence weight is active. Do not score embedding, image similarity,
  or DP costs in this experiment.
- Add a small per-step top-view source builder that can sample a top patch
  around the current traced 3D point without rebuilding unrelated fiber/global
  state.
- Derive the top-patch slicing normal/orientation from the sampled side-view
  direction at the current trace step.
- Keep the top correction one-dimensional relative to the side-view normal for
  now. Do not introduce roll optimization or arbitrary rotation around the
  fiber line in this experiment.
- Run the top-view model on that patch for each step where z/lateral inference
  is needed, after a side candidate has been selected. Do not run top inference
  over all side candidates.
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
- Store each accepted-step top patch used for z update, including the patch
  image, valid mask, accepted side point, z offset, and estimated top direction.
- Reconstruct z-corrected side and top visualization inputs from the produced
  trace state rather than from the original straight segment.
- Export a separate diagnostic image/set of rows containing:
  - left and right experimental traces;
  - top views for the resulting traces;
  - side views with the resulting paths;
  - z-corrected variants derived from the stepwise top state.
- Export two debug directories: raw local top slices and matching native-size
  direction overlays. Clear stale JPGs in those generated directories before
  writing a new run.

## Spec Update

- Add a Trace2CP experimental mode spec: default remains regular side tracing;
  the new mode is opt-in and combines regular side step tracing with a
  top-view-derived z/lateral state.
- Add that DP is a separate opt-in backend selected by `--trace2cp-dp`; plain
  `--trace2cp-z-search` uses stepwise candidate-fan z-search.
- Add that the side/top-z experiment is exclusive and writes only its own
  artifacts.
- Specify that top-view directions for this experiment are re-inferred per step
  on a local patch and aggregated with ambiguity-aligned weighted median over a
  normal neighborhood, default radius 20 px.
- Specify that top patch orientation is derived from the sampled side-view
  direction, and that this experiment only corrects angle relative to the
  side-view normal rather than optimizing rotation around the line.
- Specify that side x/y direction sampling follows regular candidate scoring
  and reads the current z-layer side prediction for both the current and
  candidate points. Top inference is run once per accepted point and only
  updates z.
- Specify that XYZ trace coordinates are stored and stepped as float/subpixel
  values; rounding is limited to z-layer lookup and column-based rendering.
- Specify the two top-slice debug output directories and their contents.
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
- Add routing tests proving stepwise combined and z-search paths do not call DP
  by default, and explicit DP paths do call the DP helper.
- Add an export regression test proving side/top-z experiment bypasses the
  normal Trace2CP refinement chain and does not write normal Trace2CP outputs.
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
