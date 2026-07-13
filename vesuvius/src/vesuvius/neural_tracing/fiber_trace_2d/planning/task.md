# Trace2CP Regular Trace With Side-View Z Experiment

- Keep the regular side-strip Trace2CP tracer as the default behavior for now,
  because it still performs better than the DP variant.
- Dynamic programming must only run when an explicit DP flag is passed. Plain
  `--trace2cp-z-search` must keep using the older stepwise z-search behavior.
- Add an opt-in experimental trace mode that remains a regular stepwise trace
  but also carries lateral motion through the side-strip z axis.
- Side-strip vertical correction remains the main movement. The z/lateral state
  is expected to move only modestly, matching the top-view/fiber-side motion.
- For side-strip x/y tracing, use the same candidate fan scoring semantics as
  the regular forward/backward Trace2CP trace: sample/interpolate direction at
  both the current/last point and the candidate point from their current
  z-layer side prediction, and include optional presence when enabled. The
  top-view inference should not run for every candidate; after the side
  candidate is selected, run one local top slice at that accepted point and use
  it only to update the carried side-view z/lateral state.
- For top-view direction inference, re-run a small planar/top patch at each
  trace step. The top view needs fresh local inference because the traced
  lateral position changes what top patch is centered at the current step.
- The normal/orientation used to slice each top patch is derived from the
  sampled side-view direction at that step. For now the experiment only
  corrects the angle relative to the side-view normal; it must not optimize or
  rotate around the fiber line.
- For each top patch, estimate a top-view direction by taking an
  ambiguity-aligned weighted median over a relatively large normally weighted
  neighborhood, initially radius 20 px.
- Produce a separate experimental output containing:
  - left and right traces;
  - top views for the resulting traces;
  - side views with the path;
  - all views corrected for the side-view z motion derived from the stepwise top
    patch inference.
- Also write all per-step local top slices used by the experiment to
  `trace2cp_side_top_z_top_slices/` and matching full-size direction overlays
  to `trace2cp_side_top_z_top_overlays/`.
- XYZ stepping should remain subpixel/floating-point. Rounding is only allowed
  for selecting cached/inferred z layers or display columns.
- This is an experiment/diagnostic path only. It must not replace the default
  regular Trace2CP behavior.
- When `--trace2cp-side-top-z-experiment` is passed, the command should run
  only that experiment and write only its experiment-specific outputs; it must
  not also run the normal Trace2CP overlay/refinement path.
- The side/top-z experiment should print throttled progress while the forward
  and backward traces repeatedly sample local top patches and run inference.
- The extracted forward/backward top-slice debug outputs must include
  direction overlays in the extra overlay directory; the compact side/top-z JPG
  should not draw those per-step ticks.
