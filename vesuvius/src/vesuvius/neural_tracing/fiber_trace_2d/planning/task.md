# Trace2CP Regular Trace With Side-View Z Experiment

- Keep the regular side-strip Trace2CP tracer as the default behavior for now,
  because it still performs better than the DP variant.
- Add an opt-in experimental trace mode that remains a regular stepwise trace
  but also carries lateral motion through the side-strip z axis.
- Side-strip vertical correction remains the main movement. The z/lateral state
  is expected to move only modestly, matching the top-view/fiber-side motion.
- For side-strip inference, reuse the existing separate side z layers; the
  sideways motion is small enough that the angle change from not rebuilding the
  side strip per z state is acceptable for this experiment.
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
- This is an experiment/diagnostic path only. It must not replace the default
  regular Trace2CP behavior.
