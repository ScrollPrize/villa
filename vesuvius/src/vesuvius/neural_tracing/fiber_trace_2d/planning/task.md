# Task: Tool1 Patch Line Tracing

Implement the first V0.1 inspection tool from `todo.md`: patch line
refinement on a side-strip patch.

- Work on one deterministic sample selected by `--sample-index`, using the same
  ordering as training, prefetch, and augment-vis.
- Load the side-strip patch for that sample.
- Run the trained direction model on the patch.
- Trace a direction-based line in both directions from the control point by
  repeatedly bilinearly sampling the predicted direction field and stepping.
- Stop before the traced point enters the border band where the model receptive
  field would touch the patch edge.
- Export a visualization showing both the original strip line and the
  direction-traced line.
