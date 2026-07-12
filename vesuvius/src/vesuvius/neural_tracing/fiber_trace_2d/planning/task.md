# Trace2CP Iterative Fused-Trace Refinement

Add an opt-in two-pass/multi-pass Trace2CP fiber tracing mode.

- When enabled, the runner takes the final fused fiber trace from the previous
  pass, smooths it, and uses that curve as the line input for another Trace2CP
  pass.
- The mode must support multiple refinement iterations, not only a hard-coded
  second pass.
- Iteration visualizations and summaries after the initial pass are written
  with iteration-specific names such as `it1`.
- The next-pass image data must be sampled from the volume using the refined
  curve geometry; it must not be an image-space warp of the previous strip.
- Refined passes must behave like normal Trace2CP runs on an independent line:
  both directions need valid endpoint context, including reverse tracing from
  the target CP.
- Refine smoothing must use a Gaussian blur over trace points, not a box or
  moving-average blur.
