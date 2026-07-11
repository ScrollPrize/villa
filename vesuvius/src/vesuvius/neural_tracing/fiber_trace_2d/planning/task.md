# Trace2CP Target-Column Metric

Switch the public Trace2CP metric back to measuring y error at the opposite
control-point columns.

The closest-point/intersection logic should remain the source for Trace2CP
fusion/refinement visualization, but `trace2cp_error` should no longer be
computed from the closest trace-to-trace intersection gap. This must apply to
all Trace2CP modes, including direction-only, median-TTA, and combined
direction/embedding tracing.
