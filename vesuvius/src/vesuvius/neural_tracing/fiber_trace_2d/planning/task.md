# Trace2CP Timing Rows

- Print timing diagnostics for Trace2CP inference and tracing stages.
- Output should be table-like with one row per stage, not one line per patch.
- Include model inference timing and the main tracing/debug stages so slow
  pieces of Trace2CP visualization can be identified.
- For whole-fiber Trace2CP, aggregate rows by stage instead of printing noisy
  per-pair timings.
- Adjust the side-strip joint DP tracer so it is less visibly choppy and closer
  to the baseline direct tracer: use a finer angular lattice from longer
  horizontal transitions and apply the existing candidate-angle limit as a DP
  direction penalty.
- Print progress while slow Trace2CP DP optimizations run, including ETA, so
  long-running side/z/top DP solves show liveness before the final timing table.
