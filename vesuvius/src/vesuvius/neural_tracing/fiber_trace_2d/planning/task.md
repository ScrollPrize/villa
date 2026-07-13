# Trace2CP Timing Rows

- Print timing diagnostics for Trace2CP inference and tracing stages.
- Output should be table-like with one row per stage, not one line per patch.
- Include model inference timing and the main tracing/debug stages so slow
  pieces of Trace2CP visualization can be identified.
- For whole-fiber Trace2CP, aggregate rows by stage instead of printing noisy
  per-pair timings.
