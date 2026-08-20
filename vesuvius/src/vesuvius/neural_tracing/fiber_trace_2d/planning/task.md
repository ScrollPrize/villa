# Task: concise replay progress and latest fiberlet optimizations

Make `vc_fiberlets fiberlet-replay` print one overall progress bar with elapsed
time and ETA instead of per-stage, per-chunk, per-restart, and per-evaluator
diagnostics during ordinary use.

Retain the detailed diagnostics behind an explicit statistics option. Commit
that output change, then merge the unmerged `fiber-lets2` performance work and
adapt it to the chunked/cached replay implementation where necessary. Preserve
exact float-cache/eager replay equivalence.
