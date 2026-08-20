# Task: reliable and parallel on-demand fiber replay

Make `vc_fiberlets fiberlet-replay` progress coherent across greedy restart
segments: distinguish local native-trace steps and their safety budget from
global reference progress, and identify the current replay segment/restart.

Fix the observed post-greedy on-demand replay stall. Cache waits remain ordinary
blocking waits; the generation scheduler itself must always reach completion or
return the exact chunk key and underlying error. Do not add polling,
heartbeats, or timeout-based recovery.

Use the configured worker count for the expensive work inside one on-demand
fiberlet chunk. Preserve deterministic candidate ordering and numerical
results.

Report replay and cache preprocessing on one monotone global reference-arc
basis. Restart-local native steps may be shown only as explicitly local
diagnostics. Report explicit terminal state for greedy and fiberlet evaluators.

Reproduce and validate against the full Paris4 reference and existing sparse
anchor/fiberlet cache that produced the final visible `step=300/1055` line.
