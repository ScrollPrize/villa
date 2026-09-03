# Task Log

## Scope

- Make total tested distance per failure the primary whole-run replay metric.
- Report that value in millimeters and as a percentage of total tested length.
- Retain detailed failure events as diagnostics.

## Validation

- Release `vc_fiber_trace_chunk` and the focused benchmark test build.
- Focused CTest passes: 5 test cases cover zero, one, and four failures,
  physical conversion, incomplete-evaluation rejection, and JSON output.

## Independent Review

- Use full evaluated directed reference length, including forward and reverse
  as separate tested distances, and reject incomplete case evaluation.
- Count every failure reason consistently.
- Preserve existing version-2 fields and add the corrected metric rather than
  reinterpreting old fields.
- Record that the percentage is `100 / max(failures, 1)`: zero and one failure
  both produce 100 percent, while zero failures use a censored convention.
