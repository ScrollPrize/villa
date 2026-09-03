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

## Benchmark

- Committed revision `cd0fbd52a` reran the frozen PHercParis4 1024 crop.
- Total tested directed length: 101.036 mm; failures: 6; mean distance per
  failure: 16.839 mm; distance per failure percentage: 16.667%.
- Runtime: 21.41 seconds wall, 117.98 seconds user, 14.28 seconds system;
  maximum RSS 13,276,964 KiB.
