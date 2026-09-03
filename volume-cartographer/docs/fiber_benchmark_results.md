# Fiber benchmark results

This index records scientific benchmark invocations. Each row links to the
complete command, source revision, effective settings, artifact identities,
timing, and detailed results. These are manual external-data measurements, not
CI performance gates.

## Reference endpoint replay

| Date | Revision | Policy | Crop | Directions | Failure-free | Failures | Mean failed span | Length success | Wall time | Run |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-09-03 | `d1c52f1b0` | Whole run | PHercParis4 1024 | 48 | 43 (89.58%) | 6 | 0.441 mm | 100.000% | 21.58 s | [record](fiber_benchmark_runs/2026-09-03-d1c52f1b0-reference-whole-run-replay.md) |
| 2026-09-03 | `1a70f9e57` | First failure (superseded) | PHercParis4 1024 | 48 | 43 (89.58%) | 5 | 0.465 mm | 90.311% | 21.37 s | [record](fiber_benchmark_runs/2026-09-03-1a70f9e57-reference-endpoint-replay.md) |

## Oracle piece pruning

| Date | Revision | Crop | Pieces removed | Piece problematic | Constraint problematic | Reference result | Wall time | Run |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-09-03 | `1a70f9e57` | PHercParis4 1024 | 363 / 1360 | 44.43% | 64.02% | 24 exact, 0 wrong, 2 missing | 57.62 s | [record](fiber_benchmark_runs/2026-09-03-1a70f9e57-oracle-pruning.md) |
