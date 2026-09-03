# Fiber benchmark results

This index records scientific benchmark invocations. Each row links to the
complete command, source revision, effective settings, artifact identities,
timing, and detailed results. These are manual external-data measurements, not
CI performance gates.

## Reference endpoint replay

| Date | Revision | Policy | Crop | Tested length | Failures | Distance/failure | Distance % | Wall time | Run |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-09-03 | `cd0fbd52a` | Whole run | PHercParis4 1024 | 101.036 mm | 6 | 16.839 mm | 16.667% | 21.41 s | [record](fiber_benchmark_runs/2026-09-03-cd0fbd52a-reference-distance-per-failure.md) |

## Oracle piece pruning

| Date | Revision | Crop | Pieces removed | Piece problematic | Constraint problematic | Reference result | Wall time | Run |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-09-03 | `1a70f9e57` | PHercParis4 1024 | 363 / 1360 | 44.43% | 64.02% | 24 exact, 0 wrong, 2 missing | 57.62 s | [record](fiber_benchmark_runs/2026-09-03-1a70f9e57-oracle-pruning.md) |
