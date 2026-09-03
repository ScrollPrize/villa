# Fiber benchmark results

This index records scientific benchmark invocations. Each row links to the
complete command, source revision, effective settings, artifact identities,
timing, and detailed results. These are manual external-data measurements, not
CI performance gates.

## Reference endpoint replay

| Date | Revision | Policy | Crop | Tested length | Failures | Distance/failure | Distance % | Wall time | Run |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-09-03 | `cd0fbd52a` | Fiberlet whole run | PHercParis4 1024 | 101.036 mm | 6 | 16.839 mm | 16.667% | 21.41 s | [record](fiber_benchmark_runs/2026-09-03-cd0fbd52a-reference-distance-per-failure.md) |
| 2026-09-03 | `6c006d9b0` | Greedy direct | PHercParis4 1024 | 101.036 mm | 13 | 7.772 mm | 7.692% | 0.49 s | [record](fiber_benchmark_runs/2026-09-03-6c006d9b0-greedy-reference-replay.md) |
| 2026-09-03 | `6c006d9b0` | Lasagna transport | PHercParis4 1024 | 101.036 mm | 57 | 1.773 mm | 1.754% | 0.09 s | [record](fiber_benchmark_runs/2026-09-03-6c006d9b0-lasagna-reference-replay.md) |

## Oracle piece pruning

| Date | Revision | Crop | Pieces removed | Piece problematic | Constraint problematic | Reference result | Wall time | Run |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-09-03 | `1a70f9e57` | PHercParis4 1024 | 363 / 1360 | 44.43% | 64.02% | 24 exact, 0 wrong, 2 missing | 57.62 s | [record](fiber_benchmark_runs/2026-09-03-1a70f9e57-oracle-pruning.md) |
