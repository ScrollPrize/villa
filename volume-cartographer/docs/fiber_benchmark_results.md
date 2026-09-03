# Fiber benchmark results

This index records scientific benchmark invocations. Each row links to the
complete command, source revision, effective settings, artifact identities,
timing, and detailed results. These are manual external-data measurements, not
CI performance gates.

## Progress Plots

![Reference-fiber replay reliability over algorithm completion time](fiber_benchmarks/imgs/fiber_reference_replay_progress.svg)

Reference replay uses `100 / max(failures, 1)`, equivalently mean tested
distance per failure divided by total tested directed length. Higher is better;
zero and one failure both saturate at 100 percent under the benchmark's censored
zero-failure convention.

![Crop constraint error over algorithm completion time](fiber_benchmarks/imgs/fiber_crop_pruning_progress.svg)

The crop plot preserves the recorded problematic-to-retained error ratio and
negates it so higher is better:
`-100 * problematic / retained_fulfilled`. The original capped Fiberlet result
is `-177.93%`; the staged uncapped result is `-139.68%`; zero is the ideal
target. The direct greedy and Lasagna markers are unmeasured plotting-floor assumptions without numeric benchmark values;
neither produces the candidate-piece graph required by this benchmark. The two
plots use different metrics and their percentages are not numerically
comparable.

The horizontal coordinate is the historical algorithm completion date, not the
later benchmark execution date. Each data row separately records algorithm and
measurement revisions. Regenerate both deterministic SVGs with:

```bash
python volume-cartographer/scripts/plot_fiber_benchmarks.py
```

Add later results to `docs/fiber_benchmark_plot_data.json`; scores are derived
from raw failure or unique-constraint counts rather than copied percentages.
Markers show every measured result at its actual score. The step line is the
cumulative best result, so a later regression remains visible without lowering
the historical progress line.

## Reference endpoint replay

| Date | Revision | Policy | Crop | Tested length | Failures | Distance/failure | Distance % | Wall time | Run |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-09-03 | `cd0fbd52a` | Fiberlet whole run | PHercParis4 1024 | 101.036 mm | 6 | 16.839 mm | 16.667% | 21.41 s | [record](fiber_benchmark_runs/2026-09-03-cd0fbd52a-reference-distance-per-failure.md) |
| 2026-09-03 | `3046918b5` | Fiberlet, staged 256/256-offset/512 | PHercParis4 1024 | 101.036 mm | 7 | 14.434 mm | 14.286% | 243.75 s | [record](fiber_benchmark_runs/2026-09-03-3046918b5-staged-reference-replay.md) |
| 2026-09-03 | `6c006d9b0` | Greedy direct | PHercParis4 1024 | 101.036 mm | 13 | 7.772 mm | 7.692% | 0.49 s | [record](fiber_benchmark_runs/2026-09-03-6c006d9b0-greedy-reference-replay.md) |
| 2026-09-03 | `6c006d9b0` | Lasagna transport | PHercParis4 1024 | 101.036 mm | 57 | 1.773 mm | 1.754% | 0.09 s | [record](fiber_benchmark_runs/2026-09-03-6c006d9b0-lasagna-reference-replay.md) |

## Oracle piece pruning

| Date | Revision | Crop | Pieces removed | Piece problematic | Constraint problematic | Reference result | Wall time | Run |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-09-03 | `1a70f9e57` | PHercParis4 1024 | 363 / 1360 | 44.43% | 64.02% | 24 exact, 0 wrong, 2 missing | 57.62 s | [record](fiber_benchmark_runs/2026-09-03-1a70f9e57-oracle-pruning.md) |
| 2026-09-03 | `3046918b5` | PHercParis4 1024, staged uncapped | 308 / 1450 | 39.04% | 58.28% | 24 exact, 0 wrong, 2 missing | 81.34 s median | [record](fiber_benchmark_runs/2026-09-03-3046918b5-staged-oracle-pruning.md) |

The older pruning row uses a capped 1,998-trace unstaged cohort; the staged row
uses the complete uncapped 2,062-trace cohort. Their difference combines stage
filtering with cohort completion and is not a controlled causal comparison.
