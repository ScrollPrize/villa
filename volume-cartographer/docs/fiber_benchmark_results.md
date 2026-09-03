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
is `-177.93%`; the ordinary density-`0.25` result is `-160.80%`; the staged
uncapped result is `-139.68%`; and no-overtrace with the `0.35` quality
threshold is `-98.51%`. Zero is the ideal target. The direct greedy and Lasagna
markers are unmeasured plotting-floor assumptions without numeric benchmark
values; neither produces the candidate-piece graph required by this benchmark.
The three plots use different metrics and their percentages are not numerically
comparable.

![Reference accuracy before oracle pruning](fiber_benchmarks/imgs/fiber_crop_reference_accuracy.svg)

The pre-pruning reference plot uses
`100 * exact / (exact + wrong)` at oracle round zero. Missing references have
no estimate and are excluded from the fraction. The fixed-quarter baseline is
80%; no-overtrace at density `0.35` is 84%; and the ordinary stored cohort
filtered at density `0.25` is 88%. These are reference-tuned measurements on
the same crop, not held-out validation results.

The horizontal coordinate is the historical algorithm completion date, not the
later benchmark execution date. Each data row separately records algorithm and
measurement revisions. Regenerate all deterministic SVGs with:

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
| 2026-09-03 | `a5e6f5d49+` | PHercParis4 1024, baseline traces, density <= 0.25 | 286 / 1221 | 40.00% | 61.66% | 24 exact, 0 wrong, 1 missing | 40.99 s | [record](fiber_benchmark_runs/2026-09-03-a5e6f5d49-baseline-q025-oracle-pruning.md) |
| 2026-09-03 | `a5e6f5d49+` | PHercParis4 1024, no-overtrace, density <= 0.35 | 141 / 807 | 30.00% | 49.63% | 24 exact, 0 wrong, 1 missing | 27.11 s | [record](fiber_benchmark_runs/2026-09-03-a5e6f5d49-no-overtrace-q035-oracle-pruning.md) |

The older pruning row uses a capped 1,998-trace unstaged cohort; the staged row
uses the complete uncapped 2,062-trace cohort. Their difference combines stage
filtering with cohort completion and is not a controlled causal comparison.

## Reference accuracy before pruning

| Date | Revision | Crop policy | Exact | Wrong | Missing | Exact / estimated | Run |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-09-03 | `a5e6f5d49+` | Fixed-quarter baseline | 20 | 5 | 0 | 80.0% | [record](fiber_benchmark_runs/2026-09-03-a5e6f5d49-no-overtrace-q035-oracle-pruning.md) |
| 2026-09-03 | `a5e6f5d49+` | Baseline traces, density <= 0.25 | 22 | 3 | 0 | 88.0% | [record](fiber_benchmark_runs/2026-09-03-a5e6f5d49-baseline-q025-oracle-pruning.md) |
| 2026-09-03 | `a5e6f5d49+` | No-overtrace, density <= 0.35 | 21 | 4 | 0 | 84.0% | [record](fiber_benchmark_runs/2026-09-03-a5e6f5d49-no-overtrace-q035-oracle-pruning.md) |

The denominator is the 25 references with an estimate at round zero. The
complete tagged stack contains one additional reference without sufficient
evidence in these runs.
