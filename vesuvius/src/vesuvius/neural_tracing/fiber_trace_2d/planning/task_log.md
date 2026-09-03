# Task Log

## Scope

- Add two percentage-versus-time step plots without changing either benchmark.
- Keep all new work uncommitted at the user's request.

## Decisions

- Use algorithm completion dates rather than the later benchmark execution
  date: Lasagna transport `2026-06-01`, greedy direct `2026-07-30`, and the
  current Fiberlet replay default `2026-08-22`. The crop Fiberlet-plus-BP point
  instead uses `2026-09-02`, when reference-conditioned pruning was completed.
- Replay uses the recorded distance-per-failure percentage directly.
- Crop preserves and negates the recorded
  `100 * problematic / retained_fulfilled` ratio: Fiberlet is `-177.9331%`
  from 44,284 problematic and 24,888 retained fulfilled constraints. Direct
  greedy and Lasagna are visibly marked as unmeasured floor points without
  numeric metric values.

## Independent Review

- Separate historical algorithm provenance from later measurement provenance.
- Derive plot scores from raw fields, preserve the exact metric names, validate
  their domains, and retain cohort identity. The review initially proposed a
  bounded fulfilled fraction; the user corrected this to the benchmark's
  existing problematic/retained ratio before finalization.
- Treat crop floor markers as assumptions in a separate visual style and do
  not imply they were crop measurements or give them fabricated metric values.
- Date the crop point to the complete BP/pruning method rather than the earlier
  Fiberlet replay milestone. Keep the two percentage families on separate axes
  and state that they are not numerically comparable.

## Implementation

- Added `volume-cartographer/docs/fiber_benchmark_plot_data.json` with raw
  benchmark counts, cohort identity, separate algorithm/measurement
  provenance, run-record links, and explicit assumed-floor rationales.
- Added `volume-cartographer/scripts/plot_fiber_benchmarks.py`. It validates
  source data, derives scores, orders algorithm milestones, and emits two
  deterministic accessible SVG step plots. Measured and assumed crop points
  differ by marker, fill, color, and line style.
- Embedded and documented the plots in the benchmark index. No benchmark
  implementation, recorded run, or result value was changed.

## Validation

- `python volume-cartographer/scripts/plot_fiber_benchmarks.py --check` derives
  replay scores `1.754386%`, `7.692308%`, and `16.666667%`, plus two unmeasured
  crop floors and the Fiberlet crop score `-177.933140%`.
- Generated both SVGs twice; SHA-256 remained
  `9463e3197aeeb38c14c68e01ae1020a718a79c4d44a8f6113c630f95adc3590f`
  for replay and
  `aecc709ead62b053c866687846cbf7f27e60f3dad17527754a385a3786924f7b`
  for crop.
- Python compilation, XML parsing of both SVGs, and `git diff --check` passed.
- Visual inspection used temporary PNG conversions and confirmed readable
  labels, distinct assumed markers, and non-overlapping plot content.
- The user reviewed the corrected `-177.933140%` crop-error presentation and
  requested that the completed visualization changes be committed.
