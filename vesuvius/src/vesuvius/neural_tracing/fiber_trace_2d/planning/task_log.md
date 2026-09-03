# Task log: baseline quality-threshold benchmark

## 2026-09-03

- The comparison uses the ordinary crop tracer with no `--stop-at-covered`.
- Thresholds must be applied online: rejected candidates do not create seed
  coverage, so filtering an old artifact is not an authoritative substitute.
- Selection uses pre-pruning exact / (exact + wrong) as the primary measure,
  while retaining wrong, missing, geometry, and final oracle-pruning statistics.
- The sweep and resulting benchmark point are reference-tuned on the 1024 crop,
  not an unbiased validation result.
- Independent review required the new plot protocol to be specified durably and
  recommended freezing all non-threshold settings and input identities. The
  plan now includes both requirements.
- An ordinary online threshold of `0.35` retained 1,857 traces, 5,265 pieces,
  and 981,686 constraints; the winding solver rejected it at the initial-state
  resource guard. An online threshold of `0.25` accepted 674 traces after
  13,915 attempts and 13,233 quality rejections, took 988.04 seconds wall, and
  produced 20 exact, five wrong, and zero missing references at oracle round
  zero. Online rejection was retained only as a diagnostic because it changes
  the baseline seed-coverage schedule.
- The authoritative baseline sweep filtered the existing complete ordinary
  artifact at BP load. Results `(threshold: retained, exact/wrong/missing)`
  were `0.245: 416, 22/3/0`; `0.250: 451, 22/3/0`; `0.2525: 468, 20/5/0`;
  `0.256482: 500, 20/5/0`; and `0.260: 525, 21/4/0`.
- Selected `0.250`: it ties the best 88% round-zero accuracy while retaining
  more traces than `0.245`. Oracle pruning retained 935/1,221 pieces, reached
  24 exact, zero wrong, and one missing, and reported 35,695 problematic versus
  22,198 retained fulfilled constraints (160.80%). Wall time was 40.99 seconds.
- Added the selected baseline threshold to the crop-error benchmark and added a
  third progress plot for pre-pruning `exact/(exact+wrong)`. The plot contains
  the 80% fixed-quarter baseline, 84% no-overtrace `0.35`, and 88% ordinary
  post-load threshold `0.25` points.
- Validation: `python volume-cartographer/scripts/plot_fiber_benchmarks.py
  --check`, a full render, `python -m py_compile` for the plotter, raster visual
  inspection of both changed plots, and `git diff --check` all passed. No C++
  source changed during this benchmark extension; the preceding feature task's
  Release `test_fiberlet_crop_trace` result remains 86 passing cases.
