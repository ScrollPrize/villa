# Task Log

## 2026-08-07

- Started documentation and workflow follow-up for calibration/reference/
  tolerance maintenance and runner-derived Ninja concurrency.
- The prior implementation hardcodes `--parallel 4` in GitHub Actions and local
  examples. The CMake artifact graph itself has no concurrency assumption.
- The checked reference already stores `tolerance: 0.1`, but evaluation also
  defaults independently to a Python `TOLERANCE` constant. This duplicate
  policy value will be removed from normal CI evaluation.
- Independent review required ordinary VC3D usage/failure documentation,
  scoping parallelism validation to the rendering job, making the runtime
  target runner-size-aware, and explaining path-filter versus branch-protection
  activation. The plan now includes all four points.
- Added `docs/benchmarks/render_valgrind_ci.md` as the VC3D operational guide
  and linked it from the synthetic-rendering design document. It separates
  routine use from model recalibration, eight-case score refresh, and
  tolerance-only policy changes, and documents path filters plus branch-rule
  enforcement.
- Added `freeze-model` to validate synthetic-only provenance, the exact
  seven-feature data-read basis, coefficient count, and cross-thread release
  parameter. Unaccepted experimental calibrations require an explicit
  `--allow-unpromoted` decision. The command reproduced the current compact
  model exactly from `/tmp/thread-sync-event-features-v4-data_reads/model.json`.
- Normal evaluation now reads tolerance from the checked reference. An
  explicit `--tolerance` remains diagnostic-only, while `freeze-reference
  --tolerance` is the supported policy update path and validates `[0, 1)`.
- The rendering workflow uses `jobs=$(nproc)` for both benchmark compilation
  and the Valgrind Ninja graph. Fixed renderer workers/replay cores remain part
  of the workload model and are unchanged.
- Added atomic `set-tolerance`; a 0.05 trial copy retained an identical
  canonical hash for every field except `tolerance` and kept all eight cases.
- Validation passed: 98 benchmark Python tests, focused Ruff format/lint,
  Python byte compilation, workflow YAML parsing, and `git diff --check`.
  A complete fresh gate using `--parallel "$(nproc)"` finished in 9.8 s; serial
  ratios were 1.000 and parallel ratios were 1.002, 0.999, 1.010, and 0.989.
