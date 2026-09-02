# Task Log

## Discovery

- The just-added BP `raw_w` is the result of a separate per-gauge raw scorer.
  The final `est_w` is selected again after gauge calibration and evidence
  aggregation, so the two columns can select different half-step candidates.
- The requested value is not another inference. It is the exact final
  `est_w` candidate expressed before benchmark sign/offset calibration.
- Solver OBJ files add `-min(active mapWinding)` to relative solver windings.
  The CLI table must add the same offset for `raw_w` to identify the actual
  `<base>_w_<index>_*.obj` layer.
- A source spanning multiple candidate-bearing gauges has no single inverse
  solver coordinate and must report `NA`.

## Deviations

- None.

## Independent plan review

- Corrected the initial inverse-calibration-only plan: that value is a latent
  half-step, not an integer OBJ layer. The implementation also removes the
  reference H/V class offset using the contributing component phase sign and
  selected phase.
- Gauge/component eligibility now follows all admitted candidate-bearing
  evidence used by the final scorer rather than the independently selected raw
  calibration votes.
- The existing publication-range calculation was extracted and reused so OBJ
  naming and diagnostic output share the same output-offset rule.

## Implementation

- The final calibrated `est_w` remains the only per-reference candidate
  selection. Its stored raw latent coordinate is now computed algebraically as
  `globalSign * est_w + gaugeOffset` for one contributing gauge.
- The compact BP table converts that latent value to integer `mapWinding` with
  the independently calibrated reference H/V class, contributing component
  phase sign, and selected phase, then adds the solver artifact output offset.
- Missing estimates and ambiguous gauges/components print `NA`; off-grid
  latent-to-integer conversion is an invariant error. Ceres reporting is
  unchanged.

## Validation

- Release build:
  `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_fiber_trace_winding_bp -j 16`.
- Focused C++ tests: 82 cases passed. Added a sign-reversed tie regression in
  which the obsolete independent raw scorer chooses `2.5`, while final
  `est_w=1.0` inverse-maps correctly to raw latent `3.0`.
- The Clang system-dependency build succeeded and the same 82 focused cases
  passed.
- Viewer tests: 41 cases passed with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src pytest -q vesuvius/tests/test_view_fiber_windings.py`.
- The standard 1024 Release diagnostic completed in 15.58 seconds. It reported
  relative solver winding range `-15..13`, published range `0..28`, and finite
  BP `raw_w` values only as integer published layers (`15`, `16`, ...), while
  retaining calibrated half-step `est_w` values (`0.0`, `0.5`, ...).
