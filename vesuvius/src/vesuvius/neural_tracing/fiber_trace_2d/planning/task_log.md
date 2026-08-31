# Task log: estimate-first reference winding calibration

## Findings

- Current calibration maximizes raw observation hits inside an inclusive
  `+/-0.5` interval. High-degree references therefore dominate calibration and
  half-step-shifted offsets can tie the correct ladder.
- The shared group scorer can infer gauge-local estimates in an identity frame;
  no duplicate objective implementation is required.

## Deviations

- None.

## Independent review

- Use one shared calibration-independent scorer for raw and calibrated
  inference; raw estimates must not read truth or tolerance.
- Use exact half-step estimate votes for calibration and exclude gauges with no
  admitted evidence.
- Select global sign by exact matches, then residual, then prefer `+1`; summed
  offset magnitude must not choose sign because it depends on gauge origin.
- Preserve all candidate-bearing observations from calibrated gauges in the
  final accuracy diagnostic, while calibration itself uses admitted evidence.
- Cover multi-gauge aggregation, empty gauges, truth independence, and the
  distinction between exact calibration matches and tolerant reporting.

## Validation

- `cmake --build volume-cartographer/build --target
  test_fiber_trace_winding_bp vc_fiber_trace_chunk -j 16`
- `volume-cartographer/build/bin/test_fiber_trace_winding_bp`: 46 test cases
  passed.
- Established 1024-crop `direction-ablation` diagnostic completed with 1,360
  retained pieces and 69,172 BP factors. Estimate-first calibration selected
  global sign `+1`, gauge offset `0.5`, and 5 exact matches from 8 equal-weight
  `(reference, gauge)` estimates. The separate reporting population contained
  2,874 constraints.
- `git diff --check`
