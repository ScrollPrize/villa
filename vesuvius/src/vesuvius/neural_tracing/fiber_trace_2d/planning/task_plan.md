# Plan: estimate-first reference winding calibration

## Implementation

- Group active admitted reference observations by integer gauge and infer one
  raw winding per `(reference source, gauge)` with the existing dominant-factor
  scorer, hard violations first and finite energy second, in an identity frame.
- Fit global sign `+1/-1` and one half-integer offset per gauge from those raw
  estimates only. Each estimate is one vote. Candidate offsets are the exact
  half-step differences between a raw estimate and the signed reference label.
  Maximize exact matches and then minimize total absolute residual. Prefer
  global sign `+1` on a remaining sign tie; choose per-gauge offset ties by
  smaller absolute offset and then lower offset.
- Exclude gauges with no raw admitted estimate. Apply the selected mapping to
  individual observation accuracy, group truth losses, and final calibrated
  estimates. Retain `+/-0.5` only in the final per-constraint right/wrong count.
- Keep the post-calibration `all` scorer as the displayed estimate so references
  spanning multiple calibrated gauges combine all admitted evidence.
- Extract one calibration-independent dominant-factor scorer and use it for
  both raw and calibrated inference. Raw inference must not read reference
  truth, reporting tolerance, or any calibration state.
- Preserve the diagnostic population after calibration: calibration itself
  uses admitted observations only, while right/wrong tables continue to report
  all candidate-bearing observations from calibrated gauges. Rename gauge
  columns to make their estimate-vote/exact-match meaning explicit.

## Tests

- Replace observation-level calibration expectations with estimate-level
  fixtures showing that high-degree references receive one calibration vote,
  half-step offsets resolve the H/V ambiguity exactly, the tolerance cannot
  influence calibration, reversed sign remains supported, and uncalibrated
  gauges are excluded.
- Verify group `all.infer_w` still equals `est_w`, build the production CLI,
  run focused winding tests, run the established 1024-crop smoke, and run
  `git diff --check`.
- Cover multiple gauges with different nonzero offsets, a separate source that
  resolves global sign, a source spanning gauges whose final combined estimate
  differs from an individual raw vote, and zero-admitted gauges.
- Verify raw estimates are unchanged when only reference truth or reporting
  tolerance changes, and verify a half-step miss is excluded from exact
  calibration matching but can still count as right in the reporting table.

## Spec Update

- Specify raw per-reference/gauge inference, exact half-step calibration votes,
  tie-breaking, uncalibrated-gauge exclusion, and reporting-only tolerance.

## Docs Updates

- Replace the observation-interval calibration description and explain the new
  estimate-first order and equal weighting per reference/gauge.

## Changelog

- Record the estimate-first winding calibration and removal of tolerance from
  calibration.
