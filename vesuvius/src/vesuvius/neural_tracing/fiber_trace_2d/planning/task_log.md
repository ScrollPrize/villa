# Task log: limit Fiberlet crop trace attempts

## Findings

- Crop seeds are already sorted by descending `predictionPresence`, then by
  stable storage key.
- `attemptedAnchors` increments after an uncovered seed is selected and before
  edge availability is checked, so both accepted and failed traces are
  attempts. Covered anchors are skipped without consuming the limit.
- Review confirmed the cap must be checked inside the active-seed loop rather
  than by truncating sorted candidates, and identified negative unsigned CLI
  counts as an existing parser edge case to close.

## Deviations

- None.

## Validation

- Built `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with 32 jobs.
- `test_fiberlet_crop_trace` passed, including capped presence/key ordering,
  covered-seed skipping, failed-attempt counting, and accepted-fiber limit
  interaction.
- Paris4 smoke with `--max-attempts 1` completed with exactly one attempt, one
  accepted bidirectional line, and 97 covered anchors.
- Negative `--max-attempts -1` is rejected while negative floating-point bbox
  coordinates remain valid.
- Removed the smoke-test output from `/tmp`.
