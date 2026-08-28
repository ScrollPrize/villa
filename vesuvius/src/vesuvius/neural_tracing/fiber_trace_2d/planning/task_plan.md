# Plan: extended distance-weighted winding constraints

## Constraint admission

1. Change the H/V diagnostic/BP finite raw winding-distance default from the
   exclusive cutoff `1.5` to `4.0` and update CLI help.
2. Keep cutoff admission before H/V half-integer conversion. Values at or above
   `4.0` remain rejected.
3. Preserve the shared API and legacy parity-labeling default at `1.5`. Select
   the H/V default explicitly in CLI modes that can represent larger offsets,
   unless the user supplied a cutoff override.

## Winding evidence weighting

1. Derive a fixed multiplier from the absolute effective half-integer target:
   `2^-floor(|target|)`, producing `1`, `0.5`, `0.25`, and `0.125` for the
   admitted `0.5` through `3.5` bins.
2. Store a winding-factor multiplier separately from the original parallel and
   perpendicular H/V scores in the prepared factor representation.
3. Multiply both the parallel same-winding term and perpendicular signed-offset
   term by the multiplier. Use it consistently in continuous initialization,
   alternating winding energy/calibration, joint-grid winding energy, hard-sign
   applicability, and winding residual diagnostics.
4. Do not decay H/V orientation energy, continuity, or the independent raw
   integer-only winding diagnostic. Keep hard signed-order admissibility based
   on the original nonzero signed observation, independent of soft decay.
5. Continue the formula for explicit cutoffs beyond `4.0` using an underflow-
   safe power-of-two implementation.
6. Expose the multiplier and effective parallel/perpendicular winding weights in factor diagnostics so
   the experiment is inspectable.

## Testing

1. Verify H/V CLI default selection is `4.0`, the shared/parity default remains
   `1.5`, and extraction accepts values below its configured cutoff while
   rejecting the exact boundary and above.
2. Verify H/V-aware factor diagnostics and winding energies use the exact multipliers
   for signed targets in all four admitted bins, including endpoint reversal.
3. Verify the original perpendicular H/V weight remains unchanged while the
   separate signed-winding weight decays, and that the raw integer-only
   diagnostic remains unscaled.
4. Run the focused constraint and winding-BP tests in the Release build, build
   the CLI with 32 jobs, run the representative fixed-calibration crop, and run
   `git diff --check`.

## Spec update

Change the default exclusive raw cutoff to `4.0` for representable H/V winding
inference and specify the fixed power-of-two decay of signed winding evidence
after half-integer conversion. State that orientation evidence is not decayed
and legacy parity labeling retains its own default and representability limit.

## Docs updates

Update the CLI default, cutoff examples, half-integer table, weighting formula,
and diagnostic fields in `volume-cartographer/docs/fiber_chunk_tracing.md`.

## Changelog

Record the wider winding evidence range and distance-dependent winding weight.
