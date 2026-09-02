# Task Plan

## Shared conflict classes

1. Add a diagnostic-only reference factor classification with seven classes:
   perpendicular magnitude 0.5, perpendicular magnitude 1.5+, perpendicular
   sign, parallel magnitude 0, parallel magnitude 1, parallel magnitude 2+,
   and parallel sign.
2. Evaluate both paths through one shared factor helper consuming already
   materialized solver targets, weights, penalties, and hardness. Do not
   reconstruct confidence or apply measurement scale a second time.
3. Classify from the effective canonical target. Keep one sign row and one
   magnitude row for a constraint when both factors are active. Count
   magnitude only for positive effective magnitude weight and sign only when
   hard or assigned a finite penalty.
4. Preserve exact BP predicates and losses: signed magnitude uses
   `abs(delta-target)`, unsigned parallel magnitude uses
   `abs(abs(delta)-distance)`, and sign conflicts use `target*delta <= 0`.
   Hard violations remain a separate count and contribute only an actual
   finite sign penalty, which is normally zero for a promoted hard factor.
5. Reuse the same class names and aggregate formatter for both requested
   tables.

## Reference-to-reference evaluation

1. Evaluate the already extracted local reference-piece constraints; do not
   create a second extraction path or collapse the piece-pair observations.
2. Exclude generated hard-continuity and other same-source links from this
   cross-reference table.
3. Use the solver-prepared factor diagnostics so admission, confidence,
   class weights, finite sign penalties, and hard-sign promotion match the BP
   model.
4. Fix both endpoints to their filename-ordered half-step winding values. The
   fixed predicted delta is `globalSign * (W_b - W_a)` in the factor's
   canonical endpoint order; a common gauge cancels. Measurement scale has
   already been applied to the measured target and must not affect these known
   endpoint labels.
5. For each admitted magnitude or sign factor, report conflict, hard
   violation, and the same weighted L1/sign loss used by the corresponding BP
   factor.
6. Add `--reference-constraint-details` to enable the existing long
   per-reference constraint listings. Suppress only those listings by default;
   retain calibration and aggregate summaries.

## Output-layer mapping

1. Retain the exact inverse mapping of the final calibrated `est_w` candidate.
2. When an otherwise valid calibrated half-step candidate lands exactly on
   the opposite H/V ladder after class-offset removal, treat it as incompatible
   orientation evidence and return no output-layer estimate, producing `NA`
   in `raw_w`.
3. Continue throwing for arbitrary off-half-step values, malformed phase
   signs, non-finite inputs, and integer overflow.

## Validation

- Extend focused C++ tests for all seven conflict classes, separated sign and
  magnitude factors, hard-sign accounting, disabled factors, and weighted
  losses for reference-to-BP and reference-to-reference diagnostics.
- Add an opposite-ladder half-step test that requires an absent output result;
  preserve the arbitrary off-lattice exception test and exact mapping tests
  for both H/V classes and phase signs.
- Build Release and Clang targets and run the focused winding BP test.
- Run the approved 1024 diagnostic command and verify that both conflict
  tables print and an incompatible `raw_w` no longer aborts the run.
- Verify that the default output omits individual constraint rows and the new
  flag restores them.

## Spec Update

- Specify the seven-class conflict summaries and reference-to-reference fixed
  endpoint semantics in `volume-cartographer/planning/spec.md`.
- Clarify that orientation-incompatible output-layer inverses print `NA`.

## Docs Updates

- Document both conflict tables and their class/factor meanings in
  `volume-cartographer/docs/fiber_chunk_tracing.md`.

## Changelog Update

- Record class-resolved reference conflict diagnostics and non-fatal
  incompatible output-layer reporting.
