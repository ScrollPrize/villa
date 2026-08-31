# Task log: direct winding-factor weight search

## Findings

- The current solver exposes only the dominant perpendicular/parallel score,
  the canonical-distance decay, and the optional parallel cutoff. The five
  displayed reference groups are not independently tunable.
- Reference observations currently reconstruct their coefficients separately;
  the new class multipliers therefore must be shared with that reconstruction
  or search scores would not match solver energy.
- The reference cross-constraints, BP topology, and optional fixed orientation
  prepass already exist before winding inference, so a direct search can reuse
  all expensive geometry and volume work.

## Deviations

- None.

## Independent review

- Restrict multipliers to strictly positive values so component connectivity,
  the retained component, topology, and fixed orientation prepass remain valid
  for every reused search scenario.
- Keep hard sign, orientation energy, Defect unary, hard continuity, and piece
  breaks outside the new scaling; include class scaling in factor diagnostics
  and reference losses.
- Use the signed parallel target as the authoritative canonical class whenever
  it exists, matching solver preparation.
- Rank on a fixed reference-source denominator before per-constraint metrics so
  making difficult endpoints Defect cannot win through abstention.
- Propagate the selected tuple into final diagnostics/output and isolate failed
  scenarios rather than aborting the complete search.
- For local search, use exact exponent-offset cache keys, cache failures, and
  keep a bounded numeric domain. Do not let residual or lexicographic tuple
  tie-breaks count as strict quality improvement, because that would drift a
  flat objective toward zero. Report an iteration-limit exit as failure and
  ensure only the selected tuple drives final artifacts.
- For default promotion, centralize the tuple rather than duplicating literals,
  make neutral-weight arithmetic tests explicit, document that raw-integer
  winding remains unscaled, and state the limited supervised validation scope.

## Validation

- `cmake --build volume-cartographer/build --target
  test_fiber_trace_winding_bp vc_fiber_trace_chunk -j 16`
- `volume-cartographer/build/bin/test_fiber_trace_winding_bp`: 48 test cases
  passed.
- Established 1024-crop reference sweep, fixed orientation, fixed phase `0.5`,
  scale `1`, Defect cost `50`, 500 message iterations, no parallel cutoff,
  grid `{1,2}`: 32 scenarios completed in one process. The all-ones baseline
  produced 5/8 exact reference estimates and 1,379/2,881 (47.865%) correct
  reference constraints. The selected tuple `2,1,2,2,1` produced 6/8 exact
  estimates and 1,526/2,859 (53.375%) correct constraints; all references had
  an estimate and the solve converged.
- The explicit all-ones per-reference rerun showed errors `1.5 -> 2.5`,
  `2.0 -> 3.0`, and `2.5 -> 3.5`. The selected tuple corrected the `1.5`
  source; the latter two remained one winding high.
- Added `/2`, `*2` coordinate descent from the selected `2,1,2,2,1` tuple,
  stopping when no single-coordinate neighbor improves the same
  fixed-denominator ranking. The local search evaluated 29 unique tuples and
  accepted two
  improvements: `2,1,2,2,1 -> 4,1,2,2,1 -> 8,1,2,2,1`. Exact reference
  estimates improved `6/8 -> 8/8 -> 8/8`; correct reference constraints
  improved `1526/2859` (53.375%) to `1631/2858` (57.068%) and then
  `1784/2839` (62.839%). No one-coordinate `/2` or `*2` neighbor of the final
  tuple improved the quality objective. Several neighbors tied it exactly;
  the search correctly stayed at the current tuple rather than drifting on a
  residual or tuple-order tie-break.
- Promoted the selected `8,1,2,2,1` tuple through one exported shared constant
  used by both the public winding configuration and CLI defaults. Existing
  neutral-arithmetic tests now explicitly request unit class weights. The
  promoted default is supervised on this 1024-crop, fixed-orientation setup;
  adaptive calibration and other datasets have not yet been benchmarked.
- A preliminary cutoff-`0.5` sweep was stopped after four cases because it
  correctly demonstrated that `parallel_1` and `parallel_2+` were suppressed,
  making those search dimensions inert. The production sweep removed the
  cutoff so all five classes participated.
- `git diff --check`.
- After promoting the default, the focused suite passed 49 cases and CLI help
  reported `[8,1,2,2,1]`. The public-config regression checks the same exported
  tuple used by CLI option initialization.
