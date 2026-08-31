# Plan: direct winding-factor weight search

## Semantics and solver integration

- Set the shared winding configuration and CLI default tuple to
  `8,1,2,2,1`. Keep explicit `--winding-weights` and both search modes as exact
  overrides of that default.
- Add five finite strictly positive class multipliers to the shared winding BP
  configuration. Apply them after dominant-hypothesis selection and after the
  existing distance decay. Hard continuity remains weight `1`.
- Classify canonical targets identically in solver factors and reference
  diagnostics: perpendicular `0.5` versus `1.5+`, and parallel `0`, `1`, or
  `2+`. A signed parallel target is authoritative over the unsigned measured
  distance in both paths.
- Pass the complete winding configuration into reference-observation creation
  so its diagnostic loss is exactly the loss used by a corresponding solver
  factor, including cutoff, decay, and the new multiplier.

## CLI and search

- Add `--winding-weights P05,PFAR,P0,P1,P2` for one fixed tuple.
- Add `--winding-weight-search V0,V1,...` for an exhaustive Cartesian grid over
  all five classes. Require the existing reference-fiber input and mixed BP.
- Add `--winding-weight-search-local` together with `--winding-weights` for
  multiplicative coordinate descent. Evaluate all ten one-coordinate `/2` and
  `*2` neighbors, cache exact exponent tuples and failures within the process,
  accept only a strict benchmark-quality improvement, and stop at a local
  optimum. Residual and tuple-order tie-breaks must not cause a move. Bound the
  exponent domain and iteration count and report per-neighborhood progress and
  ETA; reaching the guard is an error.
- Reuse extracted constraints, topology, orientation prepass, and reference
  cross-constraints. Run only winding inference for each tuple, retain the best
  report, and use its tuple for every normal diagnostic and visualization
  output. Strictly positive weights preserve factor connectivity, making this
  reuse valid.
- Print per-scenario progress with ETA and a compact result row containing the
  five weights, exact calibrated reference estimates, constraint right/wrong
  totals and fraction, active reference estimates, solver status, and time.
- Rank converged results first and then use a fixed reference-source
  denominator: exact calibrated reference estimates, fewer missing/incorrect
  estimates, more correct and evaluated constraints, and fewer wrong
  constraints. Residual and lexicographic tuple ordering are reporting and
  neighbor-selection tie-breaks only. A Defect endpoint therefore cannot
  improve the primary objective by abstaining.
- Catch and report a failed tuple without aborting the remaining grid, reject
  empty/duplicate-invalid/oversized grids, and fail only if no tuple solves.

## Tests

- Assert the promoted public and CLI defaults. Make tests of neutral
  unit-weight arithmetic explicitly request `1,1,1,1,1` instead of inheriting
  the production default.
- Unit-test all five target-class boundaries, continuity invariance, multiplier
  composition with distance decay, signed parallel authority, validation, and
  reference diagnostic parity. Hard signs remain independent of finite positive
  multiplier magnitude; class weights do not scale orientation, Defect unary,
  hard continuity, or piece-break terms.
- Add CLI parse/validation coverage where available; otherwise smoke the built
  CLI with the established 1024 crop, a deliberately small grid, and the local
  search starting from the best grid tuple.
- Build `vc_fiber_trace_chunk`, run focused winding tests, and run
  `git diff --check`.

## Spec Update

- Specify the five class multipliers, their order, exact classification,
  promoted default, and supervised search/ranking semantics.

## Docs Updates

- Document all search CLI options, output columns, graph reuse, and the intended
  use of the reference fibers as a supervised diagnostic set.

## Changelog

- Record configurable winding-factor class weights, direct reference grid
  search, multiplicative local search, and the promoted standard tuple.
