# Plan: Fixed pre-pass orientation for winding BP

## Orientation contract

1. Add a winding-orientation mode with the existing joint behavior as the
   default and an explicit fixed-prepass mode.
2. In fixed-prepass mode, run the existing `sum-product-mixed` orientation BP
   once before either winding solver. Convert each normalized H/Mixed/V
   marginal to a hard class using deterministic MAP selection; an exact tie is
   classified as Mixed instead of arbitrarily selecting H or V.
3. Pass those fixed classes through a shared H/Mixed/V-only solver API used by
   both winding variants. Reject missing, size-mismatched, or invalid-enum
   class inputs; Tie is not representable in this API.

## Solver changes

1. In alternating winding inference, remove the three-way orientation
   dimension entirely in fixed-prepass mode: each piece state is only one
   candidate integer winding, decoded together with its stored fixed class.
   Keep winding support, phase/scale calibration, component sign, and factor
   potentials unchanged.
   Apply the fixed-class accessor consistently in message passing, pair-belief
   formation, calibration updates, decoded-energy ranking, and final marginal
   decoding so no later stage reconstructs three class variants.
2. In joint-grid inference, use the same winding-only piece-state layout.
   Calibration and component-sign states remain only when their existing mode
   requires them. Ensure each component gauge fixes only integer winding zero
   and retains the pre-pass class rather than implicitly changing it to class
   A.
3. Keep Mixed semantics unchanged: it is a fixed visible class, while its
   existing normalized latent endpoint marginalization remains the winding
   potential used by the solver.
4. Report one-hot H/Mixed/V marginals internally from the winding solve and add
   explicit joint versus fixed-prepass orientation provenance plus the selected
   class vector to its report. Preserve the first-pass soft marginals in the
   ordinary BP report and consistency CSV; also persist the fixed class per
   piece so diagnostics show both the evidence and the class actually used.
   Runs without the new mode retain current numerical behavior.

## CLI and output

1. Add `--winding-fixed-orientation`, valid only with
   `--bp-only --bp-inference sum-product-mixed` and either winding solver.
2. Refactor the direction-ablation driver so joint-grid runs the same existing
   orientation pre-pass when the option is enabled; alternating reuses the
   pre-pass it already performs.
3. Include orientation mode and fixed class in console/CSV provenance, retain
   first-pass soft H/Mixed/V probabilities, and make final H/V/Mixed OBJ output
   use the hard classes actually used for winding.

## Testing

1. Add deterministic MAP/tie conversion tests.
2. Add focused alternating and joint-grid tests proving winding cannot change
   fixed H, V, or Mixed classes, including a non-A component gauge, and verify
   candidate-state accounting reflects one winding state per integer rather
   than three orientation variants.
   Check piece messages and pair-belief work through exact state accounting so
   neither solver silently retains a factor-of-three class allocation. Mixed's
   four-substitution potential is factor evaluation only, never a latent state.
3. Retain existing tests proving default joint and alternating behavior is
   unchanged.
4. Add CLI help/validation smoke tests and run both solver variants end to end
   on the established crop using Release binaries built with 32 jobs.
5. Measure default versus fixed-prepass winding phases repeatedly on the same
   crop and Release build. Report commands, state counts, and min/median/max
   wall/solver times, with a focused profile or phase attribution showing the
   removed orientation-state work.
6. Run focused binaries, registered CTest, and `git diff --check`.

## Spec update

Document fixed-prepass orientation as a hard MAP class contract, the Mixed tie
rule, unchanged winding/calibration semantics, gauge handling, and explicit
provenance. State that the existing joint behavior remains the default.

## Docs updates

Update `volume-cartographer/docs/fiber_chunk_tracing.md` with the new flag,
both supported winding solvers, its two-stage behavior, hard-class semantics,
and example commands.

## Changelog

Add a concise entry after implementation and validation.
