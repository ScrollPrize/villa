# Plan: Joint adaptive-grid winding BP

## Solver boundary

1. Preserve aligned-normal construction and signed winding-constraint
   extraction unchanged. The resulting signed deltas are fixed observations
   for both winding solvers.
2. Introduce an explicit winding-solver selection with two values:
   `joint-grid` (default) and `alternating` (the current implementation).
   Selection is valid only for the Mixed-state winding path. The legacy mode
   must retain its existing outputs and numerical behavior.
3. Keep the existing alternating solver API available. Add a separate joint
   solver entry point and shared preparation/report helpers rather than
   copying constraint preparation, energy, or report projection.

## Joint model

1. Use one local discrete state per represented piece:
   `(H | Mixed | V, integer winding)`.
2. Consume the original orientation evidence directly in the joint pair
   factors. Parallel evidence favors equal H/V states and perpendicular
   evidence favors different H/V states. Mixed remains a local uncertainty
   state: its unary cost is applied once, and an incident Mixed endpoint does
   not transmit an H/V preference.
3. Add the signed winding energy to the same pair factor. Same-trace split
   continuity remains the existing finite canonical parallel-score-1,
   zero-distance factor; it is not equality or variable collapse.
   Preserve the existing Mixed winding marginalization semantics unless a
   focused test proves that a different neutralization is required.
4. Fix the integer and H/V gauge at the established central seed of each
   constraint component. Give each component an explicit binary ladder-order
   variable `sign_c in {-1,+1}`. It is shared by every factor in that component
   and independent between components. Do not reinterpret it as an
   aligned-normal sign.
5. Use the existing continuous winding estimate only to center conservative
   integer candidate support. It must not determine H/V/Mixed, phase, scale,
   or the final integer labels.

## Global adaptive calibration

1. Parameterize calibration by positive gain `g = 1 / scale` and canonical
   phase `r = phase`, with `h = g*r` used when evaluating factors. Index gain
   uniformly in `u = log(g)` so a bounded-size window can move toward any
   positive finite scale without approaching a terminal zero-gain lattice
   cell. Index phase on a fixed absolute lattice over `0 <= r <= 0.5`.
2. For non-Mixed endpoint classes, define `b(H)=0`, `b(V)=1`,
   `delta = integer_delta + sign_c*r*class_delta`. For every measurement use
   the complete existing winding energy
   `parallel*abs(delta) + perpendicular*abs(g*delta-signed_measurement)` when
   signed evidence exists, omitting only the signed term when it does not.
   Also apply the orientation energy once: a non-Mixed same-class assignment
   costs `perpendicular`, and a different-class assignment costs `parallel`.
   This is intentional because there is no separate H/V factor pass in the
   new mode. A Mixed endpoint neutralizes that orientation energy and uses the
   existing normalized four-substitution winding marginalization; the Mixed
   unary is applied once at its node.
3. Represent calibration with one explicit crop-global latent variable `C`
   over the active `(u,r)` cells. Each constraint factor is coupled to its two
   piece variables, `C`, and its component's `sign_c`. Aggregate factor-to-C
   evidence from all components in deterministic order and broadcast the
   resulting C-to-factor message. Do not emulate C by independently replicated
   piece states, which would permit disconnected components to calibrate
   separately or double-count evidence.
4. Run all active calibration cells inside the same synchronous BP process.
   Calibration cells are states of one model, not separately completed solver
   runs. Exploit the calibration-block structure so forbidden cross-cell
   transitions are not evaluated quadratically.
5. Initialize the gain window around the current nominal scale 1 and cover the
   complete canonical phase interval unless a tested smaller phase window is
   explicitly configured. After each synchronized message iteration, form the
   global calibration posterior from all components. Shift gain support by at
   most one absolute log-gain cell only when entering-side boundary mass
   exceeds its threshold and leaving-side mass is negligible. If both sides
   retain material mass, grow the window within the resource guard instead of
   discarding either side; fail explicitly when growth is impossible.
6. Retain overlapping calibration messages after renormalizing their additive
   log-message gauge. Initialize a newly exposed cell from the configured
   calibration prior with neutral incoming cavity messages; it must not inherit
   the probability of the cell whose storage slot it replaces. Any support
   change resets the no-shift convergence counter.
7. Use one conservative integer-support union valid for every active
   calibration cell. Integer boundary probability marginalizes orientation,
   calibration, and component sign before requesting expansion. Calibration
   shifts that invalidate this union expand it without restarting retained
   messages. The resource guard counts the complete piece-state, calibration,
   sign, factor-message, and integer-support product.
8. Convergence requires message residual convergence, calibration-posterior
   stability, negligible integer-boundary mass, and a configured number of
   consecutive no-shift iterations. Adaptive support is one warm-started
   inference lifecycle over changing truncation support, not multiple
   independently initialized BP solves. No calibration update may restart BP.
9. Report the MAP and posterior-mean phase/scale, active absolute grid bounds,
   grid shifts, boundary mass, calibration entropy, joint message iterations,
   convergence state, solver mode, and posterior/MAP for every component sign.
   Compute posterior means as `E[r]` and `E[1/g]`, not ratios or transforms of
   coordinate means. Do not perform a label-changing post-fit. A finite
   message-limit result remains usable but is explicitly labeled nonconverged,
   matching the existing output policy; resource or validity failures throw.

## CLI and reporting

1. Add `--winding-solver joint-grid|alternating`, defaulting to `joint-grid`.
2. Add only the calibration-grid controls needed for reproducible experiments:
   dimensions, gain/phase lattice spacing, boundary threshold, and maximum
   shifts/resource guard. Defaults must be documented and validated.
3. Give the joint solver one coherent progress stream: message iteration,
   residual, state count, calibration MAP/mean, boundary mass, grid bounds,
   shifts, elapsed time, and convergence. Do not display legacy initialization
   or calibration-pass counters in this mode.
4. Persist the selected solver and calibration diagnostics in the existing CSV
   report. Keep winding OBJ naming and display-offset conventions unchanged so
   the visualization pipeline remains compatible.

## Correctness and resource handling

1. Validate every grid coordinate and energy as finite, enforce the phase
   wedge, and fail explicitly on resource-guard exhaustion.
2. Keep update ordering deterministic for a fixed worker count. Parallelize
   independent edge/calibration blocks without changing reduction order used
   for global posterior normalization.
3. Avoid state copying in the calibration dimension. Reuse contiguous message
   storage and retain overlapping blocks when the window shifts.
4. Do not silently fall back to the alternating solver. A failed joint solve
   must report its own failure and diagnostics.

## Testing

1. Retain all existing alternating-solver tests and add an explicit legacy-mode
   regression proving it remains selectable and unchanged.
2. Add brute-force enumeration comparisons on tiny acyclic graphs over every
   H/Mixed/V, integer, component-sign, and calibration state. Compare complete
   marginals and energies, including canonical reversal, normalized Mixed
   substitution, and same-trace split continuity. Do not claim exact loopy-BP
   marginals on cyclic graphs.
3. Add tests in which orientation evidence alone is ambiguous but signed
   winding evidence resolves H/V, proving there is no hidden orientation BP
   pre-pass.
4. Add multi-component tests proving that all components share one global
   calibration posterior while retaining their legal local gauges.
5. Force calibration outside the initial window and verify deterministic
   multi-shift movement, retained-cell physical identity and normalized
   message equivalence, canonical phase boundaries, entering/leaving boundary
   rules, convergence reset, and resource-guard errors.
6. Compare `joint-grid` against `alternating` on the established 384 and 1024
   crop artifacts. Report command, Release build, pieces/factors, elapsed and
   CPU time, iterations/state count, calibration, H/V/Mixed counts, and winding
   output differences. This is an algorithm comparison, not a bitwise-equality
   requirement.
7. Run focused C++ tests, the relevant CLI smoke test, `git diff --check`, and
   the existing winding visualization artifact loader.

## Spec update

After implementation, update `planning/specs.md` to define the two solver
modes, default selection, joint factor semantics, absolute sliding calibration
grid, global calibration posterior, convergence/resource behavior, reporting,
and the unchanged aligned-normal preprocessing contract. Do not describe the
new solver there as implemented before its code and tests exist.

## Docs updates

After implementation, update `volume-cartographer/docs/fiber_chunk_tracing.md`
with the default joint solver, the explicit legacy comparison command, grid
parameters, progress fields, output provenance, and interpretation of phase,
scale, normal sign, and component ladder gauge.

## Changelog

After implementation and validation, add a dated changelog entry describing
the default joint adaptive-grid solver and retained alternating comparison
mode. Planning alone does not receive an implementation changelog entry.
