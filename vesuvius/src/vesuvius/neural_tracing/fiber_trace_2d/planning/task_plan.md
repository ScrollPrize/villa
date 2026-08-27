# Plan: Interleaved-lattice winding inference

## Scope and invariants

- Operate in the existing C++ crop constraint/BP path. Do not add Python
  bindings or a second constraint extractor.
- Keep every split piece as one variable. Same-trace continuity remains the
  existing parallel-score-1, zero-distance factor and is not variable collapse.
- Preserve constraint ordering, aligned-normal signed targets, unsigned legacy
  winding measurements, base-coordinate conventions, and deterministic output.
- Treat class names as a gauge. Fix the crop-central piece to local class A and
  integer winding zero in every disconnected component. Give every component a
  deterministic phase-sign gauge while sharing one phase magnitude and scale;
  do not claim physical H/V or compare absolute integer gauges across components.
- Keep explicit Mixed semantics from the established orientation BP. Feed its
  normalized A/Mixed/B marginals into winding inference as soft priors so the
  Mixed unary and orientation constraints are paid exactly once.

## Joint state and factor model

1. Represent an oriented piece state by `(class, k)`, where local class A has
   latent coordinate `k` and local class B has coordinate
   `k + component_sign*phase`. `k` is integer, shared phase magnitude is bounded
   to `[0,0.5]`, and the deterministic per-component sign represents the class-
   swap gauge without changing the fixed local-A seed.
2. Run the existing Mixed-state orientation BP first. Its per-piece
   A/Mixed/B posterior becomes the joint winding solver's node prior; the
   winding stage must not repeat the same/different orientation factor.
3. Represent Mixed as `(Mixed, k)`. If either endpoint is Mixed, omit the
   orientation same/different energy and define the winding potential as the
   normalized average over all four latent A/B endpoint substitutions:
   `psi=1/4*sum_ab exp(-E_winding(a,b)/T)`. Averaging rather than summing avoids
   an entropy bonus, and substituting both endpoints makes the result invariant
   to the non-Mixed endpoint's class label. Mixed therefore retains winding
   connectivity without transmitting orientation preference.
4. For latent coordinate difference `delta`, add
   `parallel*abs(delta)` and, when signed evidence `d` exists,
   `perpendicular*abs(delta/scale-d)`. This compares the predicted raw Lasagna
   integral with the observed one; multiplying `d` by scale would also shrink
   its noise and create a degenerate preference for the minimum scale.
5. Same-trace continuity uses this exact factor with parallel one and target
   zero. Repeated endpoint pairs sum complete measurement energies.
6. Keep adaptive integer candidate bounds. Boundary probability is summed over
   all class states for that integer before deciding expansion.

## Alternating global calibration

1. Run joint sum-product BP for fixed `phase` and positive `scale` using stable
   state/factor order, synchronous damping, and the existing worker controls.
2. Compute normalized pair beliefs from the directed messages. Refit inverse
   scale `g=1/scale` and `h=g*phase` from the expected squared raw-measurement
   residual `g*integer + sign*class_delta*h - d` over fully oriented pair
   beliefs; Mixed endpoint mass does not calibrate A/B phase. This quadratic is
   only a proposal for the authoritative expected L1 winding energy.
3. Solve the bounded two-parameter least-squares proposal over the exact wedge
   implied by phase `[0,0.5]` and scale `[0.5,2]`. Detect a rank-deficient normal matrix and
   retain the previous unidentifiable parameters deterministically. Otherwise,
   backtrack from the old parameters to the proposal until fixed-belief expected
   L1 does not increase. Stop when both accepted changes meet tolerance.
4. Use a small deterministic initialization grid spanning phase and plausible
   scale. Decode each result by per-node joint marginal argmax, score that exact
   assignment with the complete model energy, and select the lowest decoded
   energy. Do not call this loopy sum-product result an exact MAP solution.
   Stable ties use initialization order.
5. The additive winding gauge and phase sign are fixed independently per
   connected component. A nonconverged inner BP stops calibration for that
   initialization but remains a labeled diagnostic candidate; converged starts
   rank first. If every start is nonconverged, publish the lowest decoded-energy
   finite result with `message_limit`. Candidate expansion restarts BP
   deterministically; resource exhaustion remains a hard error.

## Integration and output

1. Add a joint interleaved solver in the existing fiber-tracer library and
   reuse prepared BP topology and constraint structures.
2. For `sum-product-mixed`, run established orientation BP followed by the
   interleaved winding refinement; never also run the independent winding
   solver. Replace displayed V/Mixed/H marginals with the refined joint result.
   Keep other experimental BP modes on their current path.
3. Report fitted phase, scale, alternating iterations, selected initialization,
   convergence, decoded joint energy, and candidate-state counts.
4. Add joint class probabilities and latent coordinates to
   `<base>_consistency.csv`. Keep consecutive `<base>_w_N_{h,v,err,tie}.obj`
   output grouped by integer `k` and short content-only names. Record component
   identity because independently gauged components have incomparable absolute
   `k`. Joint MAP fields refer to the selected `(class,k)` state; posterior
   winding fields marginalize class at each integer.
5. Extend the winding factor CSV with the calibrated signed target used by the
   selected solution.

## Spec update

Replace the independent integer-winding model with the interleaved A/B integer
lattices, explicit gauge symmetry, Mixed marginalization, global calibration,
soft pair-belief refit, deterministic multi-start selection, adaptive integer
support, output fields, and failure contracts. Remove language claiming that
all pieces lie on one integer scale.

## Docs updates

Update `volume-cartographer/docs/fiber_chunk_tracing.md` with the state
coordinate equations, factor energy, phase/scale interpretation, alternating
solver, gauge behavior, diagnostics, and unchanged command line.

## Testing

- Unit-test exact A/B fractional coordinates and the global class-swap gauge.
- Recover a synthetic `A_k -> B_k -> A_(k+1)` chain from measurements scaled
  to `0.8` total, including non-half phase.
- Verify same-class steps are integral, split continuity stays a factor, and
  conflicting evidence can change adjacent piece labels.
- Verify Mixed endpoints do not transmit orientation preference but retain
  finite winding beliefs.
- Verify soft calibration, bounds, deterministic multi-start ties, serial/
  parallel equality, adaptive support, malformed input, and resource failure.
- Verify rank-deficient/no-signed-evidence fallback, opposite phase signs in
  disconnected components, canonical half-period behavior, exact Mixed-factor
  class invariance, and finite nonconverged inner-BP reporting.
- Verify equivalence to the independent integer solver when phase is fixed to
  zero and class coupling is disabled.
- Run focused Release tests and the real 384-base crop; record before/after
  phase, scale, variables, factors, convergence, timing, and output layers.
- Run `git diff --check` and document all deviations.

## Changelog

Add a dated entry describing joint interleaved-lattice orientation/winding BP
and learned global winding calibration.

## Progress-reporting follow-up

1. Add a public progress event and an optional callback argument to the
   interleaved winding solver. Keep numeric configuration separate from
   observation. Report preparation, each multi-start initialization, each
   calibration round, adaptive-support restart, inner message iteration and
   residual, and final completion. Invoke callbacks only outside OpenMP
   regions after synchronous message swaps, preserving ordering and arithmetic.
2. Have `vc_fiber_trace_chunk direction-ablation` install a terminal renderer
   throttled to roughly one update per second, while forcing stage transitions
   and the final newline. Show exact nested counters, candidate-state count,
   residual, phase/scale, and elapsed time. Do not claim a global percentage:
   adaptive support, early convergence, and changing state counts leave no
   valid fixed work denominator. After the first calibration completes, report
   a clearly empirical maximum-slot ETA; after the first initialization,
   replace it with mean completed-initialization duration. Label the active
   basis and never describe either estimate as conservative.
3. Unit-test callback sequencing, nested counters, final completion, and
   serial/parallel result identity. Re-run focused winding and crop tests plus
   the real 384-base crop smoke run.
4. Update the winding spec, crop-tracing documentation, task log, status, and
   changelog. No new CLI option is required; progress is enabled for this CLI
   path and remains opt-in for library callers.

## Input-quality filtering follow-up

1. Extract the existing deterministic crop-quality ordering into a shared core
   helper. Quality remains total metric cost divided by prediction-voxel path
   length; lower is better and stored ordinal breaks ties. Keep decile
   visualization on this exact helper.
2. Add `--quality-fraction F`, requiring `0 < F <= 1`, to every stored-artifact
   consumer and reject it in trace mode. Retain `ceil(F*N)` traces for nonempty
   input, then restore their original artifact order before downstream work.
3. Apply filtering immediately after strict artifact loading. Report original
   and retained counts, requested/effective fraction, and worst retained cost
   density. Constraint extraction, direction fitting, BP, and visualization
   must see only the retained vector; crop metadata and source validation remain
   unchanged.
4. Test ranking reuse, deterministic ties, rounding/nonempty behavior, original
   order, invalid qualities, and unchanged full-fraction behavior. Rebuild and
   run focused crop and winding suites, update specs/docs/log/status, and run
   `git diff --check`.
