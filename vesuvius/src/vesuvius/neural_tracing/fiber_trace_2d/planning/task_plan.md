# Plan: confidence-weighted winding evidence refinement

## Semantics and data flow

1. Retain the existing dominant-hypothesis decision. Add a decision-confidence
   transform used only after that decision:
   - `legacy`: retain the selected normalized score in `[0.5,1]`;
   - `linear`: map `s` to `x = clamp(2*s-1,0,1)`;
   - `cosine`: map `x` to `(1-cos(pi*x))/2`, the standard cosine interpolation
     between zero and one.
2. Record signed connector alignment confidence inside the existing batched
   extraction path while its connector pairs are still available, without
   additional volume reads:
   - perpendicular evidence uses the absolute aligned-normal dot product at
     its closest connector;
   - parallel evidence uses the median absolute aligned-normal dot product over
     all admitted, component-compatible connector samples used to form the
     signed parallel-target sample set. Even medians average the two central
     values, matching winding-target median behavior.
   The later public reorientation helper cannot reconstruct the parallel walk;
   it records closest-connector perpendicular confidence and leaves parallel
   confidence unavailable rather than silently substituting it.
3. Add a normal-confidence transform:
   - `none`: multiplier one;
   - `linear`: linearly map alignment angle from perpendicular to aligned,
     `1 - 2*acos(clamp(abs_dot,0,1))/pi`;
   - `cosine`: use the absolute dot product directly.
   With `none`, missing alignment retains legacy finite magnitude. With either
   weighted mode, missing, invalid, or component-incompatible alignment maps to
   zero confidence.
4. Apply decision and normal confidence identically to the selected finite
   magnitude coefficient and finite sign-infringement coefficient in solver
   diagnostics and reference-fiber inference. Hard continuity remains
   unaffected. Connectivity and per-constraint Defect incidence stay discrete:
   positive effective magnitude, positive finite-sign weight, or a hard sign
   admits one factor incidence; exact zero does not. Confidence never
   fractionally scales the Defect unary itself.
5. Preserve existing enabled-sign behavior by default. Add an optional finite
   nonnegative sign-infringement cost. When absent, enabled signs remain hard
   incompatibilities regardless of confidence, including at a decision tie or
   missing alignment, so legacy hard-sign semantics remain exact. When present,
   each enabled nonzero signed measurement adds
   `I[target*predicted_delta <= 0] * sign_cost * decision_confidence * normal_confidence`.
   Class weights and distance decay do not multiply this term. It is added per
   measurement to winding energy and divided by winding temperature through
   the ordinary BP log potential; decoded and reference energy use the same
   untempered term. Repeated measurements add independently, and a Defect state
   neutralizes the complete pair factor. A zero configured cost or zero
   effective confidence contributes no finite sign factor, graph connectivity,
   or Defect incidence.
6. Keep magnitude class weights, distance decay, target quantization, dominant
   hypothesis selection, H/V orientation costs, and extraction geometry
   unchanged.

## CLI and diagnostics

1. Add explicit CLI modes for decision-confidence and normal-confidence
   transforms and a finite sign-infringement cost.
2. Reject invalid modes and nonfinite/negative costs.
3. Extend factor/reference diagnostics with raw alignment confidence,
   transformed decision/normal multipliers, and effective finite sign weight,
   so benchmark changes can be attributed to the intended control.
4. Keep the extracted reference-constraint denominator fixed in reporting and
   report admitted versus zero-confidence factors per class. Continue to rank
   with the existing comparator only: convergence, exact references, missing
   references, wrong references, right constraints, evaluated constraints,
   wrong constraints, and residual. Active/Defect population is reported but
   is not an added ordering key.
5. Preserve current CLI defaults and output behavior when no new option is
   passed.

## Experiment sequence

1. Re-run the unchanged fixed baseline on the 1024 crop and eight tagged
   reference fibers.
2. Run the complete hard-perpendicular confidence matrix: decision mode in
   `{legacy,linear,cosine}` crossed with normal mode in
   `{none,linear,cosine}` (nine rows total).
3. At the best authoritative confidence row, run sign modes
   `{perpendicular,parallel,both}` with finite sign costs
   `{0,0.25,1,4,16,64}`. Also run the corresponding three hard-sign controls.
   Log failed and nonconverged rows; do not skip a mode based on an earlier row.
4. Continue coordinate refinement from the best authoritative row across the
   new transform categories, sign penalty, five class weights, Defect cost,
   and BP temperature. Categorical neighbors are every other decision mode,
   normal mode, hard/finite sign treatment, and enabled sign mode. Positive
   numeric parameters use `/2,*2`; sign cost is bounded to `[0.125,512]`, class
   weights to `[0.125,64]` with explicit zero neighbors retained, Defect cost to
   `[0.125,400]`, and temperature to `[0.25,20]`. Use the established comparator
   defined above.
5. Re-run the selected row for determinism and report every attempted row in
   `planning/task_log.md`; do not silently select by percentage while dropping
   reference support.

## Tests

- Unit-test transform endpoints, midpoint behavior, validation, and default
  legacy equivalence.
- Unit-test perpendicular and parallel alignment confidence extraction,
  endpoint reversal, even median, and serial/parallel/batched equivalence.
- Unit-test finite sign reversal cost versus hard rejection, correct-sign zero
  penalty, exact-zero predicted-delta penalty, sign-cost-zero removal, zero
  confidence, Defect escape, and connectivity/incidence.
- Unit-test missing-alignment policy, dominant-only application, and unchanged
  H/V orientation/prepass costs.
- Unit-test reference inference and diagnostics against identical solver
  multipliers and finite-sign semantics.
- Assert complete default topology, incidence, diagnostics, reference result,
  and decoded output equivalence, not only transform-function equivalence.
- Build Release `vc_fiber_trace_chunk`, run focused winding/crop tests, and run
  `git diff --check`.
- Benchmark with the fixed approved runner and record exact inputs/settings.

## Spec update

- Document post-decision score transforms, normal-alignment confidence fields
  and transforms, and hard versus finite enabled-sign semantics.
- State that all new controls are neutral by default and the legacy coefficient
  and hard perpendicular sign remain unchanged.

## Docs update

- Document the three new CLI controls, formulas, interaction with dominant
  hypothesis/class weights, and benchmark interpretation.

## Changelog

- Record confidence-weighted winding evidence, finite sign penalties, and the
  selected measured parameter result.

## Follow-up default promotion

1. Promote the selected CLI/shared defaults: both sign classes, finite sign
   cost `44`, winding Defect cost `100`, and orientation BP temperature `1.25`.
   Keep decision `legacy`, normal confidence `none`, class weights
   `8,1,2,2,1`, and piece-break cost `0`.
2. Preserve access to strict hard signs by accepting the literal `hard` for
   `--winding-sign-cost`.
3. Update help, specs, docs, changelog, and production-default tests.
4. Rebuild Release targets, run focused tests, and run the approved benchmark
   without the newly defaulted flags. Compare it to the explicit selected row.
5. Commit the complete current task state after validation.
