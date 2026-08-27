# Plan: Explicit Mixed-state fiber belief propagation

## Contract

- Add an experimental `sum-product-mixed` inference mode without changing the
  existing binary min-sum or binary sum-product paths.
- Reuse the validated merged perpendicular factor graph, factor costs, and
  central straight hard-H seed.
- Give each variable three categorical states: V, Mixed, and H. When both
  endpoints are oriented, use the existing same/different factor energy. An
  endpoint in Mixed disables the orientation term and pays a fixed
  `mixed_cost_per_link` for every raw measurement represented by that merged
  factor; two Mixed endpoints therefore pay twice per measurement. This
  matches the established per-link Broken penalty while making the state
  probabilistic.
- Run deterministic synchronous log-space sum-product over normalized
  three-entry messages. Apply the existing damping, post-damping residual,
  iteration limit, and temperature semantics. The seed is exactly H.
- Report normalized `P(V)`, `P(Mixed)`, and `P(H)`. Retain a scalar orientation
  value `P(H) + 0.5*P(Mixed)` only for the established H/V band visualization
  and explicitly labeled heuristic consistency diagnostics; do not describe it
  as `P(H)` or a calibrated binary marginal.
- Add `--bp-mixed-cost F`, valid only for `sum-product-mixed`, and keep a
  conservative default matching the established broken cost per link. Reject
  balance modes and other min-sum-only controls as for binary sum-product.
- Own separate `_bp_sum_product_mixed_p0..p9.obj` orientation-projection bands,
  `_bp_sum_product_mixed_mixed_p0..p9.obj` `P(Mixed)` bands, and a consistency
  CSV with all three probabilities and the projection. Print an argmax
  V/Mixed/H confusion table with exact ties separate and a tie-aware
  `P(Mixed)` AUROC against the existing direction diagnostic. H/V confusion is
  descriptive and reference-oriented; Mixed classification and AUROC are
  gauge-invariant.
- Run a small mixed-cost sweep on the centered-384 full-Mixed cohort at the
  best previously observed temperature `T=2.5`, while retaining the general
  CLI defaults unless the single-crop evidence is sufficient to justify a
  documented experimental default change.

## Implementation

1. Extend inference/config/report types with the ternary mode, Mixed cost, and
   explicit state probabilities.
2. Add a shared stable log-sum-exp helper for three-state categorical message
   updates, leaving binary solver arithmetic unchanged.
3. Extend BP-only parsing, validation, reporting, CSV, and OBJ ownership.
4. Add exact-tree, symmetry, seed, penalty, damping, nonconvergence, and
   determinism regression tests.
5. Run the centered-384 experiment and log the measured confusion/AUROC.

## Spec Update

Specify ternary factor energies, message normalization, seed/symmetry
semantics, scalar orientation projection, CLI validation, and artifact fields.

## Docs Updates

Document the experimental `sum-product-mixed` invocation, Mixed per-link cost,
three-state marginal interpretation, and outputs in
`volume-cartographer/docs/fiber_chunk_tracing.md`.

## Testing

- Compare ternary BP marginals with brute-force exact marginals on seeded trees.
- Verify a cheap Mixed state attracts mass and an expensive one approaches the
  binary result; test duplicate-measurement scaling, two-Mixed endpoint energy,
  unseeded-component H/V symmetry, isolated uniformity, damping, low
  temperature, message limits, input ordering, and invalid options.
- Preserve all existing binary BP tests and outputs.
- Build `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with `-j32`, run
  the focused suite, and run `git diff --check`.
- Run centered-384 BP-only mixed-cost experiments using the existing cached
  trace dataset and normal manifest.

## Changelog

Record the explicit three-state sum-product BP experiment and its normalized
V/Mixed/H diagnostics.
