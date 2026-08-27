# Plan: Binary sum-product fiber belief propagation

## Contract

- Reuse the exact validated and merged binary factor graph used by min-sum BP.
  Fiber states remain V/H; sum-product changes inference and interpretation,
  not the graph objective or diagnostic cohort.
- For factor costs `E_same` and `E_different` and positive temperature `T`, use
  log potentials `-E_same/T` and `-E_different/T`. Store one normalized H-vs-V
  log-message per directed factor edge. Perform deterministic synchronous
  updates in log space with the existing damping, message-iteration limit, and
  residual tolerance. Use stable two-term log-sum-exp operations.
- Define each directed message as
  `ell_i->j = log m_i->j(H_j) - log m_i->j(V_j)`. For source cavity log odds
  `r`, update it as
  `logsumexp(-E_diff/T, r-E_same/T) -`
  `logsumexp(-E_same/T, r-E_diff/T)`. A hard-H seed emits
  `(E_diff-E_same)/T`. Subtract the common minimum factor cost before scaling;
  this changes neither ratio nor marginal and keeps the exponentials bounded.
- Initialize all directed log ratios to zero and use synchronous updates.
  Apply damping as `old + damping * (raw-old)` to every message, including
  seed messages, and measure the residual after damping. If the iteration
  limit is reached, expose the final finite iterate with status
  `message_limit`; do not describe it as a converged marginal.
- Clamp only the established central straight seed exactly to H. Other nodes
  have no unary evidence in this first experiment. Unsupported components
  therefore retain their global H/V symmetry and report `P(H)=0.5`.
- Decode each nonseed fiber directly as the normalized sum-product marginal
  `P(H)=sigmoid(log_odds)`. This is an approximate marginal on loopy graphs,
  unlike the existing min-sum horizontalness, and must be named accordingly.
  It is exact on trees and a loopy-BP/Bethe approximation after convergence on
  graphs with cycles. The existing soft same-label consistency diagnostic
  remains an endpoint-independence proxy, not a pairwise BP marginal.
- Add `--bp-inference min-sum|sum-product` for BP-only direction-ablation.
  Omission preserves min-sum. Sum-product is initially incompatible with the
  drafted `--bp-balance` modes; reject that combination explicitly rather than
  approximating a global prior.
- Treat temperature differently and explicitly: min-sum applies it only when
  converting a min-marginal advantage to horizontalness, whereas sum-product
  divides every pair-factor energy by it during inference. Shared message
  iteration, damping, residual, and temperature controls remain available;
  reject min-sum-only target, strength, balance-iteration, and balance-
  tolerance controls for sum-product.
- Keep artifact families separate: min-sum retains `<base>_bp_none_*`, while
  sum-product owns `<base>_bp_sum_product_p0..p9.obj` and
  `<base>_bp_sum_product_consistency.csv`. Include inference name and
  temperature in console provenance and CSV rows.
- Run the centered-384 full-Mixed cohort at the existing `T=0.25`, then a small
  deterministic temperature sweep if runtime remains negligible. Compare
  convergence, value distributions, mismatch diagnostics, and tie-aware AUROC
  against the committed min-sum baseline.

## Implementation

1. Refactor only the shared report initialization needed by both algorithms;
   leave the existing min-sum update path numerically unchanged.
2. Add stable scalar log-space sum-product message updates and a public solver
   entry point using the existing graph builder, seed helper, and validation.
3. Add CLI selection, provenance, distinct artifacts, and BP-only dispatch.
4. Run the full-Mixed comparison and record exact results.

## Spec Update

Add sum-product factor potentials, log-message normalization, seed and
symmetry behavior, approximate-marginal semantics, CLI compatibility, and
artifact ownership to `specs.md`.

## Docs Updates

Document `--bp-inference sum-product`, temperature meaning, approximate
marginal interpretation, and output names in
`volume-cartographer/docs/fiber_chunk_tracing.md`.

## Testing

- Compare sum-product marginals against brute-force exact marginals on a small
  seeded tree at multiple factor strengths and temperatures, with damping one
  and below one.
- Test the exact seed, symmetric disconnected component at `0.5`, duplicate
  factor merging, reversed input order, finite low-temperature behavior, and
  explicit rejection of nonpositive temperature and unsupported balance use.
- Add a two-node sign oracle, a forced `message_limit` final-iterate case, a
  seed-only graph, and deterministic repeated loopy-graph inference.
- Preserve all existing min-sum test values unchanged.
- Build `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with `-j32`, run
  the focused suite, and run `git diff --check`.
- Run the centered-384 BP-only comparison without invoking HiGHS.

## Changelog

Record the experimental log-space sum-product BP mode and normalized marginal
diagnostics.
