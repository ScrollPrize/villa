# Plan: Non-transitive Mixed-state fiber BP

## Contract

- Preserve the existing V/H pairwise energies for oriented endpoint pairs.
- Set every pairwise energy involving Mixed to zero, including Mixed/Mixed.
  This is conditional neutrality: a cavity clamped to Mixed sends a uniform
  factor message, while residual V/H probability may still transmit evidence.
- Charge `bp_mixed_cost` once per node assigned Mixed as a unary energy,
  independent of degree and merged measurement count. Keep the central seed
  exactly H with no Mixed alternative.
- For every non-seed node use `u(V)=u(H)=0`, `u(Mixed)=bp_mixed_cost`, so its
  log unary is `(0,-bp_mixed_cost/T,0)`. Include it exactly once in each
  outgoing cavity and exactly once in the final marginal, using the existing
  temperature, damping, normalization, convergence, and deterministic update
  rules. The seed replaces this unary with an exact delta at H.
- Keep the CLI spelling `--bp-mixed-cost`, but rename internal/report/CSV
  terminology from per-constraint cost to unary cost. This experimental format
  is not shipped, so retain no compatibility alias for the old field name or
  CSV column.
- Leave binary min-sum and binary sum-product inference unchanged.

## Implementation

1. Rename the mixed cost configuration and report fields to `mixedUnaryCost`.
2. Build ternary factor potentials with zero energy whenever either endpoint is
   Mixed, and add the Mixed unary log-potential to non-seed cavities and node
   marginals.
3. Update CLI help, validation messages, diagnostic tables, and CSV schema to
   describe the unary cost.
4. Update the exact enumerator and focused tests for the new energy model.
5. Keep ternary final marginal accumulation at exactly one incoming directed
   message per incident factor while adding the node unary.

## Spec Update

Replace the measurement-scaled Mixed factor energy with neutral Mixed factors
and one per-fiber unary energy. Specify message/marginal inclusion of the unary,
seed behavior, degree independence, and non-transitivity.

## Docs Updates

Update `volume-cartographer/docs/fiber_chunk_tracing.md` to explain that Mixed
disables incident orientation terms and pays one node-local cost.

## Testing

- Compare seeded-tree marginals to brute-force enumeration of the new model.
- Verify a factor touching a forced/strongly favored Mixed node sends no state
  preference to its neighbor when the source cavity is exactly Mixed.
- Verify multiple consistent oriented neighbors accumulate enough evidence to
  select the corresponding H/V state.
- Verify conflicting oriented evidence can select Mixed at the conflicted node.
- Verify duplicating incident measurements changes oriented evidence but does
  not duplicate the Mixed unary prior.
- Verify an unseeded isolate has probabilities proportional to
  `(1,exp(-cost/T),1)`, the seed remains exactly H, and unary cost is independent
  of node degree/measurement multiplicity.
- Verify gauge symmetry, deterministic ordering, damping, message limits,
  negative/nonfinite unary rejection, and wrong-mode CLI rejection.
- Build `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with `-j32`, run
  the focused test binary, and run `git diff --check`.
- Run the centered-384 full-Mixed cohort at `T=2.5` with a small unary-cost
  sweep, comparing Mixed AUROC and trusted H/V errors with recorded binary and
  prior ternary results.

## Changelog

Record the corrected node-local, non-transitive Mixed-state BP formulation.
