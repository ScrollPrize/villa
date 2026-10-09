# Fixed sign-violation charge plus magnitude

User requires any wrong selected retained sign to trigger correction or at least
one endpoint broken probability >0.5. Keep magnitude penalty for larger jumps.
Existing anchor experiment and its uncommitted work/results are preserved.

## Plan

1. Keep all-pair likelihood and magnitude loss. Add fixed wrong-sign charge,
   calibrated against per-node degree-mixed rejection cost under actual E/N
   normalization. Use max(expected violating mass, retained mass of wrong selected
   pair) so a wrong selected assignment cannot hide behind diffuse hypotheses.
   Default experiment multiplier4: when both q<=.5 the selected charge alone
   favors increasing an endpoint q over retaining a wrong pair. This is a local
   gradient/cost comparison, not a claim of global convergence.
2. Explicit optional readout broken iff q>.5, choose argmax winding otherwise.
   Check strict negative signs (zero tolerance) separately from historical1e-4.
   Preserve old configs/readout. Recompute warm-start baselines under new rule.
3. Warm-start previous saved states with hash/identity checks; no re-extraction
   or changed labels/geometry. Rejection is trainable immediately. Use pilot
   multiplier sweep before both full H conventions. Keep no post-hoc rejection.
4. Optional active violation sampling mixed with uniform events; exact inverse
   proposal weights preserve the objective, including event priors. Recheck
   full graph regularly, since assignments change. Evaluate selected geometry
   freshly in float64 to match strict readout; never force stale audited edges
   wrong. Do not freeze coordinates.
5. Regression: tiny-negative vs zero, magnitude, unknown signs, normalization
   and broken-gradient dominance, diffuse distributions, q=.5 boundary,
   warm-start identity/provenance, unbiased proposal weights, legacy tests.
6. Run approved managed runner; rank zero violations first, then retention.
   Record every remaining violation and whether nonempty retained known-sign
   graph exists. Do not call nonzero or empty-retention results successful.

## Spec update

Opt-in charged sign objective and explicit >50% broken readout. No relaxed sign
test, no silent post-hoc deletion, no guarantee inferred from finite weights.
Original stage exclusions and event arrays remain immutable.

## Docs updates

New configs, reproducible results/report commands, task/status/log/changelog and
uv_layout.md. Independent plan/code/result review. Historical results unchanged.
