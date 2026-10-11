# Separate V equality constraints from weak global Z prior

User: V at connected constraint points should be equal. Approximate V≈CT-Z
belongs only to a separate weak individual-fiber layout prior; local Z reversal
must be allowed. No solve rerun or moment acceleration requested.

## Spec update
Pair metric becomes metric_weight*min(((vB-vA)/metric_scale)^2,100), preserving
current weight .01, scale128 and cap while removing manufactured target_v.
Remove target_v from producers and consumers/APIs. Old cached arrays may contain
unused target_v; ignore them without rebuilding immutable geometry evidence.
Keep existing uniform per-fiber/piece prior sampled at start/middle/end, weighted
by hypothesis probabilities, but factor V≈Z into shared helper with independent
v_prior_weight (.01 default) and v_prior_scale (1000 default). No monotonicity or
slope term. Individual angular priors unchanged. Reuse helper for categorical
diagnostics too; do not duplicate loss math. Document prior is three point samples
per piece, not supervision on pairwise endpoint Z differences.

## Validation/docs
Tests equality costs zero regardless of CT-Z, unequal V costs positive, target
absence in tensor readers, weak prior scale/gradient with reversed-Z samples,
and unchanged other objectives. Update all shared function callers and benchmark
argument lists. Run UV tests, independent review, archive previous task log and
update current status/docs. Historical outputs predate changes; no fresh solve.
