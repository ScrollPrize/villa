# Implementation plan

1. Preserve legacy pairwise objective for old configs; new explicit local_contrastive mode uses32 negatives per anchor, negative budget65536 at128patch, positive budget1/32. Temperature0.1 and margin0.2 are initial configurable choices. Stable logsumexp([0,(negative-positive+margin)/temperature]); fixed anchor-count denominator, no cross-patch competition. Empty eligible presence gives zero numerator with fixed budget.
2. Add dedicated contrastive sufficient-statistic component; share target counts with prepass and one-backward accumulation. Preserve legacy numerics. Sampling deterministic per patch, no gradients through presence. Native normal/ellipse/queue unchanged.
3. Collect detached per-patch positive/negative cosine means/percentiles, negative>0.9 fraction, mean hardest-negative and gap during testing only, reusing pair computations. Log clearly labeled patch-averaged diagnostics, weighted by patch count across microbatches/test rows; no averaging quantiles under a claim of global quantiles. Raw+EMA. No extra model forward/backward.
4. Regression tests: budget, logsumexp math/hard-negative weighting, detached presence, zero candidates, partition invariance, one backward and metric aggregation. Managed GPU tests via approved caller.
5. New explicit combined contrastive config; stop current run using approved tree-aware command; launch fresh atLR0.01 with manager and verify test metrics/finite updates. Log outcomes and limitations.

## Spec update
Opt-in local competition alongside legacy objective; fixed target-only denominator and test-only diagnostics. No manager protocol or native reader changes.

## Docs updates
Embedding objective/options, metric definitions, fresh config and task status/log.

## Validation
Independent agent plan review; CPU+manager GPU oracle; real managed startup.

Review refinements: scale the negative budget linearly and round to a multiple
of K. Undefined negative diagnostics contribute no values/counts, not cosine0;
report per-metric patch coverage. Verify old root dead and new command has no
resume. Measure isolated GPU loss time/memory, not an inferred full-pipeline speed.

## Changelog
Record local competition and fresh production launch.
