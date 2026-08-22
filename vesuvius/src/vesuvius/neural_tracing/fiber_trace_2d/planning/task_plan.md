# Plan: diagnose persistent wide-radius fiberlet replay failures

## Experiment

1. Preserve the completed matched-terminal lookahead sweep, including its exact
   settings, failure arcs, wall/CPU time, and peak memory, in the fiberlet docs.
2. Run a full-reference radius-64 replay with the best radius-768 objective:
   beam 16, checkpoint 48, lookahead 384, exact search, profile blend 0.75,
   delay 192, terminal weight 0.25, integration step 16, and 32 threads.
3. Run or recover a matched radius-768 replay with decision diagnostics enabled.
   Reuse explicit hot caches and do not regenerate compatible anchor/fiberlet
   chunks.
4. Treat radius 64 as a constrained candidate, not an automatic ground-truth
   oracle. Verify that it has no reset and bounded Euclidean/normal/tangential
   error through documented pre/post windows around the exact wide-radius
   failure arcs. Call it a reference-following small-radius route only if those
   checks pass.
5. Compare stable directed identities formed from endpoint anchor
   coordinate/variant IDs and orientation; do not compare array indices. First
   classify topology in both graphs: endpoint anchors present, edge present,
   join admissible, cycle eligible, retained completion, rank/cutoff, or an
   explicit missing reason. Radius-dependent NMS and extraction can make graph
   populations non-monotonic.
6. Establish a common decision context before making a causal cost comparison:
   the same absolute reset/search arc, forced seed, checkpoint phase, incoming
   history/common prefix, objective, and cost profile. Prefer focused bounded
   diagnostics after ordinary full traces identify failures, and verify each
   focused replay reproduces the relevant full-run signature. If the current
   diagnostics cannot establish common history, report the comparison as
   descriptive rather than causal.
7. For each persistent failure, locate the first comparable divergence and
   compare selected and reference-following edge, join, weighted profile,
   length, normalized total, rank, and cutoff. Check whether the constrained
   continuation survived in the wide-radius retained frontier. If needed, use
   a short radius ladder to identify where a competing topology first appears.
8. Describe a reusable offline tuning dataset: positive continuations from the
   correct replay plus competing retained or deliberately forced alternatives.
   Include stable route identity, raw decoded per-segment density profiles and
   offsets, edge/join component costs, geometry, checkpoint/prefix state,
   active cache/profile fingerprints, and selection/cutoff diagnostics. State
   that rescoring only reranks a fixed local collection: any parameter that can
   change the exact-search frontier or future checkpoint history still requires
   replay, while radius/search geometry requires recollection. Keep these two
   known windows as diagnostic/tuning data and require held-out fibers/windows
   for generalization claims.

## Tests

- Use the complete Paris4 reference fiber and the canonical fiber/normal
  manifests already used by the radius-768 experiments.
- Measure wall time, CPU time, peak RSS, failure count, and failure arcs for both
  matched runs.
- Validate diagnostic JSON structure and deterministic route alignment with a
  focused analysis script or equivalent structured parser.
- Record cache fingerprints. Radius 64 has a separate cache identity and may
  require a cold generation pass; only repeated runs against completed roots
  are hot-cache comparisons. Confirm no compatible chunks are rewritten.
- Account separately for candidates absent because of topology and candidates
  present but lost through objective ranking/cutoff. Do not treat a state-cap
  failure as a traced-fiber failure.

## Spec Update

- No normative tracer behavior change is planned. If a reusable diagnostic
  collection is added, specify its route identity, costs, labels, and limits.

## Docs Updates

- Retain the larger-lookahead sweep table.
- Add the small-radius correctness-reference result and the failure comparison,
  including what can and cannot be tuned offline from collected candidates.

## Changelog Update

- Record the failure-diagnosis artifact or analysis workflow only if new durable
  diagnostics are added; parameter-only findings remain in the experiment docs.

## Deferred follow-up

The constrained-radius route is not a correctness oracle. A future experiment
should search the radius-768 graph itself for the minimum-cost route that stays
inside the reference error region. The admissibility region is the current
Lasagna-normal ellipsoid,
`sqrt((normal/20)^2 + (tangential/80)^2) <= 1`, unless the evaluation contract
is deliberately changed. It should preserve the same seed, graph, checkpoint
history, and objective as the unconstrained search.

A separate future evaluation may allow bounded excursions outside that region
when the route rejoins it within a configured base-arc distance. That rule must
record excursion length, peak normalized error, integrated excess error, and
whether the route rejoins before counting it as tolerated. It is an evaluation
policy for potentially imperfect ground truth, not a change to replay cost.
