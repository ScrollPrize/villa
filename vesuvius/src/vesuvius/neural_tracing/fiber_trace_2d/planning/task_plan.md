# Plan: restore baseline fiberlet search around weighted lookahead

## Objective decomposition

1. Define every decision score as two explicit terms:
   - the prior baseline score from the segment seed through checkpoint `C`,
     computed with the original authoritative whole-edge/transition costs;
   - the new subsegment score over `[C,H]`, where `H` is the common lookahead
     horizon, using checkpoint-local delay and geometric weighting.
   Scalar ownership is independent of retained whole-fiberlet geometry. For an
   edge spanning route-distance interval `[A,B]`, the prefix receives the old
   proportional whole-edge contribution over `[A,min(B,C))`; the forward term
   receives profile cost over `[max(A,C),min(B,H)]`. Its entering join belongs
   to the prefix when `A<C` and to the forward term when `A>=C`, so a join at
   `C` is forward with weight one. These half-open rules must prevent every
   checkpoint-straddling edge and join from being lost or counted twice.

2. Keep persistent route histories, cumulative five-component costs, checkpoint
   advancement, retained lookahead suffixes, cycle checks, canonical tie
   ordering, completion selection, and final beam-width pruning unchanged from
   the prior baseline. Only the scalar ranking contribution over `[C,H]`
   changes.

3. Consume decoded segment-density profiles directly. Validate that every
   segment length is finite and positive and every density is finite and
   nonnegative, but do not normalize, rescale, or replace the decoded values to
   force agreement with the separately stored whole-edge scalar. A complete
   weight-one integral should agree naturally up to codec and floating-point
   error; measure and report that discrepancy. Edge and join totals remain
   separate.

## Efficient integration

4. Replace repeated `arcCostBetween()` rescans with one shared linear
   integrator that advances through the overlapping segment and integration-grid
   boundaries together in `O(segment_count + grid_cell_count)`. It must:
   - operate in checkpoint-relative distance;
   - split exactly at checkpoint, delay, horizon, segment, and grid boundaries;
   - handle forward and reversed profiles identically;
   - integrate only the requested partial edge interval;
   - use the same integration-grid path for every weight, including `W=1`, so
     `--cost-step` retains its normal interpolation and accumulation role.

5. Carry decision-local scalar search state incrementally. Each queued label
   stores separate prefix-edge, prefix-join, forward-edge, and forward-join
   scalar totals plus its physical endpoint distance. Extending one fiberlet
   computes only that join and fiberlet's new contribution. Normalized profile
   storage remains cache-owned and leased only while integrating an edge; it is
   never copied into histories or heuristic memo entries. No successor or
   bound query may walk parent history.

6. Initialize retained beams once per decision by scoring their existing suffix
   from `C` to the retained endpoint. Shared immutable history remains the
   geometry/identity source, but initialization work is bounded by beam width
   and lookahead length. Advancing `C` intentionally recalculates the
   checkpoint-relative weighted suffix once because the decay origin changes.
   Advance the prefix scalar to the new checkpoint with the original
   authoritative proportional-edge rule, not with the preceding decision's
   weighted score. Count this bounded initialization history work separately.

7. For a successor crossing a front or horizon, compute its exact partial-edge
   contribution while the successor is created. Retain the whole physical
   fiberlet in graph state as before, but do not rescan it when ranking the
   completion.

## Effective pruning

8. Restore a relaxed cost-to-go bound for exact search under the new objective.
   Memoize relaxed continuation cost by incoming directed fiberlet and remaining
   distance bin. For each relaxed continuation:
   - ignore visited-node/cycle restrictions, which can only lower the bound;
   - use the same decoded segment profile and checkpoint-relative weights;
   - weight the transition at its relative anchor distance;
   - integrate only the required prefix of the terminal fiberlet;
   - for remaining distance `R`, use `floor(R/bin)*bin` and place that shorter
     problem at the later relative start `H-floor(R/bin)*bin`; this both omits
     nonnegative tail cost and uses no-larger geometric weights;
   - repeat that conservative floor/later-start construction after every edge,
     making remaining bins strictly decrease for every positive-length edge.
   Prove against exhaustive small graphs that the bound never exceeds the exact
   best completion cost.

9. Make relaxed-bound graph access deterministic. Every memo state either
   enumerates all outgoing arcs through existing `ChunkCache` leases or returns
   zero for the entire state; it may never use a partial outgoing set. Selection
   between exact relaxed evaluation and zero uses a canonical per-decision work
   budget, never LRU residency, worker timing, or cache readiness. Memo entries
   retain scalar results only, are bounded and cleared per decision, and release
   all chunk/profile leases after evaluation. Record bound-state count, cache
   reads/generation, and zero-fallback count.

10. Rank the exact queue by:
   `prefix_cost_at_C + incurred_weighted_forward_cost + relaxed_cost_to_go`.
   Completion ranking uses the corresponding exact score through `H`.
   Preserve the prior global cutoff, canonical tie handling, and global top-beam
   completion selection.

11. Apply the same incremental score representation to bounded search labels,
    while preserving its existing uniform-cost label/front algorithm. Label
    dominance and queue lower bounds use exact
    `prefix_at_C + incurred_forward` at the label's physical endpoint;
    nonnegative untraversed future remains zero as in the old bounded search.
    Intermediate and final fronts rank the exact common-distance score. Do not
    add the relaxed DP to bounded mode unless a separate proof and benchmark
    show it preserves label merging, fronts, ties, and worker merge order.

12. Use the conservative zero-future fallback only when the canonical
    per-decision relaxed-bound work budget is exhausted before a memo state
    begins. Never abandon a state after partially enumerating its outgoing
    arcs. Log and count the fallback so loss of pruning effectiveness is
    visible; it must not become the normal path on the benchmark corridor.

## Tests

13. Add focused regression tests for:
    - prefix-cost preservation when alternatives have identical futures;
    - no join double counting at, before, and after the checkpoint;
    - decoded complete-edge additivity within measured codec/FP error, without
      corrective normalization;
    - intentional partial-edge distribution differences;
    - quantified `W=1` sensitivity across multiple `--cost-step` values, with
      no material route-quality or failure-count regression;
    - checkpoint advancement resetting delay/weight origin without resetting
      accumulated prefix cost;
    - linear integration matching a brute-force high-resolution oracle;
    - incremental scores matching full route rescoring;
    - exact relaxed bounds never exceeding exhaustive completion costs across
      geometric weights, delays, nonintegral lengths, joins, reverse arcs,
      segment boundaries, and distance-bin boundaries;
    - exact and sufficiently wide bounded search selecting the oracle winner;
    - eager/cached and 1/32-thread determinism.

14. Add counters/timings for queue expansions, history nodes walked during
    retained-beam initialization, profile segments and integration cells
    visited, route-payload reads, relaxed-DP states/work, bound
    hits/misses/fallbacks, and scorer time. These are diagnostic only and must
    not affect cache identity or deterministic ordering.

## Validation and performance

15. Build the immutable pre-change baseline commit
    `64e534183d6ee4a9c0c09aa08f046463464ab7fb` in a separate worktree/build
    directory. Run it and the repaired implementation
    on the same Paris4 reference interval with independently hot but equivalent
    caches. Do not call a new-profile `W=1` run the baseline.

16. Validate first on 5,000 base voxels at radius 64, beam 16, checkpoint 48,
    and lookahead 384. Compare:
    - pre-change baseline;
    - repaired `W=1` at several cost-step values;
    - repaired `W=0.99`, delay 0;
    - repaired `W=0.99`, delay 192.
    Add route-by-route decision-score fixtures against the pinned implementation
    with checkpoint and horizon both at anchors and inside edges. Full prefix
    and complete post-checkpoint edges/joins must match; only profile-scored
    partial post-checkpoint intervals may differ. Report failures,
    mean/median/max normal and tangential reference errors,
    route-to-route distance from baseline, wall/CPU time, peak RSS, expansions,
    integration work, and relaxed-bound effectiveness.

17. Run at least five identical hot-cache repetitions for the pinned and
    candidate binaries with the same Release flags, threads, input, and
    equivalent fully populated caches. Report mean/p50/p95 wall and CPU time and
    peak RSS. Require candidate `W=1` and `W=0.99` p50 wall and CPU time to be at
    most 1.25x the historical baseline on 5k, unless profiling establishes and
    documents a necessary revised threshold. Require the weighted exact search
    to stay well below the state cap and report its bound hit/fallback/prune
    rates. Profile both binaries and report their leading tracer/scorer/search
    functions by CPU time.

18. Then run the representative longer corridor that exposed the regression.
    Acceptance:
    - `W=1` has baseline-like failure count and route geometry, with documented
      differences confined to partial post-checkpoint fiberlet integration;
    - changing cost step under `W=1` produces only the small route/error
      sensitivity expected from interpolation and accumulation rounding;
    - normalized route/failure/decision artifacts and expansion/cutoff counts
      are identical between 1 and 32 threads (excluding explicit thread/timing
      metadata);
    - hot-cache tracing meets the 1.25x target, with profiler evidence that
      integration is not a leading cost;
    - weighted modes retain effective pruning and do not approach exhaustive
      combinatorial expansion.

## Spec Update

- Replace the current checkpoint-only score definition with the explicit
  baseline-prefix plus weighted-forward decomposition.
- Require direct use and validation of decoded additive profiles without
  corrective normalization to authoritative full-edge cost.
- Require incremental scoring, linear profile integration, and effective
  admissible relaxed pruning.
- Require one general integration-grid algorithm for all weights and document
  that `W=1` can retain small `cost-step` interpolation/rounding sensitivity.
- Preserve cache/payload identity rules for replay-only weighting parameters.

## Docs Updates

- Update `volume-cartographer/docs/fiberlets.md` with score decomposition,
  boundary ownership, direct decoded-profile integration, pruning, diagnostics, and verified
  baseline/weighted performance.
- Correct the existing 5k table so the new-profile control is not presented as
  the historical baseline.

## Changelog Update

- Record restoration of prefix-aware ranking and efficient admissible pruning,
  plus before/after quality and performance results.

## Long-route complexity correction

19. Initialize every retained beam from cumulative edge/transition scalars at
    the history node immediately before the checkpoint. Walk backward only far
    enough to collect the checkpoint-to-horizon suffix, then integrate that
    bounded suffix in forward order. Do not materialize or reserve storage for
    the seed-to-checkpoint history.

20. Replace whole-registry expired-node sweeps with deterministic bounded
    incremental cleanup. Persist the stable ordered-map cleanup cursor, keep the
    work budget independent of committed-prefix length, and make expired-key
    replacement preserve cursor validity.

21. Add regression coverage that compares short- and long-prefix score
    initialization work and exercises repeated bounded registry cleanup. Repeat
    the hot-cache replay and verify route/failure determinism.

22. Treat `fiberlet-replay --threads N` as one evaluator budget shared by the concurrently
    running greedy and fiberlet evaluators. Split the budget deterministically
    between them; do not give each nested evaluator all `N` workers.
