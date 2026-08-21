# Plan: exact cost-bounded fiberlet lookahead

## Contract

1. Preserve the rolling checkpoint contract: committed histories end at
   anchors, `H=192` base voxels, `D=48` base voxels by default, and commitment
   retains the complete fiberlet containing `C+D`.
2. Score every lookahead candidate at the exact logical horizon
   `T=min(C+H, selected route length)`. Complete preceding edges contribute
   full path cost. The edge crossing `T` contributes
   `(T-edge_begin)/edge_length` of each active path-cost component; its entering
   join contributes fully. Geometry beyond `T` contributes no score.
   If `T` lands exactly on an anchor, no outgoing edge or outgoing join is
   charged. A join is charged only when a positive fraction of its outgoing
   edge lies inside the horizon.
3. Because every completion has scored length `T`, rank by scored total cost,
   then deterministic logical route identity. Report normalized loss as
   `total/T`; it is diagnostic and cannot change ordering.
4. Search the route tree in nondecreasing accumulated full-edge cost. For an
   incomplete route, accumulated cost is an admissible lower bound because all
   edge and join loss components are finite and nonnegative.
   Validate that invariant whenever an edge or join is observed, including
   values reconstructed from float or quantized caches.
5. Maintain the best completion for each distinct committed prefix through
   `C+D`. Before 16 distinct prefixes complete, no global cutoff exists. After
   that, stop when the cheapest queued lower bound is strictly greater than the
   16th completion. Equal-cost states remain eligible for deterministic
   tie-breaking.
   Once a completion exists for a committed prefix, an incomplete state with
   that same fixed prefix may be pruned only when its lower bound is strictly
   worse than the prefix's best completion.
6. Do not merge states solely by anchor: incoming edge, accumulated length,
   visited anchors, and committed-prefix identity affect valid continuations.
   Preserve exact cycle rejection and all existing graph validity rules.
7. Keep the generated-state limit as an explicit safety failure. It must not
   silently convert the exact search into a beam search.
8. Keep final reference/failure materialization separate from scoring. Search
   and decision diagnostics retain the complete terminal fiberlet and report
   its full route length separately from the exact scored length `T`; only
   final reference/failure output retains existing exact reference clipping.

## Implementation

1. Extract deterministic one-edge successor generation from the exhaustive DFS
   so the exact queue and tests share join, direction, cycle, cost, length, and
   state-limit handling.
2. Add a helper that scores any complete whole-fiberlet history cumulatively
   from its segment seed through an exact path-length horizon, including
   proportional terminal-edge path cost and full entering-join cost.
3. Enumerate the distinct whole-fiberlet prefixes that cross the next shared
   checkpoint. This expansion covers only the checkpoint advance, retains exact
   cycle state, and performs no unequal-distance quality pruning.
4. For each fixed committed prefix, independently solve its cheapest exact
   continuation to the common lookahead horizon with deterministic A*. Its
   admissible cost-to-go table is keyed by incoming directed fiberlet and a
   downward-rounded 0.5-prediction-voxel remaining-distance bin. The relaxed
   recurrence uses the real outgoing edges, joins, and proportional terminal
   edge cost but ignores visited-node restrictions; it therefore never
   overestimates a valid continuation. Initial states already covering the horizon and newly
   generated horizon-crossing successors are exact-scored immediately;
   terminal-edge overshoot cost never enters the queue bound. A prefix search
   stops only when its cheapest queued `accumulated + relaxed-future` bound is strictly greater
   than that prefix's best exact completion.
5. Run independent fixed-prefix searches in parallel under `--threads`, then
   merge their single best completions in canonical prefix order and retain the
   global best 16. This is the same exact top-16-prefix result as a monolithic
   search, while exposing the natural independent work and avoiding one serial
   mixed-prefix queue.
6. Update decision diagnostics with generated, expanded, completed, and
   cost-pruned state counts, exact scored length, and complete terminal-route
   length. Remove claims that overshoot cost participates in ranking.
7. Keep cache payloads and identities unchanged. Search-only settings and
   diagnostics remain replay metadata.

## Alternatives recorded, not implemented

- Bounded speculative search: prune globally to a wider lookahead beam, for
  example 128, at `C+48`, `C+96`, `C+144`, and `C+192`. This is fast but may
  discard a temporarily expensive route with a better later continuation.
- A continuous piecewise-linear relaxed cost-to-go function would be stronger
  than the downward-binned table but may develop many breakpoints. It remains a
  fallback only if the admissible 0.5-voxel bins are still too weak.

## Testing

1. Prove proportional terminal-edge scoring with complete terminal geometry,
   full entering-join cost, exact-anchor boundary behavior, and identical
   scored horizon across candidates with different overshoot.
2. Construct a graph where raw whole-edge cost would choose the wrong prefix
   and prove exact-horizon cost chooses the correct one.
3. Prove cost-bound pruning does not start before 16 distinct prefixes exist,
   preserves equal-cost tie candidates, applies the strict same-prefix bound,
   and returns the same winners as an independently implemented exhaustive
   oracle on a small graph.
4. Cover cycles, invalid joins, negative/nonfinite edge and join costs, graph
   exhaustion, state limit, checkpoint commitment, full terminal diagnostic
   geometry, final output clipping, winner changes, and reseeding.
5. Assert deterministic repeated output and identical eager/cached float graph
   results. The queue implementation is initially deterministic and serial;
   measure before considering parallel batched expansion.
6. Build `vc_fiberlets`, `test_fiberlet_paths`, `test_fiber_replay`, and
   `test_fiberlet_storage` with `-j32` and run relevant tests.
7. Run the same completed 600-base-voxel radius-768 hot-cache interval. Compare
   route/failure metrics against the recorded physical failure and report wall,
   CPU, generated, expanded, completed, and pruned counts.
8. If the focused result is correct and practical, run the full radius-768
   fiber and report failures, distance statistics, time, and peak memory.

## Spec update

- Correct lookahead scoring to the exact logical horizon with proportional
  terminal-edge path cost and full entering-join cost.
- Replace exhaustive all-route enumeration with exact uniform-cost search and
  its strictly-worse cutoff rule for 16 distinct committed prefixes.
- Define queue state, nonnegative-cost requirement, deterministic ties, safety
  bound, and diagnostics. Preserve whole-fiberlet commitment and output
  clipping contracts.

## Docs updates

- Update `volume-cartographer/docs/fiberlets.md` with exact-horizon scoring,
  uniform-cost termination, and the distinction between scoring geometry and
  committed route state.
- Document bounded 48-voxel speculative pruning and relaxed-heuristic A* as
  possible future alternatives, not active behavior.

## Changelog

- Record the correction from overshoot-normalized exhaustive enumeration to
  proportional exact-horizon uniform-cost lookahead, including focused and
  full-fiber measurements that complete during this task.

## Performance and correctness report

- Record exact commands, revision/build type, data, cache state, radius,
  `D/H`, queue counts, wall/CPU time, failures, and reference-distance metrics.
- Record that the prior exhaustive full run reached 94.4% in 5m10s before
  manual interruption and produced no atomic partial artifact.
