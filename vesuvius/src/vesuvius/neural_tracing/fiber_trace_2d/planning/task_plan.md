# Plan: bounded intermediate fiberlet lookahead

## Search contract

1. Retain the rolling checkpoint `C`, checkpoint step `D=48` base voxels, and
   lookahead `H=192` base voxels. Whole fiberlets remain the persistent graph
   state; only scoring clips the terminal fiberlet at a logical horizon.
2. Clamp all fronts to the selected route end. The first front is
   `min(C+D, C+H, route_end)`. Each later front is
   `min(previous+P, C+H, route_end)` until the final front is exactly the
   selected scoring horizon; a nondivisible
   remainder creates a shorter final interval, and `P` larger than the
   remainder goes directly to `C+H`.
3. Enumerate complete fiberlet histories through `C+D`. Their committed-prefix
   identity is `(segment seed, ordered stable logical arc IDs through C+D)`,
   never a persistent-history pointer. A route that already overshoots a later
   front remains eligible without expansion. Other routes expand through every
   valid branch until they reach that front.
4. At every front `F`, score candidates cumulatively from the segment seed
   through `F`, not only over the latest interval. If `F` is exactly an anchor,
   charge the edge ending there and its entering join but no outgoing edge or
   join. If `F` lies inside an edge, charge that edge proportionally and its
   entering join fully. Geometry and visited state retain the entire terminal
   edge.
5. Deduplicate identical complete logical routes, then sort by exact-front total
   loss, complete logical route key, and committed-prefix key.
   Determine the best exact-front representative per prefix and globally sort
   those representatives. Retain up to `K`; if fewer than `K` prefix
   representatives exist, fill remaining slots with the globally best
   unselected routes. This prevents one prefix from consuming the frontier
   while acknowledging that more than `K` prefixes is itself approximated.
6. At the final front, retain only the best continuation for each committed
   prefix, then select the globally best 16 prefixes. Commit each selected
   prefix exactly as before.
7. `--search-width 128` selects bounded intermediate pruning. Width must be at
   least the final beam width. `--search-width 0` retains exact A* as an oracle.
   `--prune-distance 48` is finite, positive, base-voxel search metadata.
8. Keep the one-million generated-state safety failure. Intermediate pruning
   is an explicit configured approximation, not an implicit state-limit
   fallback. One budget covers checkpoint-prefix enumeration and every front in
   the rolling decision. Crossing it aborts the complete decision regardless of
   worker scheduling; partial candidates never survive.

## Implementation

1. Add replay configuration and CLI fields for search width and prune distance,
   with defaults 128 and 48 base voxels. Validate them and serialize them in
   replay diagnostics without changing cache identities or extraction padding.
   Convert all three distances from base to prediction voxels through the same
   graph scale.
2. Represent each working route with its stable logical committed-prefix key.
   Expand independent routes to one logical front in parallel under
   `--threads`. Split the initial single route into canonical root-successor
   jobs. The first front retains up to its full target width because stable
   prefixes are established there. At later fronts, each job retains one
   representative for its stable prefix plus only enough additional candidates
   to occupy globally unfilled slots. A local top `K` is unnecessary once `K`
   stable prefixes are already represented.
   Merge job results in canonical input order and rerank globally, avoiding an
   unbounded completed-candidate list. Canonical input indices, canonically
   sorted local outputs, and a complete-decision abort at the state limit make
   results independent of scheduling.
3. Add exact-front ranking and diversity-preserving working-front selection.
   Reuse the existing cost, successor, persistent-history, and exact-horizon
   helpers; do not duplicate graph rules or scoring.
4. Replace exhaustive between-front enumeration with uniform-cost distance-
   label expansion. A label is keyed by logical incoming directed fiberlet and
   a 0.5-prediction-voxel front-offset bin; only the lowest accumulated-cost
   history per state survives. Crossing routes are ranked with exact partial-
   edge scoring. Each canonical input job
   stops only after it has `K` completions and its next lower bound is strictly
   worse than its local `K`th exact completion; equality remains eligible for
   deterministic ties. This yields enough local candidates for the global
   width and diversity selection without enumerating all combinations.
5. At the final front, reuse the existing best-per-prefix/final-width retention
   and decision diagnostics. Each front records its exact distance, input,
   generated, expanded, rejected, and completed counts, distinct prefix count,
   retained representative count, global-fill count, pruned count, and whether
   `K` bound. Each decision records mode, `K`, and `P`.
   It also records cumulative decision counts and the selected committed-prefix
   key.
6. Keep the exact full-horizon A* path available only when search width is
   zero. Record the completed plain-pruning measurements separately from the
   bounded-front label-search measurements.
7. Preserve final reference-end clipping and `terminal_partial_edge`. Every
   newly selected prefix rematerializes provisional matches from the segment
   seed, replacing prior provisional state rather than appending to it.

## Logged later options

- Adaptive horizon: deterministically retry or shorten `H` based on generated
  state counts, never elapsed wall time. Candidate horizons are 192, 144, 96,
  and 48 base voxels. Any retry must restart the decision from its committed
  population, and diagnostics must record the effective horizon and reason.
- Additional search work: profile the remaining dense-tail front cost and try
  an admissible remaining-cost heuristic or a coarser explicitly configured
  state only if uniform-cost labels remain too expensive. Do not use elapsed
  time or silently reduce the configured horizon.

## Testing

1. Add small deterministic graphs proving exact-front partial-edge scoring,
   exact-anchor ownership, whole terminal geometry, multi-front overshoot,
   prefix diversity, more-than-`K` prefixes, fewer-than-`K` groups plus global
   fill, equal-score ties, representative changes at later fronts, final
   one-per-prefix selection, and width validation.
2. With a sufficiently wide frontier, assert bounded search matches exact A*
   on small exhaustive fixtures. Add a fixture where a narrow frontier
   intentionally differs, proving the approximation is active and diagnosed.
3. Assert byte-identical decisions, counts, and output for 1, 2, and 32
   expansion threads under a nonbinding state limit, and preserve
   cycle, invalid-join, malformed-cost, state-limit, reseed, and output-clipping
   coverage.
   Include non-unit prediction-to-base scale and nondivisible `D/P/H` front
   scheduling.
4. Build `vc_fiberlets`, `test_fiberlet_paths`, `test_fiber_replay`, and
   `test_fiberlet_storage` with `-j32`; run the relevant binaries and report any
   pre-existing failures separately.
5. Run at least three hot-cache radius-768 comparisons for widths 64, 128, and
   256 against `--search-width 0` on the known benign 600-base-voxel window and
   the spatial window containing the former 95.8% full-route stall. Report
   min/median/max wall and CPU time, frontier counts, failures, final-prefix
   agreement, first decision divergence, selected-route cost delta, and
   min/mean/median/p95/maximum total, normal, and tangential line separation.
   Use deterministic `--arc`, `--length`, and `--seed-key` where available,
   record peak RSS, and verify hot runs write no cache chunks.
6. Run the full radius-768 fiber with the best initial width, report failures,
   reference distances, wall/CPU time, peak memory, and whether the former
   95.8% stall is eliminated.

## Spec update

- Replace the current specification bullets that forbid intermediate pruning
  and charge complete horizon-crossing edges. Define bounded intermediate
  fronts, cumulative exact-front scoring, stable prefix
  diversity, final one-per-prefix selection, deterministic parallel merge, and
  the explicit approximation boundary.
- Define zero search width as exact-oracle mode and state that width/prune
  distance are replay-only metadata.
- Record adaptive horizon and bounded A* only as deferred options.

## Docs update

- Update `volume-cartographer/docs/fiberlets.md` with CLI usage, intermediate
  frontier behavior, diagnostics, exact-oracle invocation, and cache invariants.

## Changelog

- Record bounded intermediate lookahead and focused/full radius-768 results.

## Performance report

- Record exact commands, revision/build type, dataset, hot-cache state, radius,
  `D/H/P/K`, threads, candidate counts, wall/CPU time, failures, distance
  statistics, and peak RSS.
- State explicitly that pruning at `C+D` can discard a prefix that would rank
  better at `C+H`; this is the intended approximation boundary.
