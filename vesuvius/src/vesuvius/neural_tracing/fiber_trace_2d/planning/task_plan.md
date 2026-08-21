# Plan: persistent equal-arc fiberlet beam search

## Contract

1. Maintain up to `beam_width` complete route histories from the original seed
   of the current uninterrupted replay segment. Ordinary beam iterations must
   never collapse the population to one route. A reference-triggered reset
   closes the segment, reseeds the whole population, and establishes a new
   absolute-distance origin.
2. Separate the beam-front step `D` from lookahead distance `H`; both are base-
   voxel distances. At iteration `k`, retained prefixes end at absolute route
   length `S = k*D` from the segment seed, candidate next prefixes end at
   `S + D`, and their ranking continuations all end at `S + D + H` (shortened
   only at the requested replay end).
3. Represent a checkpoint explicitly as either `InsideEdge(edge, offset)` or
   `AtAnchor(anchor, incoming_edge)`. An exact edge end remains `AtAnchor` and
   does not choose or charge an outgoing transition prematurely. Advancing by
   `D` may complete zero, one, or multiple fiberlets. Crossing an anchor
   branches through its valid outgoing transitions; moving inside a fiberlet
   is deterministic.
4. Rank or prune candidates only after they have reached the same absolute
   scored length from the segment seed. Fiberlet-count layers and uneven raw
   endpoints are never valid pruning frontiers. Clip both the starting and
   ending fiberlet portions when a scoring interval begins or ends inside an
   edge.
5. Cumulative ranking cost covers the complete route from the segment seed
   through `S + H`. Edge cost is included in proportion to the traversed
   fraction under the current aggregate-cost approximation. Ranking uses the
   active graph view's authoritative edge total, including decoded compact
   costs. Float components remain diagnostics and carry an explicit residual
   so their reported sum equals the authoritative ranking total. An outgoing
   join is charged exactly once when its transition is crossed. Since every
   denominator is identical, rank by cumulative loss and deterministic logical-
   route keys; recorded density is only a diagnostic equivalent.
6. For every candidate prefix at `S`, generate prefixes at `S + D`, evaluate
   each through `S + D + H`, and retain the best `beam_width` distinct prefixes
   at `S + D` using their best valid lookahead continuation. Retained prefixes
   keep their own parent chains, cumulative costs, visited-node state, current
   edge/offset, and incoming-edge state. Lookahead suffixes rank prefixes but
   are not silently committed as route history.
7. Lookahead expansion may be exhaustive inside the finite `H` window, but it
   must never control growth by pruning candidates at unequal distance. Retain
   a deterministic explicit state cap and fail loudly if it is exceeded. Any
   later scalability optimization must introduce additional equal-arc
   checkpoints, not graph-edge-depth pruning.
8. After each common-horizon ranking, reconstruct and validate the entire
   current best prefix from the segment seed. A later winner may replace all
   provisional geometry that has not yet closed a segment. If that winner's
   first reference-threshold failure lies at or before the new beam front,
   freeze that winning history through the exact first failing point, close the
   disconnected segment, and reseed all beams using the existing reset-arc
   rule. Alternatives are never retained or removed based on reference error.
   At a normal selected end, choose among states reaching the same shortened
   absolute endpoint. If every route exhausts before a common target, terminate
   without comparing unequal terminal lengths.
9. Replace the unshipped local fixed-edge and local exhaustive distance search;
   do not add compatibility handling for those incorrect semantics. Keep the
   cache payload formats and cache identities unchanged.

## Implementation

1. Add a resolved base-voxel beam-step setting alongside the existing distance
   horizon. Expose it as `--beam-step-distance`; default it to graph metadata's
   stored fiberlet DP longitudinal spacing converted through its declared
   prediction-to-base scale. Require finite positive `D` and `H`, enforce
   `1 <= beam_width <= 16`, serialize the resolved values, make width 16
   effective, and remove `--lookahead` from every caller and artifact with no
   compatibility parser.
2. Replace `SourceRouteCandidate`'s edge-only endpoint assumption with a compact
   persistent beam checkpoint containing a parent-chain handle, the explicit
   inside-edge/at-anchor state, visited anchors, absolute route length, and
   cumulative authoritative plus decomposed edge/join cost. Compute absolute
   targets as `k*D` from the origin rather than repeated `S += D`, and document
   directed offsets, finite positive edge lengths, fraction clamping, and the
   equality tolerance. Store geometry by shared parent references so 16
   histories do not repeatedly copy full prefixes.
3. Extract one shared advance-to-absolute-distance helper. It must split an
   edge at arbitrary start/end offsets, charge partial active-view costs, cross
   joins once, branch only at anchors, preserve canonical outgoing order, and
   return exact target-length states within a documented floating tolerance.
4. Implement one beam iteration in two phases: materialize all distinct
   `S + D` candidate prefixes from the retained population, then evaluate their
   descendants to the common `S + D + H` horizon. Associate each lookahead
   result with its prefix and rank
   `cost(seed..S+D) + cost(S+D..S+D+H)` without adding prefix cost twice or
   retaining suffix cost. Choose each prefix's best continuation and globally
   rank prefixes. States are equivalent only when checkpoint, incoming state,
   and visited-anchor set all match; retain the lower authoritative cumulative
   cost and canonical logical-history tie only within that equivalence. Keep
   other converged geometries distinct and retain at most 16. Canonical graph
   order and a total comparator make results independent of worker/cache order.
5. Rework segment output, matching, failure detection, and reset handling so a
   provisional best history can change while alternate histories remain live.
   Freeze the selected history only at segment termination/reset and keep
   disconnected reset segments in existing replay artifacts.
6. Update decision diagnostics to record the absolute prefix frontier,
   absolute scored endpoint, current edge/offset, cumulative-from-seed costs,
   best lookahead suffix, parent identity, and retained/pruned status. Progress
   remains global over the requested reference interval.
7. Reuse `FiberletReplayGraphSource` and its cache leases for all active and
   lookahead states. Arbitrary sub-fiberlet offsets require route geometry, so
   batch and lease route payloads only while advancing a frontier, release them
   afterward, and retain only IDs/offsets plus compact parent/cost state. Define
   the route-state cap and its counting scope, check it before allocation, and
   keep decoded chunks and parent histories under explicit byte budgets.

## Testing

1. Add synthetic variable-length graphs where one fixed beam step advances
   zero, one, and multiple complete fiberlets across different beams. Assert
   every retained prefix has the same absolute length and every scored
   continuation has the same absolute prefix-plus-lookahead length.
2. Test checkpoints starting and ending inside fiberlets, exact-anchor targets
   at `S`, `S+D`, and `S+D+H`, proportional prefix and suffix costs, exact-
   boundary ownership, and joins charged exactly once. Split one edge across
   several iterations and lookahead evaluations to prove that no fraction or
   join is double-counted.
3. Add a graph where the locally best first edge is globally worse and prove a
   width-16 persistent alternative survives multiple iterations and later wins;
   assert the population is not collapsed to one between iterations.
4. Test deterministic width pruning; converged geometry with different visited
   sets; exact-state deduplication; `D` and `H` above and below one edge length;
   non-divisible repeated `D`; graph exhaustion before `H`; cycle rejection;
   selected-end shortening inside an edge; state-cap failure; Q1 decoded-total
   ranking; failure/reset origin reset; and provisional-best replacement.
5. Retain a width-one reference case and compare it with an independently
   calculated equal-arc path. Remove or rewrite tests that encode the unshipped
   per-fiberlet-collapse behavior rather than preserving it as compatibility.
6. Run the known focused float/`position_q1` failure interval with its forced
   seed and hot existing cache identities. Require zero generated chunks and
   unchanged hashes/mtimes; report failure signature, first divergence, and
   state counts, then stop for interpretation. Defer the whole-fiber radius-768
   comparison until that focused result is accepted.
7. Repeat a focused case with a tiny decoded-cache budget and different worker
   counts; require eviction/reload and bit-identical search/output results while
   decoded cache and parent-history memory remain bounded.
8. Build `vc_fiberlets`, `test_fiberlet_paths`, `test_fiber_replay`, and
   `test_fiberlet_storage` with `-j32`; run the focused tests, full relevant test
   binaries, and `git diff --check`.

## Spec update

- Replace the current receding-horizon/per-fiberlet commitment statements in
  `planning/specs.md` with persistent beam histories, absolute equal-arc
  checkpoints, partial-edge state, two-distance step/lookahead semantics,
  cumulative-from-seed scoring, final selection, and reset behavior.
- State that `beam_width` constrains the live search population, not diagnostic
  output, and prohibit unequal-distance pruning.
- Preserve the existing cache identity, active cost-view, quantization, and
  replay failure-threshold contracts.

## Docs updates

- Update `volume-cartographer/docs/fiberlets.md` with the beam-step and
  lookahead model, state diagram/pseudocode, base-voxel units, partial-edge
  accounting, persistent-width behavior, and diagnostic fields.
- Remove examples that describe `--lookahead` as a fiberlet count or imply that
  one fiberlet is committed after each local search.

## Changelog

- Record replacement of the unshipped local receding-horizon search with the
  persistent equal-arc beam and include focused/full float-versus-Q1 results.

## Performance and correctness report

- Preserve the exact command, dataset, build type, thread count, step/horizon,
  beam width, cache state, and revision for every measurement.
- Compare old and new search wall/CPU/RSS only as an implementation benchmark;
  scientific results are expected to change because the old search semantics
  were incorrect.
- Report per-iteration min/median/max generated prefixes and lookahead states,
  plus total state-cap headroom, so combinatorial growth is explicit.
