# Plan: incremental fiberlet replay prefixes

## Baseline and invariants

1. Record a hot-cache bounded replay runtime and preserve its replay JSON as a
   correctness baseline. Use the same graph, cache, interval, radius, build,
   and thread count before and after the change.
2. Preserve cumulative cost arithmetic, successor order, cutoff comparisons,
   route tie ordering, reference matching, threshold measurements, and final
   serialized segment contents exactly.

## Persistent search state

1. Replace root-to-tip logical-arc vectors used during search with canonical
   seed-local persistent logical-route nodes. Intern `(logical parent, logical
   arc)` exactly so physically distinct aliases of the same logical route share
   identity while retaining their separate physical histories, costs, joins,
   geometry, and visited state.
2. Compare logical routes exactly and deterministically with fixed-width binary
   lifting: an ancestor sorts before its descendant and otherwise the first
   divergent logical arc determines order exactly as the former vectors did.
   Allocation addresses/IDs never determine ordering. Use a sharded weak
   interner and sweep expired entries after decisions so worker scheduling does
   not affect results or retain pruned states. Materialize logical-arc vectors
   only when explicit decision diagnostics require them.
3. Replace the linked visited delta plus full `std::set` compaction with an
   immutable deterministic Patricia trie over the complete fixed-width
   `FiberletStorageKey`. Full keys remain at leaves, so membership is exact and
   insertion/query depth is bounded by key width rather than prefix length.
4. Keep per-front candidate selection and tie behavior equivalent while using
   canonical logical nodes only at the equality/diversity/order boundaries
   where the former implementation used logical vectors.

## Incremental reference evaluation

1. Associate materialization state with live persistent physical-route
   candidates. Each immutable contribution records its physical history node,
   parent contribution, last emitted point, exact matched reference arc,
   output point/step counts, per-edge points and matches, terminal status, and
   authoritative cumulative five-component costs and prediction length.
2. Retain a segment-local table of evaluated physical-history nodes. When a
   selected beam changes, walk back only to its nearest evaluated ancestor and
   evaluate the previously unseen suffix once. Do not reread or rematch its
   established prefix, and do not touch unselected route payloads.
3. Preserve the exact point-by-point reference matcher and Lasagna-normal
   threshold evaluation for each newly encountered route point.
4. Never evaluate a descendant after a contribution records a failure or
   reference-end termination. Retain selected contributions only for the
   current segment, reuse them through shared physical ancestry, and reset all
   state at reseed. Route payload leases are not retained.
5. Assemble the public segment, matches, indices, committed steps, and consumed
   nodes once at failure/reference end. Explicit full decision diagnostics may
   materialize their requested full route payloads because that artifact itself
   stores full per-decision routes; normal replay must not pay that cost.

## Performance accounting

1. Keep search counters semantically unchanged.
2. Benchmark hot-cache replay before and after, reporting wall time, CPU time,
   interval, radius, and failure counts. Confirm later checkpoints do not
   degrade merely because the selected prefix is longer.

## Testing

1. Add focused tests for canonical logical-route equality/order including
   logical aliases and equal-cost ties, exact persistent visited membership,
   and switching to a beam whose suffix was not previously materialized.
2. Cover bounded/exact search, diagnostics off/on, 1/32 threads, mid-edge
   failure, terminal partial edges, and checkpoints that add no fiberlet.
3. Compare before/after replay JSON byte-for-byte for the hot-cache workload,
   including component costs, matches, committed steps, failure locations, and
   geometry. Progress text is excluded.
4. Build `vc_fiberlets`, `test_fiberlet_paths`, `test_fiber_replay`, and
   `test_fiberlet_storage` with `-j32`; report the known
   existing fixture failures separately from new failures.

## Spec update

- Specify that persistent identities are an implementation optimization only:
  bounded reconvergence, exact search, cutoff rules, whole-fiberlet commitment,
  cache/publication formats, and final output remain unchanged. Search
  identities, cycle state, and selected-route reference evaluation are
  incremental; only final output assembly is linear in segment length.

## Docs update

- Document the incremental prefix state and clarify that full route payload
  construction is restricted to final output and explicit decision diagnostics.

## Changelog

- Record removal of repeated root-to-tip replay materialization, logical-key
  construction, and visited-set copying.
