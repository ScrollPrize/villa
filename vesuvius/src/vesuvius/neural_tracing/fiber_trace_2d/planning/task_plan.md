# Plan: cache decoded fiberlet graph chunks

## 1. Shared cache payload support

1. Add a type-erased decoded payload interface to the existing chunk fetch and
   result contract. A payload reports its resident byte size and is leased by a
   shared pointer.
2. Extend `ChunkCache` entries, result publication, pin detection, local/shared
   byte accounting, invalidation, and eviction to handle either ordinary dense
   byte chunks or typed decoded payloads. Preserve all existing byte callers.
3. Keep persistence at the fetcher/storage boundary. Do not retain a second
   serialized copy in a typed resident entry and do not introduce another LRU.
   A typed fetcher may provide transient persistence bytes, but exactly one
   decoded representation is resident and charged. Fiberlet generated caches
   continue to use their authoritative sparse datasets rather than the generic
   persistent-cache option.

## 2. Decoded anchor and fiberlet chunks

4. Add parse-once dataset read/validation and publication paths. Materialize
   anchor, prefix, and route payload objects once in the existing generated
   fetchers before publishing a cache hit; reuse that parsed object for header
   validation rather than deserializing again.
5. Build a deterministic two-endpoint incident-edge index once in each decoded
   prefix payload. Keep prefixes in canonical serialized order and make index
   memory part of the cache charge.
6. Keep anchors in the anchor cache. Keep prefix/connectivity and route levels
   in the fiberlet cache. Charge nested route vectors and index capacity, not
   only their top-level objects.

## 3. Chunk-based graph traversal

7. Replace graph-side byte deserialization with checked typed-payload leases.
   Batch-prefetch all possible neighboring owner chunks before incident lookup,
   then query their cached endpoint indices in deterministic key/order order.
   Convert on-demand fiberlet generation's anchor dependencies to the same
   typed anchor leases; no non-graph byte consumer may retain the old path.
8. Extract one shared exact endpoint-geometry helper from the existing curved
   route reconstruction and split replay edge descriptors from complete route
   geometry in both eager and cached sources. Prefix and
   anchor payloads provide endpoint IDs, total cost, length, and exact endpoint
   directions for beam ranking and transition scoring.
9. Request and reconstruct the route-level payload only after an edge is
   selected for commitment. Reuse the transition already chosen by lookahead
   instead of evaluating it again.
10. Keep stable IDs in beam state; no pointer into a cache entry survives its
    payload lease. Query results may copy small stable-ID descriptors or final
    committed geometry, but must not copy whole decoded chunk vectors.

## 4. Correctness and performance validation

11. Add cache tests proving typed payload hits reuse one object, leases pin it,
    real decoded bytes are charged, and eviction/refetch uses the existing LRU.
    Exercise two independent local cache ceilings under one shared budget and
    prove global-oldest eviction and lease pinning across both caches.
12. Add fiberlet tests proving stored, generated, concurrent same-key, and
    corrupt chunks follow the parse-once/error contract; prefix/anchor hits are
    not decoded repeatedly,
    incident lookup includes cross-chunk ownership, route chunks are untouched
    during lookahead, and selected route geometry remains exact. Cover forward
    and reverse zero-, one-, and many-interior-point endpoint geometry.
13. Require eager and cached replay results to remain identical for the focused
    fixture, including ordering, failures, costs, and geometry.
14. Build affected targets with `-j32`; run focused cache, storage, graph, and
    replay tests.
15. Instrument route-level fetch/decode counts with beam width and lookahead
    greater than one, discarded candidates, reverse arcs, reseeding, and an
    eviction/refetch case. Only committed edges may request route chunks.
16. Measure a serialized-storage-warm, decoded-cache-cold replay before and
    after using the same
    reference interval, cache roots, beam, lookahead, build type, and thread
    count. Run at least three iterations and report mean, median, min/max or
    p95, profiler hotspots, cache state, wall/CPU time, and effective cores.
    Measure graph-thread CPU separately from the concurrently running greedy
    evaluator where possible. Numeric output must match. Also distinguish this
    from steady-state decoded-cache-warm access within one process.

## Spec update

Clarify the already-required behavior: decoded typed chunks are entries in the
existing `ChunkCache` LRU; anchor and fiberlet caches remain separate; incident
halo prefetch is batched; and route blocks are loaded only for committed edges.
Remove wording that could imply per-query decoding or a separate graph LRU.

## Documentation update

Update `docs/fiberlet_storage.md` and `volume-cartographer/docs/fiberlets.md`
with the concrete cache payload types, memory accounting, neighbor-prefetch
flow, and prefix-only beam traversal.

## Changelog update

Record the removal of repeated whole-chunk decoding and route reconstruction
from on-demand beam search, plus measured replay performance.
