# Plan: remove serial full-corridor replay setup

## Selection and scheduling

1. Extract the exact segment-to-AABB distance calculation into the existing
   shared fiber geometry module and make the replay tube's R-tree query support
   exact anchor-cell intersection tests.
2. Allow cache-backed preprocessing to use an exact anchor-cell predicate in
   place of a globally materialized sorted cell vector. Enumerate owned cells
   in canonical Z/Y/X order inside each requested anchor chunk, select them with
   the shared tube query, and pass them through the existing extraction API's
   external context expansion so neighboring NMS suppressors remain available
   while only selected owner-cell anchors are persisted. Retain the existing
   post-refinement anchor-position predicate as a second exact check.
3. Build reference schedules directly at storage-chunk resolution. Preserve
   deterministic arc-based ordering, schedule only chunks intersecting the
   requested reference interval/radius, and retain the graph's existing lazy
   cache loading and eviction behavior. Keep the indexed tube point predicate
   on fiberlet DP interiors, not only endpoints. After progress bookkeeping and
   callbacks are initialized, submit anchor dependencies and fiberlet chunks
   through `prefetchScheduled(schedule, 0, schedule.size(), false)` before
   graph evaluation starts.
4. Stop collecting anchor cells when constructing the cached replay tube and
   remove cell-list serialization from its cache identity. Keep the complete
   canonical clipped reference points, interval semantics, radius, source and
   algorithm metadata, and a selection-version discriminator identically in
   both cache identities. Eager extraction and visualization retain their
   explicit cell lists.

## Progress and concurrency

5. Keep hardware concurrency as the CLI default. Ensure setup returns quickly
   enough for the already-parallel greedy and fiberlet evaluators/cache workers
   to start; detailed `--stats` output will expose chunk generation while the
   default remains the single overall progress display.

## Verification

6. Add focused tests comparing predicate-selected and explicit-selected anchor
   chunks, including a winning suppressor across a storage-chunk boundary.
   Compare retained bytes and IDs with eager extraction. Assert the scheduled
   key set is complete against projecting the explicit eager cell population to
   owner chunks, ordering is deterministic, and nonblocking prefetch submits
   every anchor dependency and fiberlet chunk. Verify geometry/radius changes
   change both cache fingerprints and stale chunks are rejected.
7. Build with RelWithDebInfo and `-j32`; run anchor, storage, replay, and path
   tests. Compare cached/eager 5,000-base-voxel radius-64 replay artifacts byte
   for byte.
8. On the RelWithDebInfo build, run three fresh-cache repetitions of the exact
   full-fiber Paris4 radius-768 command from the task, recording time to
   parallel cache/tracer work, wall time, process user+system CPU, effective
   cores, sampled CPU utilization, and mean/median/min/max. Compare with the
   reproduced greater-than-20-second serial setup baseline. Use shorter bounded
   runs if full completion is impractical, but keep command, input, cache state,
   stop condition, and any limitation explicit.

## Spec update

Document that cache-backed replay stores corridor provenance but discovers
anchor cells per requested chunk, and that reference scheduling is performed at
storage-chunk resolution rather than by precomputing the complete anchor-cell
population.

## Documentation updates

Update `volume-cartographer/docs/fiberlets.md`, the planning changelog, status,
and task log with the chunk-native selection flow, exactness constraints,
validation commands, and measured before/after startup behavior.
