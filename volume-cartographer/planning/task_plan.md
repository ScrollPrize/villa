# Task plan

## Implementation

1. Remove `beginViewRequest()` from `IChunkedArray` and `ChunkCache`.
2. Remove its implicit legacy request state: facade view ID/version, service
   view ID/epoch allocators, source view epoch, and the unused scalar entry
   priority.
3. Make context-free `tryGetChunk()` and `prefetchChunks()` explicitly use an
   empty `ChunkRequestContext`, retaining their background ownership semantics.
4. Keep scheduler group/epoch cancellation because `invalidate()` uses it to
   invalidate pending probe, source, and decode tasks.
5. Delete or rewrite tests whose only behavior is the obsolete implicit epoch.
   Preserve equivalent live coverage through explicit view snapshots and
   targeted ownership cancellation.
6. Remove dead VC3D private-cache compatibility surfaces:
   `ChunkCachePool`, `ViewerManager::chunkCacheFor()`, and the viewer
   `refreshChunkSource()` hook. Route all raw decoded reads directly through
   `Volume::sharedChunkCache()`.
7. Remove the write-only `SurfaceCache` view generation from its state, API,
   render jobs, and viewers.
8. Delete the unreferenced private-pool `FrameChunkFootprint.hpp` helper.

## Spec update

- Remove the obsolete service-wide view-epoch invariant.
- State that stable view IDs are allocated by viewers and only explicit
  versioned snapshots own interactive work.
- Clarify that scheduler group epochs remain an internal invalidation tool and
  are not an interactive priority mechanism.

## Documentation updates

- Update the current task log and status as each compatibility layer is
  removed.
- Add a changelog entry for retiring the implicit interactive epoch and private
  cache routing APIs.
- No user-facing documentation is needed because no in-tree production caller
  uses the removed interfaces.

## Testing

- Build `VC3D` and all directly affected test targets.
- Run `test_chunk_cache`, `test_chunk_cache_persist`, all chunked plane sampler
  tests, generated annotation view tests, and Lasagna line-view surface tests.
- Run repository-wide symbol searches to verify removed APIs and state have no
  references.
- Run `git diff --check`.

## Independent plan review

- The plan preserves the specification's shared source cache, explicit
  per-view ownership, background lane, atomic publication, and invalidation
  guarantees.
- Context-free access remains because it has live Python, CLI, slicing, and
  blocking-sampler callers.
- Scheduler group epochs remain because cache invalidation still cancels
  pending work by group and generation.
- No numerical, rendering, or cache-capacity behavior is intentionally changed.
