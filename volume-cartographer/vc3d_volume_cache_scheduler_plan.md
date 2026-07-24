# VC3D Volume Cache and Render Scheduling Plan

## Problem summary

The 15 GB cache limit is the decoded in-memory chunk budget, not a download or
disk cache. Eviction already exists and is approximately global LRU across
volume caches. The failure is therefore not simply that chunks cannot be
evicted.

The current rendering path can request chunks faster than a frame can consume
them:

- A render samples the target resolution and then walks through coarser
  pyramid levels to find fallback data.
- Missing samples can enqueue chunk reads while the frame is being evaluated.
- Surface/fragment views scatter framebuffer samples through 3D volume space,
  so a small screen-space pan can intersect many full decoded chunks.
- Trilinear sampling and multi-volume composites multiply those dependencies.
- New mouse-move renders supersede old render jobs, but chunk work already
  queued for old views is not cancelled.
- Each completed chunk can trigger another render, which can discover and queue
  still more missing chunks.
- A full decoded chunk can be large. For example, a 256 x 256 x 256 `uint8`
  chunk occupies 16 MiB, so roughly 960 such chunks fill 15 GiB.

When the visible target-resolution working set plus queued stale work exceeds
the cache budget, useful fine chunks may be evicted before a frame has assembled
all of them. The next render then misses those chunks again and falls back to
coarse data. This creates an eviction/re-request loop that looks like the viewer
is permanently stuck at low resolution.

Disabling caching would not fix this. It would remove reuse while leaving the
unbounded request behavior in place. The cache should remain the backing store,
but request admission must be bounded and driven by the current view.

## Proposed design

Implement an **atlas-like scheduler** for the existing CPU renderer. This uses
the useful scheduling properties of Khartes' atlas without requiring VC3D to be
rewritten around a GPU 3D texture atlas.

The key rule is:

> Sampling may inspect resident chunks, but only the view scheduler may enqueue
> new chunk reads.

For each viewer, the scheduler maintains the dependency set for the newest view,
orders it by visual value, and admits only a bounded page of missing chunks.
Chunk completion triggers refinement and admission of the next page. The global
decoded-chunk LRU remains responsible for storage and eviction.

## Implementation plan

### 1. Finish the resident-only cache API

- Keep `IChunkedArray::getChunkIfCached()` as a read that never queues I/O.
- A resident-only lookup must not promote the chunk in the decoded-cache LRU.
  Merely probing fallback data should not make coarse chunks artificially hot.
- Preserve the existing blocking/requesting API for explicit scheduler
  admission.

### 2. Separate sampling from dependency discovery

- Update the plane and coordinate samplers so rendering can run with
  `queueMisses = false`.
- When a target chunk is absent, record a dependency instead of requesting it
  from inside the sampling loop.
- Represent a dependency with at least:
  - volume and pyramid level;
  - chunk key/index;
  - estimated decoded byte size;
  - screen-space priority;
  - view generation/request ID.
- Deduplicate dependencies before admission. Eight neighboring samples,
  trilinear taps, and repeated pixels must not create duplicate requests for the
  same chunk.

### 3. Add a viewer-local dependency pager

- Give each volume viewer a pager for the newest view generation.
- On a geometry change, discard the previous generation's unadmitted
  dependency list and build a list for the new view.
- Order target-resolution dependencies using screen-space value, favoring:
  1. chunks near the viewport center;
  2. chunks covering more visible samples;
  3. chunks needed by more composite layers/samples;
  4. deterministic chunk order as a tie-breaker.
- Admit no more than:

  `min(128 MiB decoded, shared cache capacity / 8)`

  per page.
- Also cap the number of simultaneously outstanding chunk reads so unusually
  small chunks cannot bypass the byte limit.
- When a chunk becomes ready, coalesce notifications, repaint using resident
  data, and admit another page for the current generation.
- Never admit additional work belonging only to an obsolete pan/zoom
  generation.

### 4. Make both primary and overlay rendering scheduler-driven

- Run primary-volume and overlay/composite sampling in resident-only mode.
- Route all target-resolution requests through the same bounded pager.
- Account for every participating volume when applying the page budget.
- Do not give each layer an independent 128 MiB allowance; the page limit is
  per viewer generation.

### 5. Retain completed target-resolution pixels during refinement

- Store target-resolution values and coverage in the render result.
- Reuse them while geometry, volume selection, transform, interpolation mode,
  and relevant display inputs are unchanged.
- Later refinement passes fill only target pixels that are still incomplete.
- For a composite pixel, mark target coverage complete only when every required
  layer/sample is available at the target level.
- Invalidate retained coverage on any input change that can alter volume
  sampling.

This prevents a working set larger than the cache from requiring all fine
chunks to be resident simultaneously. A chunk only needs to remain resident
long enough to contribute its pixels to the retained 2D result.

### 6. Bound coarse fallback

- Read already-resident coarse chunks as non-promoting fallback.
- Do not automatically queue every coarser pyramid level for every target miss.
- If an immediate preview needs new coarse data, permit at most one adjacent
  coarser level and charge it against the same bounded page.
- Prefer target-resolution admission after a usable preview exists.
- Do not retain coarse chunks in the LRU merely because every refinement pass
  probed them.

### 7. Keep live panning

- Continue rendering while the user pans; do not freeze loading until mouse-up.
- Coalesce render jobs to the newest geometry as currently intended.
- Replace the dependency generation immediately on each pan update so new page
  admissions favor the current viewport.
- Reads already executing may finish and enter the global cache, but they must
  not cause obsolete generations to admit follow-up pages or repaint as though
  they were current.

### 8. Keep the global cache as the storage policy

- Retain the shared decoded RAM budget and strict approximate LRU eviction.
- Do not add fixed per-pyramid-level quotas initially.
- Do not pin an entire viewport working set.
- The scheduler controls what enters the cache; the cache controls what remains
  resident.

### 9. Add diagnostics

Expose counters per viewer and globally for:

- resident decoded bytes and configured capacity;
- current-view unique target dependencies;
- estimated decoded bytes for the current view;
- admitted and outstanding chunks/bytes;
- stale completions;
- evictions;
- target-resolution pixel coverage;
- fallback level currently displayed;
- request generation ID.

These should make it clear whether a poor result is caused by slow local reads,
a working set larger than capacity, stale work, or a scheduling regression.

## Relationship to the current uncommitted changes

Preserve and build on the existing work:

- resident-only `getChunkIfCached()` access;
- sampler controls for queuing misses;
- bounded queued fallback levels;
- overlay target-coverage retention;
- the overlay page cap of `min(128 MiB, capacity / 8)`;
- associated cache and sampler tests.

Those changes address important pieces of the problem, but the final design
must also bound primary-volume admission and retain primary target-resolution
results. Otherwise the main render path can still generate a working set larger
than the cache and repeatedly restart refinement.

## Tests

### Unit tests

- A resident-only lookup neither fetches nor changes LRU order.
- Fallback probing neither queues reads nor promotes coarse chunks.
- Dependency collection deduplicates chunk keys.
- A page never exceeds its byte or request-count limits.
- Trilinear sampling and composites remain inside the single per-viewer page
  budget.
- Obsolete generations cannot admit another page after the view changes.
- Chunk completion admits work only for the current generation.
- Ordering favors central/high-coverage target chunks.

### Integration tests

- A target working set larger than cache capacity eventually reaches 100%
  target pixel coverage through retained 2D results.
- Repeated small pan updates keep outstanding work bounded.
- Two viewers sharing the global budget both make progress without bypassing
  the capacity limit.
- Plane view, surface/fragment view, overlays, trilinear interpolation, and
  multi-volume composites all progressively refine.
- After the cache reaches capacity, eviction continues and the viewer does not
  remain indefinitely at a coarse level.

### Verification

Use Ninja directly:

```sh
ninja -C build
```

Run the focused chunk-cache and sampler tests, followed by the relevant VC3D
test suite. Manually reproduce the issue with a completely local volume while
watching the new scheduler counters:

1. Set a known decoded RAM limit.
2. Open a fragment/surface view.
3. Pan slowly across a small distance until the cache reaches capacity.
4. Confirm outstanding requests remain page-bounded.
5. Stop panning and confirm target coverage converges instead of cycling at low
   resolution.
6. Pan back and confirm useful resident chunks are reused.

## Deferred work

A true Khartes-style GPU 3D atlas can be considered separately. It would combine
bounded slots, LRU replacement, and shader-side coarse-to-fine lookup, but it
would also require a larger rendering architecture change. The CPU
atlas-like scheduler should be implemented and measured first because it
directly addresses over-admission, stale work, and refinement loss without that
rewrite.

## Success criteria

- Cache use never exceeds the configured decoded RAM budget apart from small,
  documented in-flight/accounting overhead.
- New requests generated by one viewer are bounded by its page and outstanding
  limits.
- A small pan does not enqueue the entire target working set or all pyramid
  fallbacks.
- Old view generations stop producing follow-up work.
- Once interaction stops, visible target-resolution coverage monotonically
  approaches completion, even when the full 3D working set is larger than the
  cache.
- Eviction no longer causes a persistent fine-miss/coarse-fallback loop.
