# Task log: remove serial full-corridor replay setup

## Reproduction

- The CLI already defaults anchor, path, and trace worker counts to
  `std::thread::hardware_concurrency()`; omitting `--threads` did not select one
  worker.
- A full-fiber radius-768 run consumed 43.3 CPU seconds over 43.7 wall seconds
  before interruption, confirming one effective core.
- A 20-second `--stats` diagnostic completed `trace_setup` in 0.006 seconds but
  never reached `cache_open`. The blocked phase is the synchronous
  `makeFiberReplayTube()` call.
- Cached replay calls `fiberAnchorCellsNearPolyline()` once while constructing
  the processing tube and again from `referenceChunkSchedule()`. The function
  visits every dense reference segment, expands its radius-sized anchor-cell
  box, performs exact segment/AABB tests, and inserts into a serial `std::set`.
  Radius 768 is about 24 anchor cells for the current 32-base-voxel cell side,
  so heavily overlapping work dominates before cache workers exist.

## Independent review

- Incorporated explicit nonblocking prefetch placement, post-refinement anchor
  and DP-interior corridor predicates, cross-chunk NMS equivalence, canonical
  cache identity/versioning, schedule completeness, and repeated CPU/wall
  measurement requirements into the implementation plan.

## Implementation

- Shared the canonical segment/AABB distance primitive with the replay tube's
  indexed containment query.
- Cache-backed replay no longer asks `makeFiberReplayTube()` to collect anchor
  cells. The on-demand preprocessor enumerates and filters only the cells owned
  by a requested anchor chunk.
- Reference scheduling now uses the canonical exact tube selector at storage-
  chunk resolution. Existing halo dependencies, cross-chunk NMS context,
  post-refinement anchor filtering, fiberlet-interior filtering, cache LRU, and
  nonblocking schedule prefetch are unchanged.
- Removed the materialized cell list from cache identity and bumped the
  corridor selector discriminator. The complete clipped reference geometry and
  radius remain part of the fingerprint.

## Verification

- Built `vc_fiberlets`, `test_fiber_replay`, `test_fiberlet_storage`, and
  `test_fiber_anchors` with `cmake --build volume-cartographer/build -j32`.
- `test_fiber_replay`: 12 cases passed, including exhaustive indexed/canonical
  anchor-cell selection equality and the no-materialized-cell path.
- `test_fiberlet_storage`: 11 cases passed. `test_fiber_anchors`: 85 cases
  passed.
- A completed Paris4 5,000-base-voxel radius-64 replay took 5.90 seconds wall,
  126.06 CPU seconds (21.3 effective cores), and 516372 KiB peak RSS. Its
  `fiber_replay.json` SHA-256 remained
  `9781e00ae129b5fef098246c163ba1f737eca3b8a3fcceba6c90e45087b10a91`.
- A fresh-cache full-fiber radius-768 run reached `cache_open` after 0.006
  seconds. The bounded 20.06-second run used 549.55 user plus 10.61 system CPU
  seconds (27.9 effective cores) and 643004 KiB peak RSS. The first nonempty
  anchor chunk used 22.01 CPU seconds in 0.713 wall seconds.
- Three additional fresh-cache radius-768 runs, each bounded at 10 seconds,
  measured 28.08/28.25/28.27 reported CPU cores (mean 28.20, median 28.25,
  min 28.08, max 28.27). Peak RSS was 586-591 MiB. These runs intentionally
  ended before replay completion and measure startup/cache extraction only.
