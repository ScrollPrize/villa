# Task log: canonical anchor reuse across quantized fiberlets

## Findings

- Anchor fitting and filtering do not depend on position or fitted-direction
  evaluation quantization. Only rounded positions require fresh endpoint
  prediction/presence/normal sampling.
- The canonical anchor metadata/root can therefore remain the exact baseline
  identity. Existing geometry-specific fiberlet metadata/root identities remain
  unchanged, including the opaque historical u8 cost tag.
- Graph consumers previously read persisted anchor payloads directly. They now
  share the same derived chunk view as fiberlet DP for seeds, endpoint lookup,
  edges, routes, transitions, and compact-cost ownership.
- The independent plan review identified that a corridor-wide transformed
  anchor map would violate the bounded-cache contract. The implementation was
  changed to a chunk-scoped, single-flight LRU. Its reserved bytes are removed
  from the shared decoded chunk-cache budget.

## Deviations and limitations

- Replay-only compact-cost contribution/range maps remain corridor-wide for
  uint8/uint16 cost scenarios. The two requested float-cost scenarios do not
  populate them. Bounding those maps is separate from canonical anchor reuse.

## Shutdown fix

- Full-corridor runs computed their comparison and then could hang with every
  worker in a futex wait, or terminate with `SIGSEGV`, before buffered stdout
  was written. Speculative generated-cache work remained queued when the
  process-static cache worker pools began teardown; active generation could
  still hand off work or enqueue persistent writes during pool destruction.
- `PriorityThreadPool` now tracks pending and active tasks per producer group.
  `ChunkCache::cancelPendingAndWait()` cancels only its own queued work, waits
  for its own active probe/fetch work, and drains issued persistent writes.
- Cached replay teardown destroys the graph, drains fiberlet generation first
  because an active fiberlet job may still need anchors, and then drains the
  anchor cache. Quantization result rows are flushed immediately.

## Verification

- Built with:
  `cmake --build volume-cartographer/build --target vc_fiberlets test_fiberlet_storage test_fiber_replay -j32`
- Passed:
  `volume-cartographer/build/bin/test_fiberlet_storage` (13 cases)
- Passed:
  `volume-cartographer/build/bin/test_fiber_replay` (12 cases)
- Passed:
  `volume-cartographer/build/bin/test_chunk_cache` (32 cases), including an
  active-read plus queued-read cancel-and-drain regression.
- Ran `compact_axis` and then `position_q4` with one fresh output root, radius
  64, length 500 base voxels, and 32 threads. The output contains one
  `anchors.zarr` and three `fiberlets.zarr` trees (baseline plus two geometry
  variants). The canonical anchor file/size/mtime digest remained
  `507951a237fadb1aaa6830105caa296b418989521f9f8378bf6d2717f231183a`
  after the second scenario.
- `compact_axis`: 0 baseline and 0 scenario failures; baseline-to-scenario
  distance mean/median/max = 0.24631/0.21727/0.62076 base voxels; scenario-to-
  reference mean/median/max = 4.84256/4.53469/18.81216 base voxels. These values
  exactly match the prior per-geometry-anchor implementation.
- `position_q4`: 0 baseline and 0 scenario failures; baseline-to-scenario
  distance mean/median/max = 1.81550/1.37767/3.63884 base voxels; scenario-to-
  reference mean/median/max = 5.69314/5.28672/18.48181 base voxels. These values
  exactly match the prior per-geometry-anchor implementation.
- Re-ran both complete 46,147.996-base-voxel, radius-768 comparisons with the
  existing canonical anchor and baseline fiberlet caches after the shutdown
  fix. Both commands exited zero and preserved their result and `/usr/bin/time`
  files instead of hanging during process teardown.
- `compact_axis`: baseline/scenario failures = 3/2; baseline-to-scenario
  Euclidean mean/median/max = 0.53954/0.12283/43.06402, normal =
  0.24871/0.05887/30.45893, and tangential =
  0.42451/0.08110/30.44278 base voxels. Wall time was 3m28.01s and peak RSS
  4,990,472 KiB.
- `position_q4`: baseline/scenario failures = 3/6; baseline-to-scenario
  Euclidean mean/median/max = 2.31442/1.38105/125.59508, normal =
  1.10138/0.72385/67.54613, and tangential =
  1.79186/0.85755/106.57210 base voxels. Wall time was 3m56.56s and peak RSS
  5,635,604 KiB.

## Compact-axis cost follow-up

- Requested scenarios: float endpoint positions, compact two-byte fitted
  directions, and either `uint8` or `uint16` replay costs at radius 768.
- Geometry cache identity excludes replay cost bits. Both new scenarios share
  the existing `compact_axis` fiberlet cache and do not create cost-specific
  anchor or fiberlet cache roots.
- Independent review requires 18 total/17 non-baseline matrix entries, exact
  compact-axis cache-identity tests, no-mutation float/u8/u16/float coverage,
  and before/after cache-tree snapshots for the full hot-cache validation.

### Radius-768 results

- `compact_axis_cost_u8`: baseline/scenario failures = 3/2. Baseline-to-
  scenario Euclidean mean/median/max = 0.87206/0.16845/41.38829, normal =
  0.38625/0.08253/33.43784, and tangential =
  0.70179/0.11017/24.39059 base voxels. Scenario-to-reference Euclidean
  mean/median/max = 6.13465/3.62426/76.78231 base voxels. The first traversal
  took 11m51.64s at 24.15 effective cores and 6,280,896 KiB peak RSS.
- `compact_axis_cost_u16`: baseline/scenario failures = 3/2. Baseline-to-
  scenario Euclidean mean/median/max = 0.53954/0.12283/43.06402, normal =
  0.24871/0.05887/30.45893, and tangential =
  0.42451/0.08110/30.44278 base voxels. Scenario-to-reference Euclidean
  mean/median/max = 6.15644/3.63473/76.78231 base voxels. The mostly hot
  traversal took 51.60s at 1.15 effective cores and 1,063,264 KiB peak RSS.
- The u8 run reused the compact-axis cache namespace but completed 1,419
  missing fiberlet chunks and 54 canonical anchor chunks in place (about
  240 MB of payload). The subsequent u16 run completed one additional boundary
  fiberlet chunk and one anchor chunk. No existing cache payload was rewritten.
  This happens because uint8/uint16 use one affine cost range per first-endpoint
  storage chunk: establishing that stable range scans every physical prefix
  chunk that can contribute, whereas the prior float-cost replay had populated
  only the graph paths reached on demand. Restricting the range to already
  visited edges would make decoded costs depend on traversal order.

### Verification

- Built `vc_fiberlets`, `test_fiberlet_storage`, `test_fiberlet_paths`, and
  `test_fiber_replay` with `-j32`.
- `test_fiberlet_storage` passes all 14 cases, including cache-profile and
  standard-matrix checks for the new scenarios.
- The current full `test_fiberlet_paths` binary has pre-existing failures in
  local-metric bit-exact, Q4 variant, and lookahead assertions; the new matrix
  count was updated from 16 to 18. This follow-up did not relax those failures.
