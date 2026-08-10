# Task log: preload and parallelize fiberlet DP

## Findings and baseline

- Candidate DP is currently serial; `--threads` only reaches per-candidate
  fiber sampling.
- Per-candidate normal sampling calls the automatic-thread overload, so even
  `--threads 1` repeatedly starts hardware-sized normal sampling work.
- Every overlapping candidate reconstructs and resamples its corridor despite
  the decoded chunk cache. The cache avoids raw chunk I/O but not voxel decode,
  normal interpolation, temporary allocation, or worker-team overhead.
- Release reference crop:
  `--crop 13600,20256,18144,192,192,192`, 200 anchors, 4,813 generated pairs,
  676 searched/accepted paths, score min/mean/max
  3.18031/11.0126/20.8916.
- Default-thread trace time was 11.6515 s. `--threads 1 --no-slices` trace time
  was 9.37721 s; process wall/user/system was 9.44/146.49/0.48 s with 73,848
  KiB peak RSS, confirming hidden normal-sampler parallelism.
- `perf` is unavailable in this checkout, so source inspection plus controlled
  wall/CPU measurements are the current profiler substitute.
- Independent review required endpoint attachment voxels in preload bounds,
  exact double-valued sample storage, an explicit rectangular-box contract,
  byte-level checked allocation, continuing workers for deterministic exception
  selection, expanded concurrency/indexing tests, repeated benchmarks, and
  structural docs. The plan incorporates these items.
- For the explicitly small-crop phase, the preload will use the existing generic
  `NormalSampleWithDerivative` batch as a temporary result and immediately
  compact it to normal vector/validity. Its derivative matrices and diagnostic
  strings increase peak preload memory; a new compact Lasagna batch API is
  intentionally deferred because it is not needed for current crop sizes and
  would broaden this optimization task.

## Implementation

- Added a thread-count-aware virtual normal batch overload; generic samplers
  retain existing behavior, while `LasagnaNormalSampler` uses its existing
  explicit-thread implementation.
- Split path processing into deterministic candidate generation, a single
  dense preload, concurrent candidate solving, and serial diagnostics.
- The preload is the rectangular enclosing ZYX box of all clipped candidate
  Hermite corridor bounds plus endpoint attachment neighborhoods. It enumerates
  unique voxels in ZYX order and samples fiber predictions and Lasagna normals
  once each.
- Dense storage retains exact `FiberStoredPredictionSample` values and
  `cv::Vec3d` normal vector/validity. Candidate workers perform checked immutable
  lookups with no nested sampling.
- A fixed worker pool writes only existing canonical candidate slots. All tasks
  continue after exceptions; the lowest search-index error is rethrown after
  joining. Aggregate diagnostics are derived serially.
- Added runtime reporting for preload voxels, estimated preload working bytes,
  candidate workers, and generation/preload/search phase times. These remain
  absent from artifacts.
- Added a serialized optional core progress callback and CLI stderr reporting
  for completion, percentage, elapsed time, rate, and ETA. Updates are monotonic
  and rate-limited; the coordinator guarantees terminal completion reporting,
  including the zero-search case, before deterministic error propagation.
- Added one-shot/unique preload, requested-thread propagation, zero-search,
  narrow-corridor attachment, generic normal sampler, and one-vs-multi-worker
  deterministic-output coverage.

## Deviations and limitations

- No production-only hook was added to force multiple candidate exceptions or
  block workers, so deterministic lowest-exception and active-overlap behavior
  are established by implementation structure rather than synthetic tests.
  Tests do verify the requested worker bound and exact multi-worker output.
- The existing generic `NormalSampleWithDerivative` array is retained as a
  temporary preload result before compacting to vector/validity. This inflates
  peak memory but is accepted for the explicitly small current test crops.
- Dense preload samples the full enclosing box, including gaps between
  corridors. Tiling/masking and a compact Lasagna normal batch are deferred.
- System `perf` is not installed. Hotspot evidence is source-level plus phase,
  wall, CPU, and RSS measurements rather than sampled call stacks.
- The progress-plan review required stale concurrent update suppression,
  search-only timing, explicit zero/failure terminal behavior, and captured
  callback exceptions; these are included. A fake clock and production worker
  blocking hooks are not added solely to test intermediate rate limiting.

## Validation

- Release build: GCC, `CMAKE_BUILD_TYPE=Release`, `-O3 -DNDEBUG`.
- Build command:
  `cmake --build build --parallel 32 --target vc_fiberlets test_fiber_anchors test_fiberlet_paths test_fiber_trace3d`.
- `ctest --test-dir build --output-on-failure -R
  'test_(fiber_anchors|fiberlet_paths|fiber_trace3d|lasagna_normal_sampler)$'`
  passed all four binaries: 19 anchor, 14 path, 46 tracer, and the Lasagna
  normal-sampler suite.
- Benchmark command used `vc_fiberlets paths` with the S1 Fiber manifest,
  regenerated 192-base-voxel crop anchors, `las_tmp.lasagna.json`, defaults,
  `--no-slices --stats`, and `/usr/bin/time`; three separate processes per
  implementation were measured after warming the OS file cache.
- Reference trace counts stayed at 4,813 candidates and 676/676 successful
  searches; score min/mean/max stayed 3.18031/11.0126/20.8916.
- Baseline wall min/median/max: 11.80/12.06/12.09 s. Optimized:
  0.43/0.43/0.44 s, a 28.0x median wall speedup. Internal trace median changed
  from 11.9944 s to 0.3661 s (32.8x).
- Baseline user CPU min/median/max: 219.52/227.78/229.04 s. Optimized:
  10.88/10.89/11.04 s. Peak RSS median increased from 74,756 KiB to 157,192
  KiB due to dense/temporary preload storage.
- All six before/after `fiberlets.json` files were byte-identical with SHA-256
  `a790790b28b482586ae16d2efad8643736649236ea7f1cb45e90e5ca1ae450ff`.
- A 512-base-voxel S1 crop produced 4,072 anchors, 291,616 candidates, and
  42,952/42,952 successful searches. It preloaded 416,176 voxels, reported
  126,517,504 estimated preload working bytes, traced in 23.0707 s, completed
  in 25.98 s wall, and peaked at 1,516,228 KiB RSS.
- The same 512 crop with progress emitted monotonic roughly one-second updates
  from 4.7 percent/20.3 s ETA through 96.7 percent/0.8 s ETA, then exactly
  `42952/42952`, 100 percent, and zero ETA. Trace/wall time remained
  23.0073/25.94 s.
