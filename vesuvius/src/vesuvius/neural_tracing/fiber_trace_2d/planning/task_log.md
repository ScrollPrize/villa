# Task log: concise replay progress and latest fiberlet optimizations

## Initial findings

- `fiberlet-replay` previously printed setup, extraction, cache-open, every
  cache chunk, every failure, both evaluator streams, cache statistics,
  visualization, and publication details unconditionally.
- `--stats` existed but was accepted only for the standalone `paths` command.
- `fiber-lets2` had three unmerged commits: path-search optimization, anchor
  extraction optimization, and their durable documentation.
- The current branch already requires byte-identical eager and float-cache
  replay, so that remains the merge correctness gate.

## Independent review

- Accepted: make concurrent evaluator accumulation explicitly monotone; keep
  100% reserved until requested output is published; test quiet/statistical
  output and final-line handling; measure actual CPU as well as wall time; merge
  cumulative specs/docs semantically; compare cache populations as well as the
  final replay.
- The review's shared-stage/sample-count findings describe an earlier benchmark
  task and do not apply to this output-and-merge task. The merged optimizations
  will still be checked for duplicate work through their existing diagnostics.

## Progress implementation

- Default replay now displays one 24-column bar. Concurrent fractions are
  clamped and accumulated independently; their minimum drives tracing progress.
- Tracing reserves the final one percent for publication. Visualization runs
  reserve twenty percent for the overview and per-failure artifacts, preventing
  a displayed 100% while requested output is still being written.
- `--stats` is now accepted by `fiberlet-replay` and restores all prior detailed
  rows. Default eager/local visualization extraction also suppresses its nested
  anchor and path progress callbacks.
- Validation: `cmake --build volume-cartographer/build -j32 --target
  vc_fiberlets` passed. A cold 500-base-voxel default replay printed only the
  progress display and terminal result; a 128-base-voxel `--stats` replay
  retained stage, chunk, evaluator, cache, and terminal rows.

## Merge

- Merged the latest `fiber-lets2` path-search and anchor-extraction
  optimizations. The shared production functions are used directly by both
  eager extraction and on-demand cache chunk generation; no separate port or
  copied implementation is required.
- Combined the cache branch's sparse per-cell result retention with the newer
  shared-observation/proposal buffers. Extraction profile version 30 reports
  both memory-accounting families.
- The incoming chordal interior-smoothness approximation was intentionally not
  ported. It changed the objective and failed the bit-exact metric fixture.
  This checkout requires performance changes to preserve numerics, so the
  merged corridor and interpolation optimizations retain the existing angular
  scorer.
- The incoming direct-centroid mode, nearest-Gaussian lookup, and reordered
  proposal geometry were also excluded because their own measurements reported
  route movement. Compact proposal records retain absolute positions and the
  original subtraction order; centroid updates always use the exact spatial
  objective and peak weights retain `exp`.
- The shared-observation, sparse proposal-index, retained-membership,
  final-Gaussian reuse, corridor-admission, and interpolation-page changes are
  used by both eager extraction and cache chunk generation without duplicated
  implementations.

## Validation and measurements

- RelWithDebInfo build: `cmake --build volume-cartographer/build -j32 --target
  vc_fiberlets test_fiber_anchors test_fiberlet_paths test_fiberlet_storage
  test_fiber_replay`.
- Passing suites: 85 anchor cases, 11 storage cases, and 11 replay cases.
  `test_fiberlet_paths` retains 295 pre-existing bit-pattern fixture failures
  at line 406 and the pre-existing synthetic lookahead expectation at line
  1486; the merge introduced no additional loader/schema failure.
- Workload: Paris4 fiber prediction and Lasagna normals, David fiber
  `dj_20260805T025256484_000003.json`, first 5,000 base voxels, radius 64, 32
  threads, three fresh output/cache roots per mode.
- Eager wall seconds: 3.96, 3.93, 3.90; mean/median 3.93, range 3.90-3.96.
  Mean process CPU was 98.8 seconds (25.1 effective cores), and peak RSS was
  1.35 GiB.
- Cold cached wall seconds: 5.78, 5.78, 5.76; mean 5.77, median 5.78, range
  5.76-5.78. Mean process CPU was 122.7 seconds (21.3 effective cores), and
  peak RSS was 0.51 GiB.
- Relative to the immediately preceding measured 4.43-second eager and
  7.04-second cold-cache runs, wall time improved about 11% and 18%,
  respectively.
- All six `fiber_replay.json` files are byte-identical with SHA-256
  `9781e00ae129b5fef098246c163ba1f737eca3b8a3fcceba6c90e45087b10a91`.
  They also match the prior exact cached artifact, confirming that the retained
  optimizations do not change emitted replay geometry or failures.
