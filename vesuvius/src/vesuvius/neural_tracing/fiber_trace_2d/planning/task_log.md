# Task log: staged Fiberlet reduction performance and reporting

## Starting state

- The previous implementation improved the established Release workload from a
  median 8.09 s to 4.55 s, but did not meet the requested order-of-magnitude
  improvement.
- A run through the Debug CI binary regressed from the earlier 52.03 s baseline
  to 66.67 s wall, 139.37 s user, and 6.00 s system.
- Built-in phase timing shows local graph materialization and simplification
  remain mostly serial. The current stage report additionally materializes the
  original, input, and output graph populations separately.
- The stage-local population structure accidentally dropped all incident
  Fiberlets and retained only interior Fiberlets.
- Earlier parallel point-query cache reads were measured and reverted because
  they caused lock contention. This task instead uses chunk-granular reads and
  parallelizes only cache-free work.

## Invariants

- Existing stage retained-ID and serialized payload digests must remain:
  - stage 1 IDs `fnv1a64:7f6182d7e61b00da`, payload
    `fnv1a64:93f875a8fd522366`
  - stage 2 IDs `fnv1a64:fa6a9290546392be`, payload
    `fnv1a64:4eed9c714ad148ec`
- No numerical, ordering, or acceptance change is permitted.

## Independent plan review

- The review required the inherited stage-input population to remain a real
  pre-write snapshot; sequential per-box analyses cannot replace it.
- The existing specification defines stage `interior` per complete stage box,
  not over the geometric union. The plan now states this explicitly and tests
  a Fiberlet crossing between adjacent boxes as `all` but not `interior`.
- The bulk path must preserve evaluated anchor views, exact owner reach,
  canonical error/order behavior, both edge-cost views, and deterministic
  lowest-index worker failure selection.
- A same-input hotspot profile was added to the required measurements.

## Implementation

- Added chunk-level route access and exact incident-prefix owner enumeration to
  the shared graph source.
- Reworked local graph materialization to read every required anchor, prefix,
  and route chunk once. Directed arcs retain their existing canonical order;
  transitions still call the shared normal/tangent-aware scorer.
- Parallelized cache-free transition construction, per-entry exact searches,
  Fiberlet/anchor serialization, and independent overlay publication through
  reusable thread pools. Indexed outputs and ordered exception scans preserve
  deterministic results.
- Assigned exact entry searches to fixed strided worker partitions. Each
  worker reads the immutable materialized graph and owns reusable thread-local
  heap, ancestry, length, count, and terminal buffers; there is no scheduler,
  lock, or atomic operation in the per-entry search loop.
- Compacted exact-search ancestry to a 32-bit arc and 32-bit parent while
  retaining length and Fiberlet count in worker-local side arrays. This reduces
  allocation and memory traffic without changing queue order or loss math.
- Reworked overlay writing to operate on bulk payloads instead of repeated
  endpoint, incident-edge, and route point queries.
- Restored stage-local `all` Fiberlet populations alongside `anchors` and
  per-box `interior`, and retained the complete selected-region joint report.
- Confirmed the canonical `volume-cartographer/build` tree is Release with
  `CMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG` and rebuilt its ordinary binary.

## Validation

- Release and Clang Debug `test_fiberlet_storage`: 28 cases passed.
- Clang Debug `test_fiberlet_paths`: 87 cases passed.
- Release `test_fiberlet_paths` retains its pre-existing 295 bit-exact
  optimized-float failures at `test_fiberlet_paths.cpp:414`; this change does
  not alter that arithmetic.
- Added one-versus-four-thread analysis/simplification equivalence checks, bulk
  versus point route decode checks, a non-identity anchor view, and adjacent
  stage-box `all` versus `interior` coverage.
- `git diff --check` passes.

## Release benchmark

Command and input:

```bash
/usr/bin/time -f 'TIME real=%e user=%U sys=%S cpu=%P rss_kb=%M' volume-cartographer/build/bin/vc_fiberlets chunk-route-stats /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json /home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/fiberlet-replay-full --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --chunk 23040,17920,54784 --region-size 512 --mode staged --stage 256,0,0,0 --stage 256,128,128,128 --storage-chunk-side 128 --anchor-cache /home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/fiberlet-replay-full/cache/fnv1a64-065534383aa4f342/anchors.zarr --fiberlet-cache /home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/fiberlet-replay-full/cache/fnv1a64-c534c99591e9caf1/fiberlets.zarr --threads 32 --stats
```

- Original Release: 7.97/8.09/9.32 s wall, min/median/max.
- Final hot Release: 2.44/2.46/2.49 s wall, min/median/max.
- Final process CPU: 217-221%; user 3.82-3.88 s, system 1.54-1.62 s.
- Speedup: 3.29x by median.
- A controlled hot run measured stage-one exact analysis at 0.852 s with one
  thread and 0.108 s with 32 threads, a 7.9x wall-time speedup. The 32-thread
  analysis consumed 1.466 CPU seconds, or 13.5 effective cores; its short,
  uneven searches and shared memory bandwidth limit scaling below 32x.
- Stage 1 ID/payload hashes remained
  `fnv1a64:7f6182d7e61b00da` / `fnv1a64:93f875a8fd522366`.
- Stage 2 ID/payload hashes remained
  `fnv1a64:fa6a9290546392be` / `fnv1a64:4eed9c714ad148ec`.

The final stage-local table reports:

```text
stage scope    original input output stage_reduction cumulative_reduction
1     anchors  4371     4371  4002   8.44%           8.44%
1     all      78462    78462 48393  38.32%          38.32%
1     interior 34287    34287 5281   84.60%          84.60%
2     anchors  631      612   543    11.27%          13.95%
2     all      13750    6638  3801   42.74%          72.36%
2     interior 5730     3397  563    83.43%          90.17%
```

## Hotspot profile

Callgrind ran the same Release input with `--threads 1` because Callgrind
aborted after accumulating roughly 500 thread records in the 32-thread run.
It collected 24.72 billion instructions. Exact per-entry route search was the
largest resolved exclusive function at 4.47 billion instructions (18.10%).
Release phase timing at 32 threads shows stage-one materialization at about
0.25 s and 4.1 effective cores, exact analysis at about 0.11 s and 13.5 cores,
and simplification at about 0.17 s and one core. Publication and filesystem
work remain the largest wall-time component and are bounded by record-exact
temporary overlay writes rather than graph cache point queries.

## Limitation

- The measured Release median speedup is 3.29x, not the requested 10x.
  Deterministic overlapping-box semantics keep boxes serial, while record-exact
  temporary overlay publication and single-threaded simplification now account
  for most remaining wall time. The Debug-to-Release difference is deliberately
  excluded from the algorithmic speedup.
