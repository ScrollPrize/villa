# Task log: cache decoded fiberlet graph chunks

## Initial finding

- Revision `74af727e8` stores materialized serialized bytes in `ChunkCache`, but
  `FiberletChunkGraphSource` deserializes entire prefix, route, and anchor
  chunks for individual graph queries.
- `bestLookaheadRoute` repeatedly calls `outgoing`, `arc`, and `transition` for
  overlapping beam candidates. The cached adapter therefore repeats whole-
  chunk decoding, route reconstruction, and source-volume sampling on one
  thread, whereas the earlier eager graph used indexed resident objects.
- This contradicts the existing storage specification, which requires a
  memory-bounded LRU of decoded anchor prefixes, fiberlet search prefixes, and
  selected route blocks.
- The correction will extend the existing shared cache abstraction. It will not
  add a fiberlet-only secondary LRU or change the persisted chunk format.

## Baseline

- Dataset: Paris4 `fiber_s1_002`, David fiber `...000003`, existing anchor and
  fiberlet cache roots under `data/workdir3/fiberlet-replay-full/cache`.
- Command settings: `fiberlet-replay`, `--length 100`, `--radius 64`,
  `--threads 32`, default beam 16 and lookahead 3, current build.
- Three cache-warm runs took 17.17, 17.09, and 17.13 seconds wall
  (mean/median 17.13 s, min/max 17.09/17.17 s; with three samples p95 is
  represented by the 17.17 s maximum). No chunks were generated and there were
  no replay failures. The run committed two fiberlet edges over 100 base
  voxels. Process CPU was 101-134%; after greedy completion, graph replay was
  visibly one-core work.

## Independent plan review

- The review identified two blockers: the dataset read/publication path already
  deserialized during validation, and on-demand fiberlet generation still
  consumed anchor-cache bytes. The plan now requires one parsed payload to be
  reused for validation/cache publication and converts every anchor dependency
  consumer to typed leases.
- The review also required a shared curved-domain endpoint helper, explicit
  two-cache/shared-budget tests, an unambiguous typed-payload persistence
  contract, route-fetch instrumentation across discarded candidates and
  reseeding, non-owning chunk query results, and a repeated cache-state-aware
  performance protocol. These are incorporated above.

## Implementation

- Extended `ChunkCache` with a type-erased decoded payload lease. Typed opaque
  entries participate in the existing local and shared byte budgets, pinning,
  invalidation, and LRU eviction. Fiberlet caches disable generic persistence;
  their sparse datasets remain the sole serialized authority.
- Added decoded anchor, prefix, and route payloads. Prefix payloads build one
  deterministic two-endpoint incident index and include it in their resident
  byte charge. Dataset reads decode and validate once before cache publication.
- Converted anchor dependencies and graph queries from serialized bytes to
  typed cache leases. Incident queries batch-prefetch the complete possible
  owner halo through the fiberlet cache.
- Extracted exact endpoint-step reconstruction from the existing curved-domain
  geometry. Beam expansion now reads prefix/anchor descriptors only. Full route
  payloads and polyline reconstruction occur only after selecting the edge to
  commit; the already-selected transition is reused.
- Added materialization diagnostics to the final `fiber_replay_cache` row. The
  measured run decoded four anchor chunks, four prefix chunks, and one route
  chunk; its two committed edges shared that route chunk.

## Validation

- Build: `cmake --build volume-cartographer/build -j32 --target vc_fiberlets test_chunk_cache test_fiberlet_storage test_fiberlet_paths test_fiber_trace3d`.
- `test_chunk_cache`: 31 cases passed, including typed-object reuse, pinning,
  eviction/refetch, two local caches under one shared budget, and concurrent
  same-key coalescing.
- `test_fiberlet_storage`: 11 cases passed, including exact endpoint steps for
  zero/one/many interior points, cross-chunk adjacency, route deferral, and LRU
  reload.
- `test_fiber_trace3d`: 54 cases passed.
- `test_fiberlet_paths` still reports the same pre-existing 295 bit-exact
  failures at `test_fiberlet_paths.cpp:406`; no new failure location appeared.
- The post-change `/tmp/fiberlet-cache-fix-final/fiber_replay.json` is byte-for-
  byte identical to the baseline artifact.

## Performance result

- Protocol: same Paris4 manifests, existing serialized cache roots,
  `--length 100 --radius 64 --threads 32`, default beam 16/lookahead 3,
  serialized-storage-warm and decoded-cache-cold process per run.
- Baseline wall seconds: 17.17, 17.09, 17.13; mean/median 17.13, min/max
  17.09/17.17.
- Changed wall seconds: 0.21, 0.19, 0.20; mean/median 0.20, min/max
  0.19/0.21. Median speedup is 85.7x.
- Changed process user time was 5.63, 5.05, and 5.29 seconds; RSS was
  89.1-89.9 MiB. Parallel cache materialization explains process CPU exceeding
  wall time; replay itself retained zero failures and the same committed path.
- `perf` is unavailable on this machine. Callgrind collected 6.00 billion
  instruction references for the full concurrent command; the largest named
  exclusive entry was the OpenMP runtime dispatcher at 36.7%. The former
  graph-side repeated deserialization/reconstruction path did not appear as a
  remaining hotspot.

## Deviations and limitations

- No install was performed to obtain `perf`; Callgrind was used instead.
- The post-change profile includes the concurrent greedy evaluator and cache
  decode workers because the CLI has no graph-only profiling mode. Disk decode
  counts provide the graph-specific route-load check.
