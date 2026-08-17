# Task Log: accelerate fiberlet tracing

## Initial Brainstorm

The replay path recomputes full tube anchor and fiberlet extraction before graph
tracing. Coarse measurements attribute about 30% of extraction time to anchors
and nearly all remaining time to fiberlets; graph construction and replay are
not current priorities. Eliminating anchor cost alone therefore has an Amdahl
upper bound of about 1.43x.

Candidate optimizations, to be ranked after profiling:

1. Add a direct stored-grid integer sampler. Exact integer lookups currently
   pass through physical-coordinate mapping and generic trilinear cube sampling;
   direct chunk-aware reads can preserve values while removing redundant corner
   work in anchor and fiberlet prediction sampling.
2. Spatially index the replay reference tube. Point containment currently scans
   the reference polyline for anchors and candidate nodes. A conservative
   cell/BVH index can retain the exact final point-segment test.
3. Reduce fiberlet geometry preparation. Accept nodes known to lie inside their
   curved layer's inner radius directly and reserve full centerline checks for
   the outer annulus.
4. Replace hot hash tables where key ranges are dense or sortable:
   interpolation corners, global scoring voxels, per-candidate node indices,
   and reusable DP scratch. Preserve deterministic ordering with sort/unique or
   dense indices.
5. Reuse anchor work: compute overlapping 27-point presence-gradient stencils
   once, reuse observation buffers, align tiles with source chunks/tube
   occupancy, and consider compact structure-of-arrays fitting input.
6. Pipeline or coalesce prediction and normal reads after measuring whether the
   batched samplers or materialization dominate.
7. Add persistent caches keyed by source/config/reference/tube identity: full
   extraction, per-cell pre-NMS anchors, and potentially final per-anchor-pair
   paths. Reuse cached primitives for failure visualization instead of local
   re-extraction.
8. Batch multiple reference fibers so overlapping cells and interpolation
   corners are loaded once.
9. Tune decoded-cache budgets, tile dimensions, batch sizes, and thread counts
   only from measured workload behavior.

## Findings

- Fiberlet reports already time candidate generation, preparation, corner
  merge, prediction sampling, normal sampling, materialization, and search.
- Anchor reports expose only aggregate elapsed time, leaving the 30% anchor
  share unexplained.
- Preparation combines curved geometry, candidate-node enumeration, repeated
  centerline/tube tests, and corner hash insertion. Materialization combines a
  global scoring hash build with per-node interpolation. Search combines a
  per-candidate node hash with DP.
- `benchmark` prints broad fiberlet phases, while `fiberlet-replay` prints only
  aggregate anchor/fiberlet timings and workload totals.

## Measured 5,000-Voxel Replay Profile

Canonical workload:

```bash
time $SRC/volume-cartographer/build/bin/vc_fiberlets fiberlet-replay \
  $FIBER \
  $VES/data/fibers/david/Paris4_fibers/dj_20260*003.json \
  ./fiberlet-replay-full \
  --normal-manifest $NORMALS \
  --length 5000
```

- Anchor extraction: 23.476 seconds wall, 722.314 seconds CPU, 30.8
  effective cores.
- Fiberlet extraction: 43.749 seconds wall, 1254.625 seconds CPU, 28.7
  effective cores.
- Fiberlet preparation: 36.082 seconds wall, 82.5% of fiberlet extraction.
  Node enumeration accounts for 1147.129 summed worker-seconds from 56,523,776
  lattice positions, 239,866,482 local corridor segment tests, and 38,599,833
  replay-tube predicate calls. It retains 35,839,395 nodes.
- Candidate generation is a separate 3.666-second serial phase and also invokes
  the replay-tube predicate 260,944 times.
- Fiberlet prediction plus normal sampling is only 0.058 seconds wall. Corner
  collection is 6.581 summed worker-seconds despite 287,171,520 insertion
  attempts, and 170,493 globally unique voxels. Materialization and search are
  1.804 and 1.376 seconds wall respectively.
- `FiberReplayTube::containsPredictionPoint()` currently reaches the linear
  `projectPointToPolylineArc()` scan for every accepted local-corridor node.
  This repeated exact global-tube scan is the first optimization target. The
  local corridor itself averages 4.24 segment tests per lattice position and
  can test one adjacent layer segment with the exact existing predicate before
  scanning the remainder.

Theoretical removal of all fiberlet preparation would improve the measured
67.225-second extraction total by 2.16x. This is only an upper bound; accepted
nodes, interpolation, and search remain required.

## Deferred Anchor Fitting Proposal

Anchor work is intentionally out of scope for the tube-index milestone, but the
profile establishes the next target:

- 9,221 of 13,027 anchor work cells are NMS context cells (70.8%).
- Fitting consumes 595.692 of 727.091 summed measured worker-seconds (81.9%),
  versus 71.400 for prediction sampling and 60.000 for observation
  construction.
- The fitter receives 429,474,136 retained observations from 1,267,996,072
  candidates and records 275,921 accepted refinement iterations.

Proposed follow-up after tube containment:

1. Profile `fitFiberCellAnchors()` below its current aggregate boundary,
   separating seed evaluation, assignment, component update/PCA, objective
   evaluation, and peak refinement without clocks inside observation loops.
2. Fit selected cells first, then use actual selected-anchor locations plus
   conservative cell/support bounds and a complete suppressor dependency closure
   to identify only context cells that provably cannot affect unchanged NMS.
   Preserve all suppressor and transitive-ranking semantics; final selected
   locations alone are insufficient and context independence must not be
   assumed.
3. For the remaining fitter scans, evaluate reusable buffers, structure-of-
   arrays observations, invariant-weight hoisting, and vectorization while
   preserving accumulation order and exact outputs.
4. Treat shared gradient stencils and direct integer sampling as secondary:
   together their measured stages are much smaller than fitting.

## Decisions

- Milestone 1 changes diagnostics only. Sampling, fitting, candidate generation,
  DP math, ordering, and artifacts remain unchanged.
- Use coarse phase clocks and per-worker accumulators rather than clocks inside
  voxel or transition loops.
- Make the profile visible in the real replay path, not only the benchmark.
- Optimize replay-tube containment first. Keep anchor changes deferred so
  before/after attribution remains clear.
- Preserve existing linear tube methods for compatibility and exact anchor
  diagnostic distances; accelerate extraction with an explicit immutable
  boolean-query snapshot used tens of millions of times.
- Extract shared point-to-segment geometry rather than copying the private
  fiberlet implementation into the replay index.
- After plan review, the user explicitly relaxed exact numeric identity:
  float32 is sufficiently accurate and the intended boolean predicate is the
  unordered union of continuous segment capsules. Reuse a packed Boost.Geometry
  R-tree over expanded segment AABBs, return on the first passing float32
  segment test, and measure legacy differences near the radius boundary.
- A dense distance transform remains a measured follow-up only. The repository
  already provides a portable 3-D float EDT in `libs/edt/edt.hpp`, but its
  rasterization error and crop-sized memory are unnecessary until the segment
  R-tree is measured.

## Plan Review

- Independent review found that the current projection is not an unordered
  minimum: its epsilon-aware reducer and source order can let an earlier segment
  just above the threshold suppress a later segment just below it. The index
  must therefore traverse exact candidates in source order and preserve the
  reducer, not return true for any independently passing leaf.
- `referenceIntervalBase` may simplify vertices near clipped boundaries. The
  index must use exact clipped records produced from original source segments;
  the interval polyline remains display/crop geometry only.
- A direct center-distance local shortcut is mathematically sufficient but can
  round differently from the existing segment projection. The fast path will
  run the same segment predicate against an adjacent layer segment before the
  complete fallback scan.
- Embedding an index beside mutable public tube fields could produce stale
  state. The hot extraction path will explicitly construct one immutable owning
  query snapshot, leaving existing tube queries linear and compatible.
- Formal timings use a dedicated `RelWithDebInfo` tree and fresh output
  directories. Exact tests include epsilon-straddling self-near segments,
  clipping near sharp vertices, lifetime/copy/move behavior, exceptions, and
  concurrent read-only queries.

## Validation

### RelWithDebInfo Baseline

- Dedicated build:
  `cmake -S volume-cartographer -B volume-cartographer/build/fiberlet-perf -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DVC_USE_SCCACHE=OFF -DVC_TESTING=ON -DVC_BUILD_APPS=ON -DVC_BUILD_FLATBOI=OFF -DVC_BUILD_PYTHON=OFF`
  with GCC 16.1.1 on 32-core x86-64 Linux at commit `ec835ee67` plus the
  uncommitted profiling/planning patch.
- Canonical absolute inputs resolved from the preceding replay artifact:
  `fiber_s1_002.lasagna.json`,
  `dj_20260805T025256484_000003.json`, and `las_008.lasagna.json` under the
  user's Vesuvius data root. Each run used `--length 5000` and a fresh `/tmp`
  output directory.
- Three warmed wall times were 58.45, 58.61, and 58.63 seconds
  (min/median/max 58.45/58.61/58.63). Peak RSS was 1.75-1.76 GiB.
- Representative run: anchors 20.360 seconds, fiberlets 37.856 seconds,
  candidate generation 3.420 seconds, preparation 30.820 seconds, and summed
  node-enumeration work 978.561 worker-seconds. Workload counters match the
  supplied profile: 56,523,776 lattice positions, 239,866,482 corridor segment
  tests, 38,599,833 tube predicate calls, and 35,839,395 retained nodes.
- All three complete output inventories produced the same aggregate SHA-256:
  `48eac30b92ce088aace367b19a03e8bbf82d6de4ac343ecb074d1efba4aebfb8`.

### Float32 Boost Segment Index Result

- Three warmed optimized wall times were 25.20, 26.61, and 25.32 seconds
  (min/median/max 25.20/25.32/26.61), a 2.31x median end-to-end speedup over
  the 58.61-second baseline median.
- Representative optimized fiberlet extraction was 4.19 seconds versus
  37.86 seconds, and preparation was 0.570 seconds versus 30.820 seconds.
  Candidate generation fell from 3.420 to 0.058 seconds. Summed node-enumeration
  work fell from 978.561 to 10.632 worker-seconds.
- Adjacent-segment-first corridor evaluation reduced actual segment tests from
  239,866,482 to 166,401,328. Float32 boundary classification retained 243
  additional search nodes out of about 35.8 million and reduced the unique
  sampled-voxel count by 16; successful-path count and replay failures remained
  unchanged.
- Despite those internal boundary-node differences, all three optimized output
  inventories and all three baseline inventories have the same aggregate
  SHA-256:
  `48eac30b92ce088aace367b19a03e8bbf82d6de4ac343ecb074d1efba4aebfb8`.
- The existing 3-D EDT was not implemented because the segment R-tree removed
  the measured bottleneck without dense crop memory or rasterization error.
- A final run after preserving double-precision lattice sizing completed in
  25.23 seconds with the same optimized counters and the same complete artifact
  inventory hash.

- Reconfigured the existing `volume-cartographer/build` tree after switching
  branches, then successfully built `vc_fiberlets` with 32 jobs before edits.
- Rebuilt `vc_fiberlets` with 32 jobs after adding profiling instrumentation.
- Rebuilt the user-facing QuickBuild binary at
  `volume-cartographer/build/bin/vc_fiberlets` after the optimization.
- The profile formatter is shared by benchmark and replay and emits schema
  version 1 with identical field names and units.
- Configured the CI-equivalent QuickBuild test tree without unavailable
  `sccache`, built `test_fiber_anchors`, `test_fiberlet_paths`, and
  `test_fiber_replay` with 32 jobs, and passed all three focused tests.
- Validation commands:
  `cmake -S volume-cartographer -B volume-cartographer/build/ci-fast-core -G Ninja -DCMAKE_BUILD_TYPE=QuickBuild -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_Fortran_COMPILER=flang -DCMAKE_AUTOGEN_PARALLEL=4 -DVC_QUICKBUILD_OPT_LEVEL=0 -DVC_USE_SCCACHE=OFF -DVC_TESTING=ON -DVC_BUILD_APPS=OFF -DVC_BUILD_UI_TRACER=ON -DVC_BUILD_FLATBOI=OFF`,
  `cmake --build volume-cartographer/build/ci-fast-core --target test_fiber_anchors test_fiberlet_paths test_fiber_replay --parallel 32`, and
  `ctest --test-dir volume-cartographer/build/ci-fast-core --output-on-failure -R '^(test_fiber_anchors|test_fiberlet_paths|test_fiber_replay)$' --parallel 3`.
- Built and passed `test_fiber_anchors`, `test_fiberlet_paths`,
  `test_fiber_replay`, and `test_fiber_trace3d` in both the dedicated
  `RelWithDebInfo` tree and the CI-style QuickBuild tree after implementation.
- The representative replay command is now recorded above. A local warmed
  baseline in the dedicated profiling build remains pending implementation work.
