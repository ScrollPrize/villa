# Plan: accelerate fiberlet tube containment

## Baseline

1. Configure a dedicated `RelWithDebInfo` profiling tree, build
   `vc_fiberlets`, and record the commit plus dirty-patch identity, compiler,
   flags, host, hardware thread count, expanded environment variables and glob,
   absolute manifest/fiber paths, and cache state. Keep the user's existing
   build tree untouched and use the same profiling tree before and after.
2. Use the supplied workload as the canonical benchmark:

   ```bash
   time $SRC/volume-cartographer/build/fiberlet-perf/bin/vc_fiberlets \
     fiberlet-replay \
     $FIBER \
     $VES/data/fibers/david/Paris4_fibers/dj_20260*003.json \
     ./fiberlet-replay-full-RUN_ID \
     --normal-manifest $NORMALS \
     --length 5000
   ```

3. Preserve the supplied profile as the initial reference, then run a warmed
   baseline before implementation. Use at least three comparable runs when
   practical and report min/median/max wall time plus the profile's anchor,
   fiberlet, preparation, node-enumeration, and candidate-generation values.
   Give every run a fresh output directory so retained immutable generations
   cannot contaminate timing or hashes. Capture a canonical hash inventory of
   the complete fresh output, identifying separately the authoritative root and
   its referenced immutable generation.

## Float32 Segment Index

1. Add an immutable, copy-safe replay-tube containment snapshot built from the
   authoritative clipped source segments, not `referenceIntervalBase`. Convert
   clipped segment endpoints and the radius into prediction-space float32 once.
2. Reuse Boost.Geometry's packed R-tree implementation already used by
   `PointIndex`, `FiberIntersections`, and `SurfacePatchIndex`. Store each
   segment's radius-expanded float32 AABB plus its float32 endpoints.
3. Query the R-tree directly with each float32 prediction-space point. Iterate
   candidates without collecting or sorting and return on the first segment
   whose continuous point-to-segment squared distance is within the float32
   radius. The intended predicate is an unordered geometric union; the old
   `1e-12` source-order projection tie behavior is not part of this optimized
   boolean-query contract.
4. The snapshot owns all segment and tree storage. It retains no pointers into
   the mutable/copyable `FiberReplayTube`; concurrent const queries perform no
   writes. Keep the existing public double-precision distance/projection methods
   unchanged for anchor diagnostics and compatibility.
5. Do not use `PointIndex`: it indexes points, while this query needs continuous
   segment capsules. Do not add a second custom BVH when the existing Boost
   dependency provides the required unordered broad phase.

## Tube Query Integration

1. Keep all existing `FiberReplayTube` query methods on their current linear
   double-precision projection path. Anchor retention records diagnostic
   distances, and its 5,732 calls are not the measured bottleneck.
2. Add an explicit `FiberReplayTube` containment-query factory that snapshots
   the tube's authoritative geometry, arc range, radius, and prediction scale
   into the immutable owning query. `extractTubeFiberlets()` constructs it once
   and captures it as const for candidate generation and node enumeration. This
   avoids a hidden index that could become stale when public tube fields change.
3. Add a local-corridor fast acceptance before its full segment loop. For a node
   on layer `i`, first evaluate its existing float-rounded coordinate with the
   float32 point-to-segment predicate against one adjacent corridor segment
   having layer `i` as an endpoint. If it accepts, the existing any-segment
   predicate accepts. Otherwise scan the remaining segments without testing the
   adjacent segment twice.
4. Keep candidate generation, node order, interpolation corners, sampling, DP,
   and all scoring unchanged. Land the shared-helper refactor separately from
   the indexed/fast-path behavior where practical.

## Measurement And Iteration

1. Re-run the canonical workload with identical inputs and cache conditions.
   Compare generated/prepared candidates, retained nodes, unique sampled
   voxels, DP counters, anchors, paths, and failures. Record all differences;
   local corridor segment-test counts should decrease.
2. Compare the complete artifact hash inventory. Changed artifacts are allowed
   only when explained by float32 containment decisions within a measured
   boundary tolerance. Any change farther from the radius blocks the change.
3. Rank success primarily by `fiberlet_preparation_seconds` and
   `fiberlet_node_enumeration_work_seconds`, then total extraction and command
   wall time. CPU reduction must accompany wall-time reduction.
4. If the Boost segment index leaves global containment dominant, measure the
   actual crop dimensions and evaluate the existing portable 3-D float EDT in
   `libs/edt/edt.hpp`. Do not introduce rasterization error or dense-field
   memory without reporting it and obtaining another review.
5. Stop and record the result if the optimization does not materially reduce
   preparation time; do not compensate by changing geometry, radii, precision,
   or accepted candidates.

## Tests

1. Add deterministic tests comparing indexed float32 containment with a direct
   float32 segment scan for straight, bent, long, self-near, and clipped
   polylines; fixed-seed random points; boundaries; large coordinates/radii;
   interval endpoints; repeated vertices; and invalid construction/query input.
   Compare the legacy double predicate as a diagnostic and require agreement
   away from a documented float32 boundary band.
2. Exercise one immutable index concurrently and verify every result against
   the linear predicate. Cover copy, move, assignment, destruction of the source
   tube, and independent snapshots before running under existing sanitizer/CI
   coverage without query-time writes.
3. Add local-corridor parity tests proving the adjacent-segment fast path and
   fallback equal a complete float32 segment scan, including points near the
   radius and both endpoint layers.
4. Extend the serial/parallel fiberlet test to require unchanged candidates,
   node populations, scores, paths, and deterministic counters. Keep existing
   replay and anchor tests.
5. Build `vc_fiberlets`, `test_fiberlet_paths`, `test_fiber_replay`, and
   `test_fiber_anchors` with 32 jobs; run the focused CTest set and
   `git diff --check`.
6. Run the canonical before/after replay benchmark and artifact comparison
   after focused tests pass. Attribute any output difference to a measured
   float32 boundary classification.

## Spec Update

- Specify float32 continuous-segment containment as an unordered geometric union
  for the hot replay filtering path, with measured boundary tolerance.
- Specify Boost segment broad-phase reuse, immutable owning query
  lifetime/thread-safety, and unchanged public linear diagnostic methods.
- Specify that local-corridor interior acceptance is permitted only when the
  accepted point is proven to lie within the same existing corridor.
- Retain deterministic-output and measurement-first requirements. Exact numeric
  identity is intentionally relaxed only for this float32 containment filter.

## Documentation Updates

- Document the indexed tube predicate, linear diagnostic-distance fallback, and
  local-corridor proof in `volume-cartographer/docs/fiberlets.md`.
- Record baseline and after measurements, artifact comparison, review findings,
  deviations, and failed attempts in `task_log.md`.
- Add a changelog entry after the optimization and validation are complete.
