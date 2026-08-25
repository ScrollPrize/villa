# Task log: classify crop traces by principal fiber direction

## Findings

- Crop traces are already complete base-coordinate polylines in
  `FiberletCropTraceResult::lines`; classification can run after tracing and
  cannot affect the trace search or coverage state.
- The existing shared `writePolylinesObj` writer preserves individual line
  objects and supports empty outputs, so the complete and classified artifacts
  can use one implementation.
- Consecutive trace points provide the requested small local steps. Weighting
  normalized axial samples by step length makes the PCA and per-fiber vote
  independent of point sampling density.
- Independent review found that using the first two ordinary PCA eigenvectors
  directly would force orthogonality and is undefined for equally supported
  orthogonal modes. The corrected plan uses PCA only to seed the existing
  non-orthogonal axial two-line objective.
- Bidirectional output polylines do not begin at the seed anchor. Seed-point
  artifacts therefore require storing the anchor's base-coordinate position
  explicitly in each accepted trace result rather than inferring an endpoint.

## Deviations

- None.

## Validation

- GCC Release build:
  `cmake --build volume-cartographer/build --target
  test_fiberlet_crop_trace vc_fiber_trace_chunk test_fiber_anchors -j 32`
- GCC Release tests:
  `volume-cartographer/build/bin/test_fiberlet_crop_trace` passed 8 cases;
  `volume-cartographer/build/bin/test_fiber_anchors` passed 86 cases.
- Clang Debug portability build:
  `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target
  test_fiberlet_crop_trace vc_fiber_trace_chunk test_fiber_anchors -j 32`
- Clang Debug tests:
  `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_fiberlet_crop_trace`
  passed 8 cases;
  `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_fiber_anchors`
  passed 86 cases.
- `git diff --check` passed.

## Follow-up: irrelevant partial halo tuples

- The reported crop failed on combined tuple `15,41,18`, which contained
  anchors and prefixes but no routes. Seven broadly loaded halo fiberlets
  referenced the tuple, but none was incident to an in-crop anchor.
- Replay materialization previously collected every endpoint from the prefix
  halo and filtered non-incident fiberlets only after loading those endpoints.
  It now applies that same incidence predicate before endpoint collection.
- Required partial tuples remain strict errors. The stored-dataset regression
  covers both the ignored irrelevant tuple and the directly required failure.
- Independent plan review approved the early exact-key incidence filter and
  retaining the later endpoint/geometric validation.
- GCC Release: `test_fiberlet_storage` passed 37 cases five consecutive times;
  `test_fiberlet_crop_trace` passed 8 cases.
- Clang build succeeded for `test_fiberlet_storage` and
  `vc_fiber_trace_chunk`; the Clang storage test passed 37 cases.
- The reported crop completed 500 attempts with 500 accepted traces and 27,715
  covered anchors in 86.80 seconds wall time without a partial-tuple failure.
- One initial full storage-test invocation transiently failed an unrelated
  existing decoded-LRU eviction/reload assertion. The focused test and the next
  six full invocations passed; no production or test change was made for it.
- No implementation deviation.
