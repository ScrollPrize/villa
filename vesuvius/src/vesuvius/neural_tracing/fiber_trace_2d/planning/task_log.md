# Native C++ Trace2CP Parallel Candidate Scoring Task Log

## Notes

- Whole-fiber metric tracing remains sequential at the segment level because a
  segment's start point/direction depends on the previous segment's traced or
  restarted state.
- Parallelism was added inside each beam generation in the same broad shape as
  the Python tensor path:
  - candidate tasks are built in original beam/candidate order
  - persisted prediction interpolation cube requests are prepared for all
    candidate points
  - involved prediction chunks are prefetched/decode-materialized once per
    generation
  - candidate prediction samples are materialized as a batch before scoring
  - Lasagna candidate normals are sampled in batch when normal-aware
    smoothness is active
  - independent candidate loss evaluation runs through OpenMP from the
    materialized prediction/normal samples when the prediction source and
    normal sampler are concurrent-safe
  - `nextFrontier` is rebuilt serially in the original task order
- Generic `FiberPredictionSource` remains serial by default. Persisted
  `FiberPredictionField` opts into concurrent sampling and implements batched
  sampling for presence/nx/ny chunks.
- `FiberTraceConfig::parallelThreads` controls candidate scoring worker count:
  `0` is the default and uses the OpenMP default, `1` forces serial lazy
  scoring, and positive values greater than one clamp
  to the current candidate count.
- `vc_fiber_trace_metric` exposes this as `--threads`.
- The first parallel attempt was slower because it called `predictions.sample()`
  from inside the OpenMP loop. That caused fine-grained contention on the
  shared sparse chunk cache. The current path removes cache/Zarr access from
  the parallel scoring loop.
- Follow-up measurement showed only about a 2x speedup while consuming several
  cores and underutilizing a 32-thread machine. The remaining reason is that
  chunk materialization/prefetch is still a large serial/bounded stage while
  the fine-grained loss math is relatively small. Prediction prefetch was
  changed from sequential presence/nx/ny prefetch calls to concurrent
  per-channel prefetch, matching the normal sampler's structure.
- A further check found that "batched" still only applied to sampler
  materialization and candidate loss evaluation. Beam expansion still rebuilt
  the same cone offset table for every beam, allocated a candidate-direction
  vector per beam, reserved a hard-coded 81 candidates, and copied the full
  traced path vector whenever creating a child candidate state.
- Beam path storage is now parent-linked internally and only materialized back
  to a vector when returning a public trace result. Candidate generation now
  caches the per-trace cone offsets once and appends candidates directly in the
  same beam/candidate order.
- The duplicate candidate-point rebuild for batched normal sampling was also
  removed; prediction and normal batch materialization now share the same
  candidate point vector.

## Deviations

- No full remote S3 workload benchmark was run in this agent shell because the
  local `$SRC`/`$VES` environment variables used by the user's command are not
  set here and the full workload can be long-running. Validation is focused on
  build/tests/help smoke output.

## Validation

- `cmake --build volume-cartographer/build --target test_fiber_trace3d vc_fiber_trace_metric -j 4`
  passed.
- `volume-cartographer/build/bin/test_fiber_trace3d` passed: 25 test cases.
- `ctest --test-dir volume-cartographer/build -R test_fiber_trace3d --output-on-failure`
  passed.
- `volume-cartographer/build/bin/vc_fiber_trace_metric --help` shows
  `--threads`.
- After concurrent prediction-channel prefetch:
  - `cmake --build volume-cartographer/build --target test_fiber_trace3d vc_fiber_trace_metric -j 4`
    passed.
  - `volume-cartographer/build/bin/test_fiber_trace3d` passed: 25 test cases.
  - `ctest --test-dir volume-cartographer/build -R test_fiber_trace3d --output-on-failure`
    passed.
- After parent-linked beam paths and cached candidate offsets:
  - `cmake --build volume-cartographer/build --target test_fiber_trace3d vc_fiber_trace_metric -j 4`
    passed.
  - `volume-cartographer/build/bin/test_fiber_trace3d` passed: 25 test cases.
  - `ctest --test-dir volume-cartographer/build -R test_fiber_trace3d --output-on-failure`
    passed.
  - `git diff --check` passed.
