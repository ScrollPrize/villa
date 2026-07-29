# Native C++ Trace2CP Parallel Candidate Scoring Plan

## Implementation

1. Add explicit prediction-source concurrency support:
   - keep generic `FiberPredictionSource` serial by default
   - mark persisted `FiberPredictionField` as concurrent-safe
   - add batched prediction sampling for persisted presence/nx/ny channels
2. Add parallel candidate scoring:
   - build candidate tasks in original beam/candidate order
   - cache the per-trace cone offset table instead of rebuilding it per beam
   - keep internal beam paths as parent-linked nodes so candidate expansion
     does not copy the full traced point vector for every candidate
   - prepare all candidate interpolation requests up front
   - prefetch/decode the involved prediction chunks once per generation
   - materialize prediction samples as a batch before scoring
   - sample Lasagna normals in batch when normal-aware smoothness is active
   - score independent candidates with OpenMP from materialized samples when
     prediction and normal samplers both advertise concurrent sampling
   - rebuild `nextFrontier` serially in original task order to preserve
     deterministic beam pruning/reached semantics
3. Add control surface:
   - add `FiberTraceConfig::parallelThreads`
   - expose `vc_fiber_trace_metric --threads`, default `0` for the OpenMP
     default thread count
   - allow `--threads 1` to force serial candidate scoring

## Spec Update

- Document that native Trace2CP may score candidates in parallel only through
  concurrent-safe samplers, must keep Zarr/cache access in chunky batched
  materialization, and must preserve deterministic task-order output.

## Docs Updates

- No separate user docs are needed beyond CLI help and planning notes.

## Tests

- Extend `test_fiber_trace3d` for the new config default and worker gating.
- Add a regression proving the parallel trace path uses batch prediction
  sampling rather than per-candidate `sample()` calls.
- Build and run:
  - `cmake --build volume-cartographer/build --target test_fiber_trace3d vc_fiber_trace_metric -j 4`
  - `volume-cartographer/build/bin/test_fiber_trace3d`
  - `ctest --test-dir volume-cartographer/build -R test_fiber_trace3d --output-on-failure`
  - `volume-cartographer/build/bin/vc_fiber_trace_metric --help`
