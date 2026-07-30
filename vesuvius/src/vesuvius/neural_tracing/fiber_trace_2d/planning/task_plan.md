# Native C++ Trace2CP Parallel Runtime Plan

## Implementation

1. Establish a real baseline:
   - rebuild `vc_fiber_trace_metric`
   - run the provided remote manifest + local fiber workload
   - capture wall time, tracer progress summary, restart count, and thread
     behavior
2. Add low-overhead profiling/logging:
   - keep normal output usable by default
   - add a debug/profile flag if detailed stage timings are needed
   - measure candidate generation, prediction sampling/materialization,
     normal sampling, candidate scoring, frontier rebuild, pruning, and segment
     overhead separately
3. Optimize only measured bottlenecks:
   - improve batching and parallelism where the profiler shows serial hot
     work
   - avoid changing candidate math, candidate order, beam pruning, reached
     selection, or scale handling
   - prefer reusable buffers, fewer allocations, and coarser OpenMP work over
     fine-grained cache contention
4. Rebuild and rerun after each meaningful change:
   - log successful and failed attempts in `planning/task_log.md`
   - keep the best measured result and any remaining bottlenecks visible

## Spec Update

- Keep the existing spec requirement that native precomputed Trace2CP preserves
  deterministic beam semantics.
- Add that native tracing performance work must be measured on the remote
  manifest workload and must report stage timing when optimizing parallelism.

## Docs Updates

- Planning docs are sufficient unless a new persistent CLI option is added. If
  a profiling option is added, document it in CLI help and planning specs.

## Tests

- Build after every code change:
  - `cmake --build volume-cartographer/build --target test_fiber_trace3d vc_fiber_trace_metric -j 4`
- Run focused tests:
  - `volume-cartographer/build/bin/test_fiber_trace3d`
  - `ctest --test-dir volume-cartographer/build -R test_fiber_trace3d --output-on-failure`
- Run the representative workload for performance:
  - `volume-cartographer/build/bin/vc_fiber_trace_metric s3://philodemos/hendrik/fiber_vols/fiber_s1_001.lasagna.json /home/hendrik/business/aiconsulting/vesuviuschallenge/data/train_fibers/fibers_test_paul_4/kb_20260605T150824406_000001.json --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --remote-cache-dir /home/hendrik/business/aiconsulting/vesuviuschallenge/vesuvius_fiber_trace_zarr_cache`
