# Native C++ Trace2CP Parallel Candidate Scoring Status

- [x] Re-check native tracer loop and sampler thread-safety hooks.
- [x] Replace stale task and task plan with the parallel scoring task.
- [x] Add prediction-source concurrent sampling capability.
- [x] Add persisted prediction batch materialization.
- [x] Parallelize candidate scoring while preserving output order.
- [x] Remove per-candidate path vector copying from beam expansion.
- [x] Cache per-trace cone offsets and append candidates without per-beam
      direction-vector allocations.
- [x] Add `--threads` control to `vc_fiber_trace_metric`.
- [x] Add focused worker-gating/config tests.
- [x] Update specs/task log/changelog.
- [x] Build focused native targets.
- [x] Run focused native tests.
