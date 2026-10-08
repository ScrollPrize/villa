# Unsplit local-event AFV benchmark

User task: run cleaner local-event constraint extraction without splitting
fibers; estimate whole-volume extraction cost from spatial subsets of Paris4 AFV.

## Implementation

- Add localKeepWholeFibers / --local-constraint-events-unsplit to shared native
  extraction. Do not reuse preserveInputLinesAsPieces: that also relaxes endpoint
  certification. Keep all observations, support and normalization unchanged.
- Add a small native AFV benchmark entrypoint to the existing approved
  vc_fiber_trace_chunk tool. Use FiberCollection spatial queries to select source
  IDs, then load full original polylines with fiberBlocks (no geometric clipping).
  Use the existing LasagnaNormalSampler without a normal-alignment field.
- User correction: no sign-alignment BP, dense lattice, tracing or inference.
  Select a thin full-XY Z slab, retaining whole curves. Integrate unsigned winding
  distances only along required connectors; keep two-sided unsigned certification.
- Report query/load and extraction phases (prepare/search/orientation/
  winding), selected fibers/arc/sample points/candidate counts/accepted events.
  No final winding BP solve, reference fibers or optimizer in this benchmark.

## Benchmark

Use the existing Paris4 Lasagna sheet-normal manifest las-sd2-5b17ff6c, native
L0 coordinates, exact same default local-event parameters as shared extraction.
Select deterministic spatial subsets; begin small then grow if practical.
Report initial and repeated warm timings, CPU threads, build, RSS and extrapolate
using sampled arc/candidate density rather than only fiber count. A spatial
subset omits neighbors not selected by its box: explicitly report that boundary
bias, and do not claim a precise global ETA. No whole-volume run in this task.
Initial extraction has no normal-alignment warmup; OS/source cache may still
be warm from previous work. Compare a thicker slab if feasible to check scaling.

## Tests / docs / spec update

Regression: identical local observations/weights/support before and after
disabling cuts; one piece per nondegenerate input, no continuation, no endpoint
relaxation. Existing native tests and benchmark smoke via approved commands.
Update task_log/status and benchmark report with commands, parameters, results,
timing uncertainty and limitations. No change to existing default mode.

## Dependency resolution

User installed SQLite development headers/library. Native rebuild and regression
suite now pass. No dependency installation was performed by the agent.
