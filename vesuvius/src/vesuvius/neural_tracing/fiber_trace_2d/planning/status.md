# Native 3D Trace2CP Tool Status

- [x] Read local `fiber_trace_2d/AGENTS.md` workflow.
- [x] Inspect current 2D Trace2CP direction/presence scorer.
- [x] Inspect current 3D projected Trace2CP bridge.
- [x] Replace `planning/task.md` with the current planning task.
- [x] Replace `planning/task_plan.md` with the native 3D implementation plan.
- [x] Record that this is planning-only and implementation is pending.
- [x] Update specs during implementation.
- [x] Update docs during implementation.
- [x] Implement separate native 3D Trace2CP tool.
- [x] Add focused tests.
- [x] Run validation commands.

## Implemented Scope

- [x] Added `fiber_trace_3d.trace2cp_tool` native CLI.
- [x] Added overlapped trusted-core inference block cache/router.
- [x] Added deterministic 3D cone candidate stepping with direction/presence
  scoring.
- [x] Added target-plane crossing stop condition.
- [x] Added side/top Trace2CP strip visualization built directly from the
  traced native 3D polyline.
- [x] Added native fw/bw trace progress bars.
- [x] Changed native inference patch default to 64 voxels per axis.
- [x] Added synthetic tests for cone generation, plane crossing, block routing,
  and constant-field native tracing.
