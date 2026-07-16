# Native 3D Trace2CP Tool Task Log

## Planning Notes

- Inspected the current 2D Trace2CP direction and combined
  direction/presence scorer in `fiber_trace_2d.runner`.
- Inspected the current 3D Trace2CP bridge in `fiber_trace_3d.trace2cp_bridge`
  and `fiber_trace_3d.train`.
- Confirmed that the current 3D path projects 3D model outputs onto a 2D
  Trace2CP strip and reuses the 2D scorer; it is not a native 3D cone tracer.
- Replaced `task.md`, `task_plan.md`, and `status.md` with a planning-only
  task for a separate native 3D Trace2CP tool.

## Deviations Or Deferrals

- No implementation was done in this planning step.
- The native 3D metric is planned as tool-local debug output first and should
  not replace the existing projected Trace2CP metric or best-checkpoint
  selection in the first implementation.

## Validation

- Planning-only; no code tests were run.
