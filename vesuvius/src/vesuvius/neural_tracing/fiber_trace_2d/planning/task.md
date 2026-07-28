# Task: Fail-Fast Native 3D Trace2CP Acceleration Comparison

We want to accelerate native 3D Trace2CP again, but the previous sparse/GPU
normal path hurt quality. Add a debug-only parallel comparison path so a run can
execute the baseline normal sampler and the accelerated normal sampler at the
same candidate points, while the tracer still uses the baseline outputs.

Requirements:

- The default tracer behavior must remain the restored baseline path.
- The accelerated comparison path must be opt-in from the CLI.
- The comparison must fail fast on significant differences instead of recording
  long traces.
- Failures must identify the compared path, call number, point coordinate,
  baseline normal/valid state, accelerated normal/valid state, and angular
  difference when available.
- Valid-mask mismatches must fail immediately.
- Angular differences must fail when they exceed a configurable threshold.
- Do not use the comparison path for production scoring until differences are
  understood.
