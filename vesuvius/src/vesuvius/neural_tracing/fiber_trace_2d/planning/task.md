# Task: Native 3D Trace2CP Accelerated Normal Debug Run

We want to run native 3D Trace2CP with the accelerated sparse corner/tensor
Lasagna normal sampler while still comparing it against the baseline normal
sampler at the same candidate points.

Requirements:

- The normal comparison path must remain opt-in from the CLI.
- When the comparison path is enabled, tracing must use the accelerated
  sparse corner/tensor normal outputs after they pass comparison.
- The comparison must fail fast on significant differences instead of recording
  long traces.
- Failures must identify the compared path, call number, point coordinate,
  baseline normal/valid state, accelerated normal/valid state, and angular
  difference when available.
- Valid-mask mismatches must fail immediately.
- Angular differences must fail when they exceed a configurable threshold.
- Do not reintroduce any path that interpolates raw compact `nx`/`ny` and then
  reads `normal_3d`.
