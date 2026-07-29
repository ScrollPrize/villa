# Task: Native 3D Trace2CP Point-Lookup Optimization

Continue optimizing native 3D Trace2CP while preserving the current
quality-matching tracing behavior.

Requirements:

- Keep the sparse corner/tensor Lasagna normal sampler and `eigh` principal
  axis reconstruction as the default quality path.
- Reuse the same approved whole-fiber benchmark command; change defaults or
  config rather than introducing new benchmark flags.
- Try the more invasive lookup optimization: avoid repeated small
  `grid_sample`/block-copy work for cached inference field point sampling.
- Test whether larger missing-block inference batches improve speed without
  changing the metric result.
- Measure every optimization with the same benchmark and compare restarts,
  wall/CPU time, and profile stages.
- Document any experimental path that worsens quality or is not kept as the
  default.
