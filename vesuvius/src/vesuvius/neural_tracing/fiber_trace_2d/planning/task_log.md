# Task Log: 3D Fiber PyTorch DataLoader Parallelism

## Planning Notes

- The previous implementation used `_OrderedBatchLoadPipeline`, a
  `ThreadPoolExecutor` around complete `loader.load_batch(...)` calls.
- That is now explicitly treated as the issue to fix. It does not fulfill the
  requested PyTorch/process data-loader parallelism.
- New plan is to use `torch.utils.data.DataLoader` with worker processes.
- DataLoader items should be complete `FiberTrace3DBatch` objects and should be
  passed through without default collation.
- Each DataLoader worker must lazily instantiate its own
  `FiberTrace3DLoader(load_config(config_path))`, opening VC3D sampler state in
  that worker process.
- Deterministic order is preserved by mapping DataLoader item index directly to
  deterministic training batch start sample index.

## Deviations / Simplifications / Deferred Items

- No code implementation has been done for this task yet. This update only
  creates the plan requested by the user.

## Validation

- Not run for this planning-only step.
