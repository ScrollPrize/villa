# Whole-Batch Loader Parallelization Plan

## Implementation

- Route `run_benchmark(..., load_only=True)` through `_TrainingBatchPipeline`
  when `training.pipeline_enabled` is true.
- Keep `_TrainingBatchPipeline` consumption ordered by step number and preserve
  `_load_training_batch` as the only batch-construction entrypoint.
- Keep normal pipeline workers on the shared base loader/sampler path by
  default; set `training.pipeline_isolated_loaders=false`.
- Keep isolated worker-local VC3D samplers available as an opt-in path, but
  build them from shared parsed records and shared deterministic order caches
  instead of rediscovering fibers/manifests.
- Add optional `volume_cache_memory_mib` and `volume_io_threads` wiring to the
  VC3D sampler factory. `volume_cache_memory_mib: null` leaves VC3D's default
  cache behavior intact; positive values cap each VC3D sampler cache.
- Reduce deterministic batch lookahead from `batch_size * 100` to a bounded
  `max(batch_size * 4, batch_size + 1000)` window to avoid excessive random
  order/cache bookkeeping.
- Update `loader_example.json` to the measured best current shape:
  `loader_workers=4`, `pipeline_depth=16`, `pipeline_workers=4`,
  `pipeline_isolated_loaders=false`, `volume_cache_memory_mib=null`.
- Ensure profile rows keep reporting `pipeline_wait`, process CPU factor, and
  loader worker factor.

## Spec Update

- Clarify that `--load-only` uses the same bounded whole-batch queue as CUDA
  training when `training.pipeline_enabled` is true.
- Clarify the intended defaults: queue depth 16 and 8 whole-batch loader
  workers.
- Document `training.pipeline_isolated_loaders`, `volume_cache_memory_mib`, and
  `volume_io_threads`.
- Document that the tuned example config uses 4 whole-batch workers plus 4
  CP-prep workers because that measured faster than 8 independent whole-batch
  workers or isolated samplers on the current Staticsheep workload.

## Docs Updates

- Update `planning/local_development.md` if the benchmark command or expected
  interpretation changes.
- Update `docs/code_structure.md` if the profile-column meaning changes.
- Keep `planning/task_log.md` scoped to this task.

## Testing

- Compile-check `train.py`.
- Run the focused loader tests.
- Run the established load-only profile benchmark command and compare
  throughput/process CPU factor against the previous baseline.
- Record tested worker-shape regressions in `planning/task_log.md`.
