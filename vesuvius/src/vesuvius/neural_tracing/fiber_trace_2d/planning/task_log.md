# Whole-Batch Loader Parallelization Log

Current task: make load-only benchmark exercise true parallel batch loading and
measure whether this improves the fiber-strip loader bottleneck.

Implementation notes:

- Routed `run_benchmark(..., load_only=True)` through `_TrainingBatchPipeline`
  when `training.pipeline_enabled` is true.
- Kept batches consumed strictly by step number, so deterministic sample order
  is unchanged even when futures complete out of order.
- Added `training.pipeline_isolated_loaders`, default `false`.
- Added `_PipelineLoaderProvider` support for optional worker-local loader
  clones. Clones reuse parsed records, Lasagna/zarr handles, sample identity
  keys, and deterministic random-order cache, but create fresh VC3D samplers.
- Added optional `volume_cache_memory_mib` and `volume_io_threads` wiring to
  VC3D sampler construction. `null` leaves VC3D's default cache behavior
  intact; positive values cap each sampler.
- Reduced `load_batch` deterministic lookahead from `batch_size * 100` to
  `max(batch_size * 4, batch_size + 1000)`.
- Updated `loader_example.json` to the best measured current shape:
  `loader_workers=4`, `training.pipeline_workers=4`,
  `training.pipeline_depth=16`, `training.pipeline_isolated_loaders=false`,
  `volume_cache_memory_mib=null`.

Validation commands:

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/sampling.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --benchmark --load-only --profile`

Results:

- Compile check passed.
- Focused loader tests passed: `108 passed`.
- Previous best recorded in this task context: `166.91 patches/s` for
  `loader_workers=1`, `pipeline_depth=16`, `pipeline_workers=8`.
- Isolated loader clones with per-sampler `volume_cache_memory_mib=512` were
  worse: `126.42 patches/s`. Clone startup was fixed, but duplicated VC3D
  cache state and the small cache cap hurt throughput.
- Shared loader/sampler with `loader_workers=1`, `pipeline_depth=16`,
  `pipeline_workers=8`, and `volume_cache_memory_mib=null` improved to
  `185.48 patches/s`.
- `pipeline_workers=16`, `pipeline_depth=32` was stopped early because the
  profile showed much higher coord/line costs and contention.
- Balanced nested parallelism with `loader_workers=4`,
  `pipeline_workers=4`, `pipeline_depth=8` measured `184.27 patches/s`.
- Balanced nested parallelism with `loader_workers=4`,
  `pipeline_workers=4`, `pipeline_depth=16` measured best:
  `197.48 patches/s`, `process_cpu_factor=3.407`,
  `loader_thread_factor=0.860`.
- Raising inner `loader_workers` to 8 was stopped early because coord/cache/line
  worker time ballooned and was slower than the 4-worker setting.

Residual bottleneck:

- The measured best is a real improvement but not the requested 4x. The warm
  profile is still limited by VC3D volume sampling variability and strip-cache
  reads, not by lack of queue depth. Additional worker count alone regressed.
