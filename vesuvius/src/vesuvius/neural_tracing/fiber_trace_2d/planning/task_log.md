# Task Log: 3D Fiber Loader Performance Rewrite

## Baseline

Command:

`PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_3d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all.json --benchmark --load-only --benchmark-batches 10`

Result, current zarr-crop + torch-resample path, one 192^3 patch per batch:

- load_ms values: 9732.84, 3465.15, 12461.35, 9384.66, 8682.22, 17110.43, 6297.84, 10574.39, 5098.02, 7889.71
- mean: 9069.66 ms
- median: 9033.44 ms
- min/max: 3465.15 / 17110.43 ms

## Implementation Notes

- Added a VC3D coordinate sampler to each 3D loader record. Real datasets use
  `Vc3dCoordinateSampler`; array-backed tests use `NumpyZarrCoordinateSampler`.
- Changed normal 3D training samples to build `backward_source_zyx` coordinates
  and sample the final 3D patch directly through `CoordinateSampler.sample_coord_batch`.
  The old oversized zarr crop plus torch `grid_sample` path is no longer used
  by `FiberTrace3DLoader.load_sample`.
- Removed dense forward-map construction from the hot path. Fiber line points
  are mapped to output-patch coordinates analytically from the same augmentation
  parameters.
- Replaced dense NML full-grid nearest-segment target generation with direct
  bounded rasterization of transformed output-space segment capsules.
- Updated 3D prefetch to collect VC3D coordinate dependencies from the same
  explicit coordinate path as training and to use the shared Python prefetch
  downloader/cache writer with atomic renames and `.empty` marker handling.
- Added an ordered training `load_batch` queue controlled by
  `training.pipeline_enabled`, `training.pipeline_depth`, and
  `training.pipeline_workers`.
- Updated the checked-in S1A NML 3D config to enable the queue with depth 2 and
  two worker slots.
- Updated specs, code-structure docs, local-development benchmark notes, status,
  and changelog.

## After-Change Benchmark

Same command as baseline:

`PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_3d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all.json --benchmark --load-only --benchmark-batches 10`

Result, VC3D coordinate-sampling path, one 192^3 patch per batch:

- load_ms values: 3790.22, 1355.94, 1369.31, 1356.73, 1364.38, 1363.39, 1355.16, 1354.55, 1363.38, 1355.27
- mean: 1592.83 ms
- median: 1359.06 ms
- min/max: 1354.55 / 3790.22 ms
- Excluding the first warmup batch: mean 1359.79 ms, median 1356.73 ms

## Validation

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

Result: `17 passed in 2.59s`.

## Deviations / Simplifications / Deferred Items

- The process-per-loader multiprocessing pool from the plan is not implemented
  in this pass. Instead, training gets an ordered `ThreadPoolExecutor` queue
  around complete `load_batch` calls. This preserves deterministic ordering and
  can overlap VC3D/CPU work with the model step, but it is not the fully
  independent process/CUDA-context design requested in the broader plan.
- The after-change measurement is the approved load-only benchmark command. I
  did not run an additional full forward/backward benchmark with a different
  command shape.
