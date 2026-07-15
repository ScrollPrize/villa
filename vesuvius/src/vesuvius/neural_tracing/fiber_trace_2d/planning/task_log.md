# Task Log: 3D Fiber GPU-Side Target Materialization

## Implementation

- Added compact 3D target metadata to `FiberTrace3DSample` and
  `FiberTrace3DBatch`.
- Changed `FiberTrace3DLoader.load_sample(...)` so workers build only compact
  target specs:
  - CP-only samples store transformed local tangent metadata;
  - NML samples store transformed output-space segment endpoints and segment
    bboxes clipped to the radius-expanded patch.
- Added `fiber_trace_3d.targets.materialize_targets(...)` to create dense
  direction/presence targets on the training device using torch and the shared
  Lasagna 3x2 direction helpers.
- Wired target materialization into training, dense test loss evaluation,
  TensorBoard sample visualization, benchmark/load-only, and the 3D tests.
- Updated benchmark columns to report worker compact-target spec timing and
  main-process GPU target timing separately.
- Set S1A 3D configs to `loader_workers: 32`.

## Deviations / Simplifications / Deferred Items

- The first GPU dense-line rasterizer still loops over samples and line
  segments in Python while all bbox voxel distance/update work runs on GPU.
  If this becomes the next bottleneck, the follow-up is grouping/vectorizing
  segment regions without changing label semantics.
- Normal training uses non-synchronizing target submit timing; benchmark
  profiling synchronizes and reports `target_gpu_total_ms`,
  `target_gpu_raster_ms`, `target_gpu_encode_ms`, and `target_gpu_mask_ms`.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: `18 passed in 2.69s`.
- `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_3d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all.json --benchmark --load-only --benchmark-batches 40`
  - Result: completed.
  - Worker overlap: `items=40`, `span_ms=22299.5`, `avg_active=27.49`,
    `max_active=32`, `worker_cpu_x=18.25`, `construct_items=32`.
  - After startup, rows 9-40 were mostly `26-41 ms` total with occasional
    waited rows; `target_ms` was typically `18-24 ms` after warmup.
