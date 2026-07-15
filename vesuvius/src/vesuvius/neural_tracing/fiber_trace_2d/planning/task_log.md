# Task Log: 3D Fiber Sparse Line Supervision Targets

## Implementation Notes

- Added sparse direction target fields to `FiberTrace3DBatch`.
- Replaced the 3D NML target materializer's radius-expanded
  distance-to-segment tube rasterization with vectorized GPU centerline voxel
  drawing from transformed clipped segment endpoints.
- Kept dense `presence_target` and `presence_mask` creation in the main
  process on the training device.
- Switched direction loss to gather predicted six-channel Lasagna 3x2 outputs
  at `direction_indices_bzyx` and compare against sparse encoded targets.
- Updated train-sample visualization to scatter sparse direction angle errors
  only for supervised voxels in the displayed principal planes.
- Updated benchmark column labels from dense raster/encode terminology to
  sparse target stages (`line_idx`, `cp_idx`, `scatter`, `dir_enc`, `linePts`,
  `dirPts`).
- Updated specs, code-structure docs, local benchmark notes, changelog, and
  status.

## Deviations / Simplifications / Deferred Items

- Non-NML CP-only radius-neighborhood semantics remain unchanged as planned.
- The old dense direction dataclass fields remain as compatibility slots, but
  normal materialization leaves them `None` and uses sparse direction fields.
- No custom CUDA/Triton kernel was added.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/targets.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: `18 passed in 2.62s`.
- Benchmark command:
  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_3d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all.json --benchmark --load-only --benchmark-batches 40`
  completed.
- Benchmark dataset/config: `train_s1a_nml_all.json`, `batch_size=1`,
  `patch_shape_zyx=[192,192,192]`, `loader_workers=32`, `device=cuda`.
- Benchmark rows 33-40 after worker startup:
  - total ms: mean `112.82`, median `12.59`, min `12.09`, max `809.34`;
  - excluding the one row-35 wait outlier: mean `13.32 ms/crop`
    (`75.1 crops/s`);
  - median steady-state throughput: `79.5 crops/s`;
  - target materialization rows 33-40: mean `2.73 ms`, median `2.54 ms`.
