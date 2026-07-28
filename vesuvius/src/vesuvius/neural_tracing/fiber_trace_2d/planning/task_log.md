# Task Log: Native 3D Trace2CP Hot-Path Acceleration

## Implemented

- Added `Volume.sample_zyx_block(...)` to the VC3D Python binding for strict
  requested-level axis-aligned ZYX block reads through the VC3D chunk cache.
- Added `CoordinateSampler.sample_block_zyx(...)` and implemented it for the
  production VC3D sampler and local NumPy test sampler.
- Switched native 3D Trace2CP inference-block loading from dense
  `[D,H,W,3]` coordinate grids plus `sample_coord_batch(...)` to direct
  selected-level block reads.
- Added missing-block materialization batching in `NativeTraceFieldCache` and
  batched model forwards controlled by `--inference-block-batch-size`
  (default `2`).
- Updated native 3D Trace2CP defaults to the trained/in-use path:
  `--inference-patch-shape-zyx 128 128 128` and `--core-margin-voxels 48`.
- Updated `FiberTrace3DPredictAdapter.preprocess_tile(...)` to accept batched
  tiles while preserving the existing per-tile normalization semantics.
- Updated tests for the direct block sampler, native field-cache block routing,
  and new native 3D Trace2CP defaults.
- Rebuilt and reinstalled the local editable VC3D Python binding so the active
  user-site `vc.volume.Volume` exposes `sample_zyx_block`.

## Audit Notes

- Candidate scoring was already torch-vectorized for flattened candidate sets;
  this task kept that math unchanged.
- Lasagna normals are already sampled once per flattened candidate batch in the
  current native tracer path. I did not add a spatial normal cache in this task
  because that would change cache behavior without a fresh profile showing it
  dominates after block-read removal.
- Point lookup still groups by inferred block and transfers each resident CPU
  block needed by the lookup call to CUDA. The change now materializes misses
  before that loop, but it does not add a packed global GPU texture/cache.

## Deviations / Deferred Items

- No fiber-only zarr/raw reader was added; all production reads stay behind the
  shared VC3D-backed sampler boundary.
- No scoring, metric, restart, fusion, normalization, or checkpoint-selection
  semantics were changed.
- I did not run a full native Trace2CP metric benchmark in this pass. The code
  and targeted tests are validated; a full before/after wall-time comparison
  still needs the exact long-running metric command/dataset run.

## Validation

- `cmake --build volume-cartographer/build/python-bindings --target vc_volume -j 8`
  passed.
- `python -m pip install -e volume-cartographer --no-deps --break-system-packages`
  passed.
- `python -c "import vc.volume; print(vc.volume.__file__); print(hasattr(vc.volume.Volume, 'sample_zyx_block'))"`
  printed the user-site VC3D extension path and `True`.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/inference_adapter.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/sampling.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d or whole_fiber_trace"`
  passed: `60 passed, 89 deselected`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "vc3d_coordinate_sampler_direct_block_read_uses_binding or numpy_coordinate_sampler_direct_block_read_masks_out_of_bounds"`
  passed: `2 passed, 147 deselected`.
- `git diff --check` passed.
