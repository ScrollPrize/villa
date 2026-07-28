# Task Log: Native 3D Trace2CP GPU Sparse Sampling

## Planning

- Read `fiber_trace_2d/AGENTS.md`.
- Checked current `planning/specs.md` and `planning/plan.md`.
- Inspected existing Lasagna sparse-cache code:
  - `lasagna.fit_data.load_3d_streaming(...)`
  - `FitData3D.grid_sample_fullres(...)`
  - `SparseChunkGroupCache`
  - `TensorStoreSparseChunkGroupCache`
- Confirmed Lasagna already has the right zarr-backed GPU sparse sampling path
  for `grad_mag`, `nx`, and `ny`; the tracer should use that for candidate
  normals instead of the current CPU geometry callback.
- Wrote the task and detailed implementation plan.

## Deviations / Deferred Items

- Full Python-free beam object storage/reconstruction is explicitly deferred in
  the plan because current profiles show it is not the dominant bottleneck.
- The shared sparse inferred-field cache from the plan is not complete in this
  implementation. The existing Lasagna sparse cache/sampler is uint8
  zarr-chunk oriented and uses a dense pointer table over a known volume grid,
  which fits `grad_mag`/`nx`/`ny` normals but does not directly fit live
  float32 checkpoint outputs with overlapping trusted cores. Implementing that
  correctly requires a shared typed float32 sparse sampler/table extension in
  Lasagna, not a tracer-local duplicate table.
- As an intermediate step, native Trace2CP inferred blocks now stay
  device-resident under the existing LRU byte budget and candidate lookup no
  longer copies resident block outputs from CPU back to GPU for each sampled
  block. Point-to-block routing still uses CPU grouping by trusted-core block.

## Implementation

- Added `_NativeSparseLasagnaNormalSampler`, backed by Lasagna streaming
  `FitData3D.grid_sample_fullres(...)`, restricted to `grad_mag`, `nx`, and
  `ny`.
- The sparse normal sampler converts selected-level ZYX trace points to
  base/fullres XYZ tensors, prefetches/syncs the Lasagna sparse caches once per
  batched call, samples via `grid_sample_fullres`, and uses `normal_3d` for the
  established Lasagna ambiguous-normal decode.
- Native Trace2CP chooses the sparse normal sampler for CUDA runs when the
  configured dataset record has a `lasagna_manifest_path`; the existing
  geometry-loader normal sampler remains the fallback for non-CUDA/test paths.
- Updated point-choice lookup helpers so torch-capable caches receive torch
  tensors directly.
- Updated `NativeTraceFieldCache` to store decoded inferred output tensors and
  valid masks on `cache.device` instead of CPU.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "sparse_lasagna_normal_sampler or point_choice_helper_keeps_torch_points or lasagna_normal_sampler_uses_geometry_record"`
  passed: 3 passed, 155 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d or whole_fiber_trace"`
  passed: 67 passed, 91 deselected.
- `PYTHONPATH=vesuvius/src:. python -c "from vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool import _import_lasagna_fit_data; m=_import_lasagna_fit_data(); print(m.__name__)"`
  passed and printed `fit_data`, matching the script-style Lasagna import path
  used by this checkout.
- `git diff --check` passed.
- Full user-dataset before/after metric profiling was not run in this shell
  because `$SRC` and `$VES` are unset; running a rewritten command would not be
  a directly comparable validation command.
