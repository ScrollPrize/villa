# Thin Lasagna Port For Fiber 3D Inference Task Log

## Implementation

- Moved generic OME-Zarr output, product group creation, Lasagna manifest
  writing, and product pyramid helpers into `lasagna/tiled_predict3d.py`.
- Made `preprocess_cos_omezarr.py` use the shared product adapter, group
  creation, manifest writing, and pyramid helper directly.
- Extended the shared tiled product runner with `accumulator_channel_count` and
  `finalize_product_slab(...)`, so raw model channels can be accumulated first
  and converted only when complete output chunks are finalized.
- Kept Lasagna cos predict3d behavior on the shared path by implementing
  product splitting/finalization on `LasagnaCosPredict3DAdapter`.
- Reworked `fiber_trace_3d/inference_adapter.py` so fiber options accumulate
  the raw seven model channels internally but persist only `presence`, `nx`,
  and `ny`.
- Removed the intermediate fiber output adapter class, raw-bundle constant,
  and raw-bundle conversion helper from active source and public exports.
- Reworked `fiber_trace_3d/infer.py` so `--output` is a `.lasagna.json`
  manifest path. The manifest stem defines per-channel OME-Zarr outputs, and
  stale temp cleanup, OME-Zarr group creation, manifest writing, and pyramid
  generation all go through shared predict3d helpers.
- Updated tests, specs, code-structure docs, changelog, status, and todo
  entries to describe the final Lasagna-style output rather than the
  intermediate raw-bundle V0.

## Legacy Removal Check

- No active source exports or imports `FiberTrace3DOmeZarrOutputAdapter`.
- No active source exports or imports `FIBER_TRACE_3D_OPTION_CHANNELS`.
- No active source keeps `product_channel_arrays_from_output(...)`.
- No active CLI accepts `--output-prefix`; the fiber inference CLI derives
  output products from the `.lasagna.json` manifest stem.
- No code writes `fiber_trace_3d_inference.json`; the only active test
  reference asserts that it is not created.
- No fiber-local manifest writer, output group creator, or pyramid writer
  remains in `fiber_trace_3d/infer.py`.
- Remaining old-name hits are in planning text that documents the required
  removals, not in runtime code or public APIs.

## Deviations / Deferred Items

- No legacy compatibility shim was kept for the intermediate fiber raw-bundle
  inference API.
- Multi-option fiber inference remains supported as separate coherent
  `presence/nx/ny` option products. No option collapse/merge mode was added.
- `--pyramid-workers` was added as an explicit CLI control for the existing
  Lasagna pyramid helpers; this does not change output semantics.

## Validation

- `python -m py_compile lasagna/tiled_predict3d.py lasagna/preprocess_cos_omezarr.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/infer.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/inference_adapter.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/__init__.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py lasagna/tests/test_preprocess_cos_omezarr.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=lasagna:vesuvius/src:. pytest -q lasagna/tests/test_preprocess_cos_omezarr.py`
  passed: 31 tests.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "fiber_infer or fiber_inference_adapter or fiber_output_adapter"`
  passed: 5 tests, 123 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 126 tests, 2 skipped.
