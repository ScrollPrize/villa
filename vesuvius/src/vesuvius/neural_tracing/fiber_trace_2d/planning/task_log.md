# Shared 3D Tiled Inference Task Log

## Implementation Notes

- Implemented the first numbered plan step only: extract shared tiled
  predict3d support without changing Lasagna predict3d behavior.
- Added `lasagna/tiled_predict3d.py` containing the reusable helper layer:
  canonical tile/grid helpers, OME-Zarr chunk path/existence/completeness
  helpers, temp cleanup, atomic chunk writes, rolling z-band accumulator,
  progress formatting, tile reading, and the current dual-resolution tiled
  3D inference loop.
- Updated `lasagna/preprocess_cos_omezarr.py` to import/re-export those helper
  names from `tiled_predict3d`, preserving existing private imports used by
  tests and scripts.
- Removed the moved helper implementations from `preprocess_cos_omezarr.py`
  so the wrapper no longer carries duplicate rolling/tiled inference code.
- Updated tests that monkey-patched moved internals to patch the function
  implementation module instead of the compatibility wrapper.
- Implemented the second numbered plan step only: define shared adapter
  interfaces without porting Lasagna predict3d to those interfaces yet.
- Added `OutputChannelSpec` and `OutputProductSpec` to describe coherent
  product/channel bundles, output level/scaledown, chunk size, dtype/range,
  and pyramid policy.
- Added runtime-checkable `ModelAdapter` and `OutputAdapter` protocols for
  product-specific model loading/tile inference/accumulation and
  chunk-completeness/write/metadata behavior.
- Re-exported the new interface names from `preprocess_cos_omezarr.py` for
  compatibility with the current wrapper-style imports.
- Implemented the third numbered plan step: port current Lasagna predict3d
  model/product boundaries onto the adapter layer without changing the CLI or
  rolling accumulator semantics.
- Added `LasagnaCosPredict3DAdapter` describing the existing model-emitted
  products:
  fine `cos` and the coarse `grad_mag/nx/ny` normal bundle. Optional
  `pred_dt` is represented as a derived product so missing `pred_dt` remains
  independent of model inference.
- Added `LasagnaOmeZarrOutputAdapter` for product chunk completeness,
  per-channel bundle checks, and atomic chunk writes through the shared
  output boundary.
- Routed `run_preprocess_3d(...)` model loading and tile inference through
  `LasagnaCosPredict3DAdapter`.
- Routed current `cos` and `grad_mag/nx/ny` resume checks and chunk writes
  through `LasagnaOmeZarrOutputAdapter`; manifest/pyramid generation remains
  in the Lasagna wrapper to preserve current behavior.
- Added focused tests for Lasagna product schema, normal-bundle completeness,
  partial product writes, and adapter-driven tiled inference.
- Implemented the fourth numbered plan step: add the 3D fiber inference
  adapter and fiber output product schema without adding the CLI yet.
- Added `vesuvius.neural_tracing.fiber_trace_3d.inference_adapter` with
  `FiberTrace3DPredictAdapter`, preserving one coherent seven-channel product
  per branch/recurrent option:
  `dir0_z`, `dir1_z`, `dir0_y`, `dir1_y`, `dir0_x`, `dir1_x`, and
  `presence`.
- The fiber adapter uses `build_fiber_trace_3d_model(...)` for model
  construction, the existing training snapshot loader for checkpoints, the
  existing mixed-precision/autocast helpers for tile inference, and keeps
  conditioned recurrent output as separate grouped options when requested.
- Added `FiberTrace3DOmeZarrOutputAdapter` so fiber option completeness is
  evaluated as an indivisible seven-channel bundle and chunk writes go through
  the shared atomic Zarr writer.
- Exported the adapter names from `vesuvius.neural_tracing.fiber_trace_3d`
  while keeping training-helper imports lazy, so running the training module
  does not import itself through package initialization.
- Added focused fiber tests for multi-branch schema, branch-output splitting,
  conditioned recurrent grouping, and seven-channel option completeness.
- Implemented the fifth numbered plan step: add a fiber 3D inference CLI on
  top of the shared tiled inference helpers.
- Added shared helpers to `lasagna.tiled_predict3d` for crop bounds,
  downsample index/size calculations, S3 auto-download from `_download`
  metadata, and base-shape resolution so fiber inference can use the same
  input/crop/download semantics as Lasagna `predict3d`.
- Added `_infer_tiled_products_3d(...)`, a shared single-resolution product
  runner for coherent output bundles. It uses the same canonical tile lattice,
  rolling z-band accumulators, output chunk completeness checks, input-support
  skips, temp-file cleanup, progress output, and atomic chunk writes, but does
  not force products into Lasagna's `cos` plus `grad_mag/nx/ny` split.
- Added `python -m vesuvius.neural_tracing.fiber_trace_3d.infer` with common
  tiled inference arguments: `--input`, `--output`, `--checkpoint`,
  `--tile-size`, `--overlap`, `--border`, `--scaledown`, `--crop`,
  `--device`, `--no-download`, `--levels`, and `--ome-chunk`, plus fiber
  arguments `config`, `--recurrent-steps`, and `--output-prefix`.
- The fiber CLI defaults `--overlap` to 16 so the common tiled arguments have
  a usable default with 64-voxel fiber tiles; callers can still pass an
  explicit overlap to reproduce another lattice.
- Fiber inference now normalizes each input tile through the existing 3D
  fiber `_normalize_image(...)` path before model inference, preserving the
  training `image_normalization` semantics.
- The fiber CLI writes one OME-Zarr group per output channel under each
  coherent option bundle, e.g.
  `fiber/option_000/dir0_z`, ..., `fiber/option_000/presence`; resume
  completeness for an option requires all seven channel chunks.
- Added a tiny CPU end-to-end regression test that runs the fiber inference
  writer on a local zarr volume and verifies all seven option channels are
  materialized as uint8 OME-Zarr arrays.
- Implemented the sixth numbered plan step: verify crop-composable output and
  output-chunk-only resume behavior with a deterministic shared-runner
  regression.
- Re-exported `_infer_tiled_products_3d(...)` through
  `preprocess_cos_omezarr.py` so compatibility-style tests can exercise the
  shared product runner through the same wrapper module used by existing
  predict3d tests.
- Added a non-constant fake product adapter test that runs the same input
  through two overlapping crops into separate OME-Zarr outputs and verifies
  the shared output chunk bytes are identical.
- The same test reruns the completed full-crop output with an adapter that
  raises on model inference, proving resume skips completed product chunks
  based only on durable output chunk completeness.
- Implemented the seventh numbered plan step for fiber output metadata and
  pyramid policy.
- Added an atomic `fiber_trace_3d_inference.json` manifest at the fiber output
  root. It records the input/base/output scaledowns, crop and output region,
  tile parameters, product paths, and the coherent seven-channel option
  bundle schema.
- The fiber manifest explicitly marks products as
  `fiber_trace_3d_option_bundle` with `lasagna_3x2_ambiguous` direction
  encoding and sets `not_lasagna_normal_products=true`, so fiber outputs are
  not confused with Lasagna `grad_mag/nx/ny`.
- Fiber products remain data-level-only for V0. The manifest records
  `pyramid.policy=data_level_only` and
  `coarser_fiber_pyramids_built=false`; no coarser fiber pyramid generation
  was added.
- Implemented the eighth numbered plan step: backwards-compatible migration
  coverage.
- Added a regression asserting shared tiled predict3d helpers and adapter
  protocols are still re-exported through `preprocess_cos_omezarr.py`, so old
  wrapper imports continue to work after the extraction.
- Added a CLI-dispatch regression for `preprocess_cos_omezarr.py predict3d`
  that verifies legacy arguments still reach `run_preprocess_3d(...)` with
  the same parsed values and that `--no-download` suppresses automatic
  download work.
- Did not change 3D fiber training or Trace2CP runtime code paths in this
  migration step.
- Implemented the docs/spec/changelog step.
- Updated Lasagna predict3d docs in `lasagna/README.md`,
  `lasagna/docs/tifxyz_training.md`, and `lasagna/docs/3d_unet_training.md`
  to describe `lasagna.tiled_predict3d` as the shared tiled mechanics layer
  while keeping Lasagna `predict3d` CLI/output compatibility explicit.
- Updated `fiber_trace_2d/docs/code_structure.md` with the new
  `fiber_trace_3d/inference_adapter.py` and `fiber_trace_3d/infer.py`
  responsibilities, CLI command, output channel layout, resume behavior, and
  data-level-only manifest policy.
- Added the shared 3D tiled inference contract to `planning/specs.md`,
  including adapter boundaries, crop-composable global tile/chunk lattices,
  output-chunk-only resume, product-independent completeness, atomic writes,
  and the fiber seven-channel option-bundle schema.
- Added a `2026-07-27` changelog entry for the shared tiled inference
  extraction and fiber inference CLI.

## Deviations / Deferred Items

- The current Lasagna dual-resolution rolling accumulator still performs
  product accumulation inside the existing loop; the adapter boundary now owns
  model invocation and chunk state/writes for Lasagna, while
  `_infer_tiled_products_3d(...)` owns generic single-resolution product
  accumulation for fiber/shared outputs.
- Fiber OME-Zarr pyramid construction remains deliberately out of scope for
  V0. The CLI may create requested OME-Zarr metadata levels, but only writes
  model output data-level chunks; the manifest now records that policy
  explicitly.
- No production-code change was needed for the backwards-compatibility
  migration step; the existing wrapper already preserved the old import and
  CLI surfaces.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=lasagna:vesuvius/src:. pytest -q lasagna/tests/test_preprocess_cos_omezarr.py`
  passed: 28 passed in 1.36s.
- `python -m py_compile lasagna/preprocess_cos_omezarr.py lasagna/tiled_predict3d.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "fiber_inference_adapter or fiber_output_adapter"`
  passed: 4 passed, 122 deselected in 1.97s.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 124 passed, 2 skipped in 3.21s.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/inference_adapter.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/__init__.py`
  passed.
- `python -m py_compile lasagna/tiled_predict3d.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/infer.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/inference_adapter.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "fiber_infer or fiber_inference_adapter or fiber_output_adapter"`
  passed: 5 passed, 122 deselected in 1.89s.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 125 passed, 2 skipped in 3.57s.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=lasagna:vesuvius/src:. pytest -q lasagna/tests/test_preprocess_cos_omezarr.py`
  passed: 28 passed in 1.43s.
- The same pytest command without `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` failed
  before test collection because the environment auto-loaded a missing
  `zarr.testing` pytest plugin.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=lasagna:vesuvius/src:. pytest -q lasagna/tests/test_preprocess_cos_omezarr.py`
  passed after the crop-composability regression: 29 passed in 1.44s.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "fiber_infer or fiber_inference_adapter or fiber_output_adapter"`
  passed after the shared-runner regression: 5 passed, 122 deselected in 1.79s.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/infer.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/inference_adapter.py`
  passed after the fiber manifest update.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "fiber_infer or fiber_inference_adapter or fiber_output_adapter"`
  passed after the fiber manifest update: 5 passed, 122 deselected in 2.02s.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=lasagna:vesuvius/src:. pytest -q lasagna/tests/test_preprocess_cos_omezarr.py`
  passed after the backwards-compatibility migration tests: 31 passed in 1.44s.
- `python -m py_compile lasagna/preprocess_cos_omezarr.py lasagna/tiled_predict3d.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/infer.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/inference_adapter.py`
  passed after the backwards-compatibility migration tests.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed after the backwards-compatibility migration tests: 125 passed,
  2 skipped in 3.37s.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=lasagna:vesuvius/src:. pytest -q lasagna/tests/test_preprocess_cos_omezarr.py`
  passed after docs/spec updates: 31 passed in 1.53s.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed after docs/spec updates: 125 passed, 2 skipped in 3.52s.
- `python -m py_compile lasagna/preprocess_cos_omezarr.py lasagna/tiled_predict3d.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/infer.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/inference_adapter.py`
  passed after docs/spec updates.
