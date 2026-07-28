# Task Log: Native 3D Trace2CP Shared Live Inference

- Added `lasagna/normal_encoding.py` and moved Lasagna 3x2 normal estimation
  plus compact `nx/ny` uint8 encoding there. `preprocess_cos_omezarr.py` and
  `fiber_trace_3d/inference_adapter.py` now import the shared helper directly.
- Kept `preprocess_cos_omezarr.py` usable in both script-style and
  package-style imports by making its `common` import package-safe.
- Added `fiber_trace_3d/prediction.py` for shared raw fiber output decoding:
  six-channel direction-only output and grouped seven-channel branch/option
  output both decode through the existing analytic Lasagna 3x2 decoder.
- Adapted `NativeTraceFieldCache` to use `FiberTrace3DPredictAdapter` for tile
  preprocessing, recurrent/conditioned inference, mixed precision, and raw
  option splitting. The sparse block cache, trusted-core routing, strict VC3D
  sampling, and trace search/fusion behavior were left intact.
- Added an internal `NativeTracePredictionField` protocol as the boundary for
  future precomputed prediction providers; no precomputed provider was
  implemented in this task by user request.
- Fixed checkpoint/config mismatch handling: `FiberTrace3DPredictAdapter` now
  uses an embedded checkpoint `config` for model construction and option count,
  so a legacy two-branch checkpoint is not loaded into a conditioned-decoder
  model just because the runtime config is newer. For older checkpoints without
  embedded config, it can infer the minimal non-conditioned branch layout from
  `net.decoder.final_seg_layer.weight` when that shape is available.
- Validation:
  `PYTHONPATH=vesuvius/src:. python -c "import vesuvius.neural_tracing.fiber_trace_3d; import vesuvius.neural_tracing.fiber_trace_3d.infer; import vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool"`
  passed.
- Validation:
  `PYTHONPATH=. python -c "import lasagna.preprocess_cos_omezarr"`
  passed.
- Validation:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "package_style_lasagna_path or compact_normal_encoding or fiber_prediction_decodes or native_3d_trace2cp_block_router or field_cache or constant_field_reaches_target_plane"`
  passed: 9 passed, 134 deselected.
- Validation:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 142 passed, 2 skipped after the checkpoint-config regression was
  added.
- Validation:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=lasagna:. pytest -q lasagna/tests/test_preprocess_cos_omezarr.py`
  passed: 31 passed.
- No planned simplifications or postponed implementation items beyond the
  explicit non-goals: no `__init__` eager-import cleanup and no precomputed
  prediction provider.
