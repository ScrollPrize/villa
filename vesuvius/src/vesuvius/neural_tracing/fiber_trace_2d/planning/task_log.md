# 3D Direction-Conditioned Recurrent Decoder Task Log

## Implementation Notes

- Checked `fiber_trace_3d.direction`: Lasagna 3x2 uses six sigmoid-compatible
  encoded channels from three ambiguous 2D projection pairs. The all-zero
  six-channel vector is off the valid encoding manifold and can be reserved as
  an unconditioned query token, but should not be decoded as a real direction.
- Added opt-in `model_3d.conditioned_decoder_enabled` support in
  `fiber_trace_3d.model`. In conditioned mode `Vesuvius3dUnetModel` emits a
  configurable latent volume and a pointwise `1x1x1` decoder consumes
  `latent + query_6ch` to produce one seven-channel prediction.
- Kept `forward(volume)` as zero-query compatibility output and added recurrent
  grouped output for visualization/native Trace2CP compatibility.
- Added conditioned training loss in `fiber_trace_3d.train`: sparse positives
  use zero-query plus deterministic perpendicular-query supervision; dense
  negatives use zero-query plus deterministic random-query presence BCE over
  all `presence_mask` voxels, including positives by design.
- Updated the active `train_s1a_nml_all_64_sd2.json` and
  `train_s1a_nml_all_128_sd2.json` configs to enable conditioned mode and
  removed the stale two-free-branch output keys from those configs.
- Updated TensorBoard sample-sheet layout text, specs, code-structure docs,
  and changelog for the conditioned mode.

## Deviations / Deferred Items

- The independent-agent review step from the local workflow has not been
  performed because delegation requires explicit user authorization with the
  available tools.
- The ordinary 3D Trace2CP projection metric still projects the zero-query
  field. Native 3D Trace2CP uses grouped zero/recurrent conditioned outputs so
  its existing branch-aware candidate scoring can choose between strongest and
  recurrent secondary predictions.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/model.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'conditioned or zero_query'`
  passed: 5 passed, 112 deselected.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/model.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/targets.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 117 passed.
- `PYTHONPATH=vesuvius/src:. python -c "import json; from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model; p='vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all_64_sd2.json'; cfg=json.load(open(p, encoding='utf-8')); m=build_fiber_trace_3d_model(cfg); print(type(m).__name__, m.conditioned_decoder_enabled, m.conditioned_latent_channels, m.output_channels)"`
  printed `FiberTrace3DNet True 64 7`.
- `PYTHONPATH=vesuvius/src:. python -c "import json; from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model; p='vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all_128_sd2.json'; cfg=json.load(open(p, encoding='utf-8')); m=build_fiber_trace_3d_model(cfg); print(type(m).__name__, m.conditioned_decoder_enabled, m.conditioned_latent_channels, m.output_channels)"`
  printed `FiberTrace3DNet True 64 7`.
- `git diff --check` passed.
