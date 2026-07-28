# Task Log: Native 3D Trace2CP Pyramid Scaledown

## Implementation Notes

- Replaced native 3D Trace2CP product signal scaledown with Lasagna's shared
  `_pyrdown3d` Gaussian pyramid downscale helper.
- Added a `B,C,D,H,W` wrapper that flattens batch and channel into the channel
  axis expected by `_pyrdown3d`, then restores the batch layout.
- Preserved the processing order: model inference, optional pyramid scaledown,
  optional inference-field Gaussian blur, trusted-core crop/cache.
- Kept validity-mask downsampling conservative: a scaled field voxel is valid
  only if all source voxels in the corresponding cell were valid.
- Updated CLI help/spec wording from box scaledown to Gaussian pyramid
  scaledown.

## Deviations / Deferred Items

- None. Validity-mask reduction remains conservative support handling, not
  signal scaling.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "scaled_inference or inference_blur or cli_defaults or help_shows_defaults"`
  - Result: `5 passed, 151 deselected`.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d or whole_fiber_trace"`
  - Result: `65 passed, 91 deselected`.
- `git diff --check`
  - Result: passed.
