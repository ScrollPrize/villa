# Task Log: Native 3D Trace2CP Scaled Inference Field

## Implementation Notes

- Added `NativeTrace2CpConfig.inference_scaledown_power` and CLI flag
  `--inference-scaledown-power`.
- Native 3D Trace2CP still reads and infers the configured native patch shape.
  The raw model products are box-downsampled with `avg_pool3d` before being
  copied into the CPU field cache.
- Downsampled validity uses the same box factor and requires every source voxel
  in the box to be valid.
- Cached inferred blocks now store native-coordinate sample origin plus
  sample spacing, so point lookups keep selected-level voxel coordinates while
  sampling the scaled field.
- Startup output and JSON summaries include the scaledown power and factor.
- Follow-up: native 3D Trace2CP `--help` now shows defaults for optional
  arguments, including options that previously had no explicit help string.
- Follow-up: native 3D Trace2CP `--help` now uses compact defaults in the
  option detail column, formatted as `[value] explanation` or just `[value]`.

## Deviations / Deferred Items

- None. The default power `0` preserves the previous unscaled behavior.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "scaled_inference or defaults_to_training_patch_size or cli_defaults or field_cache_uses_block_sampler"`
  - Result: `5 passed, 149 deselected`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d_trace2cp and (scaled_inference or field_cache or defaults_to_training_patch_size or cli_defaults or constant_field_reaches_target_plane or trace_step_limit_stops_partial_trace)"`
  - Result: `9 passed, 145 deselected`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d or whole_fiber_trace"`
  - Result: `63 passed, 91 deselected`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "cli_defaults or help_shows_defaults or scaled_inference"`
  - Result: `4 passed, 151 deselected`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "cli_defaults or help_shows_defaults"`
  - Result: `2 passed, 153 deselected`.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: passed.
- `git diff --check`
  - Result: passed.
