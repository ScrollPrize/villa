# Native 3D Trace Smoothness Weight Plan

## Implementation

- Change `NativeTrace2CpConfig.smoothness_weight` default from `0.0` to `2.0`.
- Change the native Trace2CP CLI `--smoothness-weight` default from `0.0` to
  `2.0`.
- Keep `--smoothness-free-angle-degrees` unchanged at `10.0`.

## Spec Update

- Update native 3D Trace2CP candidate-selection specs to state that smoothing
  is enabled by default with weight `2.0` and remains CLI-overridable.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP description to document
  the default-on smoothing weight.

## Tests

- Update the default config assertion.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

## Changelog

- Add/adjust the `2026-07-17` changelog entry for the new smoothing default.

## Non-Goals

- Do not change the smoothing formula or free-angle default.
- Do not change candidate grid generation, fusion scoring, or rendering.
