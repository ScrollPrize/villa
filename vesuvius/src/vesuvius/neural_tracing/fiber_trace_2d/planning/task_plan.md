# Trace2CP Top-Model Direction Debug Plan

## Implementation

- Replace the top-model offset-layer direction picker with a fused median:
  normalize every decoded layer direction, keep only valid directions within
  45 degrees of horizontal using `abs(dx) >= cos(45 degrees)`, align signs to
  the positive horizontal direction, take the component-wise median, and
  normalize the result.
- Keep returning a debug layer map by selecting the contributing layer closest
  to the fused median direction.
- Keep top traces ambiguity-aware when bilinearly sampling the fused field.
- Draw the forward and reverse top traces with equal stroke weight and opacity.

## Spec Update

- Update `--trace2cp-top-model-dir-vis` semantics from single most-horizontal
  layer selection to aligned median fusion over horizontally plausible layers.
- Document that the top traces remain visualization-only.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP/top-view runner notes.
- Add a changelog line and current-task log entry.

## Tests

- Update focused tests so top-direction fusion verifies median behavior,
  horizontal thresholding, and sign alignment.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
