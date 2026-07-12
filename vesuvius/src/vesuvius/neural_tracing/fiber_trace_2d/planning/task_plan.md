# Trace2CP Vertical Range Increase Plan

## Scope

- Change only the Trace2CP segment strip height multiplier.
- Preserve width generation, CP anchoring, z-search layer bounds, metrics, and
  tracing/scoring logic.

## Implementation

- Increase the Trace2CP segment height multiplier from `4` to `8`.
- Update the regression test that asserts segment height.

## Spec Update

- Update Trace2CP segment height wording from four times configured patch height
  to eight times configured patch height.

## Docs Updates

- Update `docs/code_structure.md`, changelog, status, and task log.

## Tests

- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
