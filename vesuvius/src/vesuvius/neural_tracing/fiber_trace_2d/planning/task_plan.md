# Trace2CP Center-Biased Metric Plan

## Implementation Plan

- Add a small Trace2CP center-penalty helper in `runner.py`.
- Apply that helper while evaluating overlap candidates in
  `_closest_trace2cp_approach`.
- Select the candidate with the smallest considered distance, where considered
  distance is actual vertical gap multiplied by the center penalty.
- Preserve actual vertical gap diagnostics separately from considered metric
  distance.
- Add summary/visualization fields for the penalty and considered distance.
- Compose Trace2CP visualization columns so `--med-tta` shows the current
  median-TTA result plus a second reference-only inference column.
- Keep non-TTA Trace2CP visualization as a single reference-only column.
- Add tests covering:
  - penalty is `1x` at CP midpoint and `2x` at either CP;
  - a centered candidate can win even when its actual gap is larger than a
    CP-edge candidate gap;
  - existing closest-approach behavior remains unchanged when the best
    candidate is already centered.

## Spec Update

- Update Trace2CP score wording: public score uses center-penalized considered
  distance, while actual y separation remains diagnostic.
- Add that `--trace2cp-vis --med-tta` renders a reference-only comparison
  column next to the selected median-TTA column.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP section with center-penalty
  semantics and reference comparison column behavior.

## Tests

- Run:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog Update

- Add a 2026-07-11 changelog line for the center-biased Trace2CP score.
