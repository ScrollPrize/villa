# Trace2CP Metric Stdout And Presence Visualization Plan

## Scope

- Keep Trace2CP scoring and metric definitions unchanged.
- Change only Trace2CP stdout formatting and presence visualization.
- Do not change whole-fiber visualization or training metrics.

## Implementation

- Split single-pair Trace2CP stdout:
  - first line exactly `trace2cp_error=<value>`;
  - second `trace2cp details ...` line contains raw y error, horizontal span,
    target-column status, and other diagnostics.
- Add fixed-scale presence visualization:
  - derive `presence_hw` from the model presence output already used by
    `--trace2cp-use-presence`;
  - render `0..1` probability as black-to-white with invalid pixels black;
  - overlay fiber line, start/target CPs, and selected forward/reverse traces;
  - append this panel as a column in single-pair `trace2cp_vis.jpg` and as a
    row in whole-fiber `trace2cp_fiber_vis.jpg` only when presence scoring is
    active.
- Add z-search corrected presence visualization:
  - reconstruct forward, reverse, and fused presence maps column-by-column from
    the selected trace z layer, using the same mechanism as the z-corrected
    image;
  - show those maps in the z debug column;
  - use fused z-corrected presence for the whole-fiber presence row when it is
    available.

## Spec Update

- Clarify that single-pair stdout emits `trace2cp_error=<value>` as a
  standalone line.
- Clarify that presence scoring enables a presence map debug column/row in
  Trace2CP visualizations.
- Clarify that z-search presence visualization is z-corrected from the selected
  per-column trace layer, not the center-layer map.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP runner docs.
- Update `planning/status.md` and `planning/task_log.md`; add a changelog line
  only if this becomes more than a small formatting/visualization tweak.

## Tests

- Add/adjust focused unit coverage for single-pair and whole-fiber presence
  visualization.
- Add focused unit coverage for z-corrected presence column selection.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
