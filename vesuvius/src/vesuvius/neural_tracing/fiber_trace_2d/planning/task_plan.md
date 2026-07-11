# Bidirectional Trace2CP Plan

## Implementation

- Keep `build_trace2cp_segment_patch` unchanged; it already returns one segment
  strip containing both selected CPs.
- Add a small runner helper that traces and scores one CP direction on an
  existing direction field.
- Run that helper twice in `--trace2cp-vis`: start-to-target and
  target-to-start.
- For `--med-tta`, trace both directions through the same reference/TTA
  direction-field list.
- Report the public `trace2cp_score` as the arithmetic mean of the two
  directional scores, while preserving per-direction raw errors, statuses, and
  trace point counts in the summary.
- Draw both directional traces in `trace2cp_vis.jpg` with distinct colors and
  mark both CPs.

## Spec Update

- Update `planning/specs.md` so Trace2CP is defined as a bidirectional
  segment inspection.
- Specify that the normalized score is the average of forward and reverse
  normalized errors.
- Specify that the single JPG draws both directional traces.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP runner documentation to describe the
  two traces, averaged score, and summary fields.
- Replace `planning/task_log.md` with this task's implementation notes and
  validation results.
- Update `planning/changelog.md` with a one-line dated entry.

## Testing

- Add focused tests for the bidirectional trace helper:
  - both traces reach their opposite target on a constant direction field;
  - the aggregate score is the average of the two directional scores.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
