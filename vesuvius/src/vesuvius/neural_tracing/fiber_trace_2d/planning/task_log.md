# Trace2CP Similarity Debug Column Task Log

## Notes

- Current combined Trace2CP scoring already has start/target CP embeddings and
  a same-fiber CP embedding bank, but only aggregate component losses are
  retained for summaries.
- The requested debug output can be generated from the predicted embedding
  field after inference without changing candidate selection or metrics.
- Whole-fiber Trace2CP pair evaluation disables retained similarity maps to
  avoid storing five full-size debug images per pair when the long-strip
  compositor does not use them.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: 159 passed in 6.11s.
- `git diff --check` on touched files returned clean.
