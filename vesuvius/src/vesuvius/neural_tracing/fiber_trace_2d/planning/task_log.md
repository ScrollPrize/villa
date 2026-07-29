# Python Native 3D Trace2CP Multi-Fiber Task Log

## Implementation Notes

- Changed `trace2cp_tool.py --fiber-json` to accept one or more paths.
- Added indexed output stems while keeping existing single-run filenames
  unchanged.
- Added `run_native_trace2cp_many(...)`, which reuses one loaded model and runs
  the existing whole-fiber tracer sequentially over the supplied JSON fibers.
- Multi-fiber mode rejects sample-index and explicit CP selectors because the
  accumulated score is defined over complete fibers.
- Added accumulated scoring from summed restarts and summed original-line
  reference lengths. Optional physical `err/m` is reported only when every
  fiber result has a VC3D-derived physical length.
- Added `trace2cp_native_3d_summary_all.json` for aggregate output and
  per-fiber indexed summaries/JPGs.

## Deviations / Deferred Items

- No parallel multi-fiber execution was added; the request specified sequential
  execution.
- No model/checkpoint reload per fiber is performed; this was improved by
  sharing the loaded model across the sequential runs.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  - passed
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - passed: 164 tests, 2 skipped
- `git diff --check`
  - passed
