# Trace2CP Refined Strip Off-Strip Clipping Log

## Implementation Notes

- Task started from user report that refined Trace2CP visualization raises when
  traced points leave the source strip valid area. This should clip display
  points instead of aborting, because off-strip points are irrelevant to the
  rendered refined strip.
- Updated `build_trace2cp_refined_segment_source` to filter source-grid samples
  by the source strip validity mask before building the regenerated strip.
- Follow-up failure showed that original trace endpoints can also be clipped by
  the source strip domain. The helper now uses the first and last remaining
  valid trace points as the regenerated strip endpoints instead of raising
  `lost a CP endpoint`.
- Added focused loader regressions for clipping an off-strip interior trace
  point, clipping off-strip endpoint trace points, and still rejecting a trace
  with fewer than two valid points after clipping.
- Updated specs/changelog to state that refined/regenerated strip construction
  clips off-strip traced points instead of treating them as fatal.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'refined_segment_source'`
  passed after endpoint clipping update: `5 passed, 271 deselected in 3.39s`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed after endpoint clipping update: `276 passed in 7.97s`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed after endpoint clipping update: `104 passed in 3.31s`.
- `git diff --check -- <touched files>` passed.

## Deviations

- No intentional simplifications or deferred requirements.
