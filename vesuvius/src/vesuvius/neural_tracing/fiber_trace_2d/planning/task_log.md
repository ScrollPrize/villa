# Trace2CP Compact Geometry CP-Span Coverage Task Log

## Findings

- The `invalid normal without stored reason` diagnostics came from line points
  that were marked invalid in compact geometry but had no Lasagna failure
  reason.
- The reason is a preload coverage bug: compact geometry sampled CP
  source-window ranges, but Trace2CP can request the full centerline span
  between two CPs. Interior points between two disjoint CP windows were never
  sampled.
- Strip width is not involved. The affected points are fiber centerline points
  inside the CP-to-CP span.

## Implementation Notes

- `_required_line_ranges_for_record(...)` now includes consecutive CP-to-CP
  line spans in addition to CP source windows.
- `_FiberLineGeometry` keeps line-level invalid Lasagna reasons.
- The Trace2CP invalid-interval error now reports invalid runs, overlapping
  valid intervals, stored reasons, and for unexpected unsampled points a direct
  Lasagna probe with grad/nx/ny ranges and principal-axis status.
- No normal fill-in, propagation, or interpolation fallback was added.

## Deviations Or Deferrals

- Existing compact geometry stores built by older code can still show the
  defensive `not sampled by compact geometry preload` diagnostic until rebuilt.
  New loader construction should sample the CP-to-CP span and either validate it
  or report the real sampled Lasagna failure reason.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k "compact_geometry or required_line_ranges or trace2cp_segment_trims or trace2cp_segment_rejects"`
  passed: 7 passed, 266 deselected.
