# Trace2CP Top-Direction DP Optimized-Line Diagnostics Plan

## Implementation

- Add a small debug payload for the top-model DP path containing the optimized
  top-strip path, selected z-layer offsets, and the derived xyz trace.
- Change `_trace2cp_top_model_direction_overlay` to return that payload in
  addition to the existing overlay image/count/debug text.
- Extend traced top-strip sampling to accept an optional column-wise top-row
  offset so the resliced image follows the optimized top path rather than only
  drawing the path over the reference slice.
- In Trace2CP pair evaluation, when top-dir visualization is enabled:
  - reconstruct the optimized top strip from the optimized top path and z
    offsets;
  - reconstruct the optimized side slice from the optimized side-z offsets via
    the existing z-corrected side-slice helper;
  - reconstruct optimized top and side presence views with the same path/offsets.
- Append the optimized panels below the current top-direction/path panel in
  single-pair and fiber visualizations.

## Spec Update

- Document that top-dir visualization reports the DP optimized line with both
  top-path and side-z-offset derived panels.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP runner notes for the extra
  optimized-line diagnostics.

## Testing

- Add a focused unit test for traced top-strip column offsets.
- Add a focused unit test for the top-model direction overlay returning a
  finite optimized xyz debug path.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

- Add a 2026-07-14 changelog entry for top-dir optimized-line diagnostics.
