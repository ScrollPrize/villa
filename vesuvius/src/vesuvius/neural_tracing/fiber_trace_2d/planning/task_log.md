# Native 3D Whole-Fiber Continuous Strip Visualization Log

## Implementation

- Replaced the native 3D whole-fiber segment-column renderer with
  restart-delimited visual spans.
- Whole-fiber spans are rendered through the existing Trace2CP strip source
  builder with `cross_strip_height_px=64`.
- Whole-fiber mode no longer calls `_adaptive_trace2cp_cross_strip_height(...)`;
  the adaptive-height path remains for single-pair native visualization only.
- The whole-fiber callback still overwrites `trace2cp_native_3d_vis.jpg` after
  each completed CP segment. The active span is re-rendered as it grows, and
  completed restart spans are kept in the composed sheet.
- Failed overlays are clipped using that segment's own start/target CP
  coordinates inside the long span, not the span endpoints.

## Docs And Specs

- Updated `planning/specs.md` to describe restart-delimited long strips and
  fixed 64 px whole-fiber cross-strip width.
- Updated `docs/code_structure.md` with the same behavior.
- Added a 2026-07-17 changelog entry.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: `104 passed in 3.34s`.
- `git diff --check` over touched files passed.

## Deviations

- No intentional simplifications or deferred requirements.
