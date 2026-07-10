# Task Plan: Median TTA Line Tracing

## Scope

Add an opt-in median-TTA tracing mode to the runner line-trace visualization.
This should not change training, prefetching, augment-vis, dir-vis, checkpoint
loading, normal `--line-trace-vis` output, or the existing two-column TTA flock
view unless `--med-tta` is supplied.

## Plan

- Add `runner.py --med-tta` as an option for `--line-trace-vis`.
- Replace the fixed line-trace TTA transforms with one shared random geometric
  TTA scheme for both the flock and median trace. The scheme samples a
  configurable number of random combined augmentations from the regular
  training geometric ranges, defaulting to 100.
- Add a small internal representation for a TTA direction field containing:
  predicted direction image, validity mask, reference-to-TTA transform, and
  TTA-to-reference inverse transform.
- Build TTA direction fields by running the checkpointed model on each random
  geometric TTA-warped patch. The flock traces and median-TTA trace must use the
  same generated field list.
- Implement reference-space median tracing:
  - for each step, transform the current reference point into every TTA field;
  - bilinearly sample the decoded ambiguous direction field in that TTA space;
  - transform the sampled orientation back to reference space;
  - resolve ambiguous sign by keeping the sign aligned with the previous
    reference-space trace direction;
  - take the component-wise median over surviving candidate directions and
    normalize it before stepping.
- Draw the median-TTA trace as a third column in `line_trace_vis.jpg` when
  `--med-tta` is present.
- Record whether median TTA was used and the median trace point count in
  `line_trace_summary.txt`.
- Remove CPU-pinning from non-prefetch center-patch and loader coordinate
  generation. Use the configured `augment_device` for normal training,
  benchmark/profile/load-only, augment-vis, line-trace-vis, dir-vis, and direct
  chunk-request helpers. Keep prefetch dependency generation explicitly on CPU.

## Spec Update

Update `planning/specs.md` to document `--med-tta`, `--line-trace-tta-count`,
reference-space per-step median-TTA tracing, ambiguous sign handling, the
optional third JPG column, the shared random geometric TTA scheme, and
configured-device coord generation for non-prefetch paths.

## Docs Updates

Update `docs/code_structure.md` to describe `--med-tta`, how it differs from
the existing TTA flock visualization, and which loader paths use configured
devices versus CPU-only prefetch.

## Testing

- Add focused tests for:
  - reference/TTA point transform round trips still work;
  - median-TTA tracing resolves ambiguous sign and steps in reference space.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

Add a changelog entry for median-TTA line tracing.
