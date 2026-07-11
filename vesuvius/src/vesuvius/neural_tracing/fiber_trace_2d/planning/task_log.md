# TTA Reference Mapping Performance Log

Current task: fix large Trace2CP median-TTA inference appearing to hang after
analytic direction decoding was added.

Findings:

- The interrupted stack stopped in `_nearest_tta_point_for_reference()`, which
  scanned a full dense TTA output-to-reference coordinate image for every trace
  step and every TTA field.
- That made tracing cost scale with `trace_steps * tta_fields * height * width`
  even though the augmentation object already builds the opposite map.
- The loader's TTA patch builders now expose both directions: `source_xy_grid`
  for output-to-reference mapping, and `reference_to_tta_xy_grid` for direct
  reference-to-TTA point lookup.

Implementation notes:

- Added `reference_to_tta_xy_grid` to `FiberStripTtaPatch` and
  `_TtaDirectionField`.
- Replaced dense nearest-grid lookup in median-TTA direction sampling with
  bilinear lookup against the prebuilt reference-to-TTA map.
- Kept output-to-reference maps for mapping TTA directions and traced TTA
  flock points back to the reference strip.
- Removed the remaining legacy dense nearest output-pixel helper, unused
  formula-based affine point-mapping helpers, and the test-only Jacobian
  pseudo-inverse helper so geometric augmentation map inversion code does not
  exist in `fiber_trace_2d`.
- Added tests for explicit shifted forward/backward TTA maps and a structural
  guard preventing a dense nearest reference-point helper from returning.
- Updated the specs/docs to require prebuilt paired map directions and to
  forbid after-the-fact inversion by search, iterative solvers, or runtime
  analytic inverse formulas.

Validation:

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed with `115 passed in 5.69s` after the stricter cleanup.
