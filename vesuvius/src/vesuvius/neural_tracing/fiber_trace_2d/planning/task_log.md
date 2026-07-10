# Task Log: Trace2CP Metric Command-Line Tool

Implemented:

- Added CP-pair side-strip segment helpers in `strip_geometry.py`:
  `control_point_line_index`, `side_strip_segment_line_window`,
  `source_line_xy_from_line_window`, and `source_point_xy_for_line_index`.
- Extended side-strip grid builders with optional `anchor_column_px` so segment
  strips can place the start CP at an explicit x-column instead of always at
  image center.
- Added `FiberStripSegmentSample` and
  `FiberStrip2DLoader.build_trace2cp_segment_patch`, which resolves the start
  CP from deterministic sample order, validates a same-fiber target CP, builds
  a Lasagna/VC3D-style segment strip, and samples the center strip-z image.
- Added runner helpers for one-way trace-to-target-column tracing and
  normalized trace2cp scoring.
- Added `--trace2cp-vis`, `--trace2cp-target-offset`, and
  `--trace2cp-target-cp-index` to `runner.py`.
- `--trace2cp-vis` now writes `trace2cp_vis.jpg`,
  `trace2cp_summary.txt`, and prints a concise score/status line to stdout.
- Updated `planning/specs.md`, `docs/code_structure.md`,
  `planning/changelog.md`, and `planning/status.md`.

Deviation:

- The plan mentioned optional trace2cp TTA. V1 intentionally implements the
  unaugmented trace2cp metric only, matching the plan's conservative option.
- The plan mentioned a full CLI/export monkeypatch test. This pass added
  focused metric and segment validation in the existing test file; full CLI
  export coverage remains a useful follow-up if runner CLI behavior grows.

Validation:

```bash
python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/strip_geometry.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

Result: passed.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

Result: `100 passed in 4.63s`.
