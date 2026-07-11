# Bidirectional Trace2CP Log

Current task: make `--trace2cp-vis` trace selected CP segments in both
directions and report the average of the two directional Trace2CP scores.

Plan review:

- Independent read-only plan review found the plan consistent with
  `task.md`, `specs.md`, and `plan.md`.
- Review gaps addressed during implementation:
  - specs now explicitly define bidirectional tracing instead of one traced
    line;
  - reverse scoring evaluates at the original start CP x-column with its own
    denominator/status;
  - each direction's ambiguous sign is seeded toward that direction's target;
  - summary/stdout use explicit forward/reverse fields;
  - tests cover bidirectional averaging and decreasing-x target-column
    precedence over RF-margin rejection.

Implementation notes:

- Left `build_trace2cp_segment_patch` unchanged; one segment strip already
  contains both CPs.
- Added private runner helpers that trace and score a single Trace2CP direction
  and aggregate forward/reverse results.
- Base `--trace2cp-vis` now traces start-to-target and target-to-start on the
  same decoded direction field.
- `--trace2cp-vis --med-tta` now traces both directions through the same
  reference/TTA direction-field set.
- `trace2cp_vis.jpg` draws the original strip line, the forward trace, the
  reverse trace, both CP columns, and an averaged-score label band.
- `trace2cp_summary.txt` and stdout keep the averaged score and raw error, plus
  per-direction scores, raw errors, target x-columns, reach statuses,
  termination reasons, and trace point counts.

Validation:

- Compile check:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed.
- Focused loader/runner tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: `108 passed in 3.85s`.
