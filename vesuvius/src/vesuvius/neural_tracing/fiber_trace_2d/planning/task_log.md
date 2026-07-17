# Native 3D Trace2CP All-Pairs Direction Product Log

## Planning

- Replaced current planning docs for the native 3D Trace2CP all-pairs
  direction product scoring task.
- Planned a default-enabled switch with an opt-out CLI flag for comparison.

## Deviations / Deferred

None.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: `101 passed in 3.12s`.
- `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/changelog.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/status.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_log.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_plan.md`
  - Result: clean.

## Implementation Notes

- Added `all_pairs_direction_product` to native 3D Trace2CP config and summary.
- Added CLI opt-out `--no-all-pairs-direction-product`.
- Default candidate score now multiplies all six pairwise direction dots among
  previous step direction, current sampled direction, candidate step direction,
  and candidate sampled direction, times presence.
- First-step root scoring neutralizes previous/current tangent-plane pair terms
  so the CP-root normal-only relaxation remains intact.
- Legacy two-dot score remains available through the opt-out flag and tests.
