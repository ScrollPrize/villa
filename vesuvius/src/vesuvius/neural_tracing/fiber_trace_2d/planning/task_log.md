# Native 3D Trace2CP Cumulative Tangent Smoothness Log

## Planning

- Replaced current planning docs for the cumulative tangent-only smoothness
  task.
- Preserved existing native 3D Trace2CP first-step relaxation and normal-aware
  local smoothness requirements.

## Deviations / Deferred

None.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: `99 passed in 3.48s`.
- `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/changelog.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/status.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_log.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_plan.md`
  - Result: clean.

## Implementation Notes

- Added `cumulative_smoothness_steps` and
  `cumulative_smoothness_tangent_weight` to native 3D Trace2CP config and CLI.
- Added a tangent-plane-only cumulative smoothness helper using
  candidate-point Lasagna normals.
- Greedy and beam trace states now carry a running history heading.
- The cumulative term is zeroed for first-step states, preserving the CP-root
  tangent relaxation.
- The cumulative term is skipped for invalid/unavailable normals or degenerate
  tangent projections.
