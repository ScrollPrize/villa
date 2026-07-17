# Native 3D Trace2CP First-Step CP-Tangent Relaxation Log

## Planning

- Replaced `planning/task.md` with the current user task: relax native 3D
  Trace2CP first-step CP tangent scoring.
- Replaced `planning/task_plan.md` with a detailed plan following
  `fiber_trace_2d/AGENTS.md`.
- Replaced `planning/status.md` with a current-task checklist.
- Reviewed the current native 3D Trace2CP spec section and preserved the
  existing requirements:
  - CP-local tangent seeding remains;
  - candidate Lasagna normals are sampled directly at trace candidate points;
  - no reference-line normal interpolation;
  - regular direction/presence/smoothness scoring resumes after the first step.

## Deviations / Deferred

None.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: `96 passed in 8.93s`.
- `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/changelog.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/status.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_log.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_plan.md`
  - Result: clean.

## Implementation Notes

- Added `first_step_mask` to the native 3D candidate scorer.
- Added a normal/elevation-only first-step gate using candidate-point Lasagna
  normals and preserving normal sign ambiguity.
- First-step candidates now have zero smoothness loss.
- Greedy tracing passes the mask for `_step_index == 0`; beam tracing passes
  it for root-depth frontier states.
- Native Trace2CP summaries include `first_step_cp_tangent_relaxed: true`.
- Added focused scorer tests plus a greedy/beam trace-path regression.
