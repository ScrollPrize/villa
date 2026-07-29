# Task Log: Native 3D Trace2CP Point-Lookup Optimization

## Starting Point

- Current quality-matching native 3D Trace2CP path uses sparse corner/tensor
  Lasagna normal sampling with `eigh` principal-axis reconstruction.
- Last retained benchmark result before this task:
  - restarts: 3 / 87 segments
  - trace wall: about 102 s
  - dominant remaining stages: `trace_candidate_score`,
    `score_sample_points`, `inference_forward`, and `field_sample_lookup`.
- Previous analytic principal-axis experiment was not retained as default:
  - analytic method: 18 restarts, trace wall about 103 s
  - default remains `eigh`.

## Implementation Notes

- Added `--profile` as an explicit native 3D Trace2CP option. Metric-only
  runs still report the final metric and total wall/CPU trace timing, but do
  not allocate/use the detailed per-stage profiler unless requested.
- Kept the sparse corner/tensor Lasagna normal sampler and `eigh` principal
  axis reconstruction as the default quality path.
- Added an opt-in `--normal-principal-axis-method analytic` implementation for
  testing the closed-form tensor principal-axis reconstruction. It is not the
  default because the approved benchmark changed the restart metric.
- Tried the invasive direct cached-block trilinear point lookup for model
  output field sampling. It preserved the restart metric but was slower than
  the current grouped `grid_sample` lookup, so it was reverted.
- Tried larger default inference-block batches:
  - batch 8 completed faster but changed the benchmark metric from 3 restarts
    to 7 restarts, so it was reverted.
  - batch 4 showed the same early quality regression pattern and was reverted.

## Validation

- Focused tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - result: 160 passed, 2 skipped in 5.76 s.
- Syntax check:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  - result: passed.
- Whitespace check:
  `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_plan.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/status.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_log.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/changelog.md`
  - result: passed.
- Approved whole-fiber benchmark command was reused for each timing run.
  Retained final run without `--profile`:
  - restarts: 3 / 87 segments
  - `err/kvx=0.2`
  - `err/m=19.6 (38.2mm)`
  - `trace_wall_s=99.790`
  - `trace_cpu_s=601.873`
- Rejected direct cached-block lookup benchmark:
  - restarts: 3 / 87 segments
  - `trace_wall_s=112.503`
  - slower than the retained grouped `grid_sample` path, so not kept.
- Rejected larger inference-block batch benchmark:
  - batch 8: restarts changed to 7 / 87 segments, `trace_wall_s=99.982`
  - batch 4: interrupted after the same early quality-divergence pattern
  - both reverted.

## Deviations / Deferred Items

- Direct cached-block lookup did not become the default because it regressed
  wall time despite preserving the restart metric.
- Larger inference-block batching did not become the default because it changed
  trace decisions/restart metric.
- The analytic principal-axis method remains opt-in only; the quality-matching
  default remains `eigh`.
