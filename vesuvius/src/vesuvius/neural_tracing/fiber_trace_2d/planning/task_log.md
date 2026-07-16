# Native 3D Trace2CP Product Candidate Scoring Task Log

Current task log only.

## Planning

- Replaced the previous requested-level sampler task log with the native 3D
  Trace2CP product candidate-scoring task.
- Planned a local scorer/tool update only; no changes to model inference,
  block loading, trace fusion, strip rendering, or training losses.

## Implementation

- Updated native 3D Trace2CP candidate scoring to maximize:
  `dot(current_dir, step_dir) * dot(candidate_dir, step_dir) * candidate_presence`.
- Changed the first native 3D search step to use the adjacent CP-local
  fiber-line tangent in the direction of the target CP's line index. This is
  not the straight CP-to-CP chord and not the sampled model direction at the
  start CP. Subsequent steps still use sampled model directions aligned to the
  previous accepted trace direction.
- The current sampled axis is aligned to the candidate step direction before
  scoring, and the candidate sampled axis is aligned to the candidate step
  direction before scoring, preserving sign-ambiguous direction semantics.
- Invalid candidate points remain rejected by assigning them `-inf` score.
- Removed obsolete native 3D Trace2CP additive candidate-selection CLI/config
  knobs `--direction-weight` and `--presence-weight`.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d_trace2cp"`
  - Passed: `22 passed, 39 deselected`.
- `PYTHONPATH=vesuvius/src:. python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  - Passed.
- `git diff --check -- <touched files>`
  - Passed.

## Deviations Or Deferrals

- None.
