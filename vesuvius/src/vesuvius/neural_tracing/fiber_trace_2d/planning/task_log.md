# Regular Combined Trace2CP Candidate-Point Direction Scoring Task Log

## Implementation Notes

- Updated `_trace_combined_direction_line_to_target` so regular combined
  tracing samples the candidate-point direction for each viable candidate.
- Direction loss now averages current-point direction disagreement and
  candidate-point direction disagreement, matching the z-search scorer.
- Candidate points with missing direction samples are marked invalid.
- Added a regression where candidate-point direction evidence changes the first
  selected step away from the current-only straight candidate.
- Updated the combined Trace2CP spec and code-structure docs.

## Deviations

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - Passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'trace2cp_combined_candidate_point_direction or trace2cp_combined_direction_only_selects_center_candidate or trace2cp_combined_embedding_weight_can_choose_off_axis_candidate'`
  - Passed: 3 passed, 167 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Passed: 170 passed.
- `git diff --check -- <touched files>`
  - Passed.
