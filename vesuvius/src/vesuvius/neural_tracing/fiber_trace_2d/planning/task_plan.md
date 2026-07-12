# Regular Combined Trace2CP Candidate-Point Direction Scoring Plan

## Scope

- Change only the regular non-z combined Trace2CP scorer.
- Keep z-search scoring, embedding terms, public `trace2cp_error`, training
  losses, and checkpoint selection unchanged.

## Implementation

- In `_trace_combined_direction_line_to_target`, after a candidate point passes
  margin and embedding checks, sample the candidate-point direction with
  `_sample_trace2cp_combined_direction`.
- Use the same direction source as the current trace mode:
  direct `direction_xy` for reference mode, or `tta_fields` for median-TTA
  combined mode.
- Compute direction loss as the average of:
  - `1 - dot(candidate_unit, current_oriented_direction)`;
  - `1 - dot(candidate_unit, candidate_point_oriented_direction)`.
- Treat candidates with missing candidate-point direction as invalid.

## Spec Update

- Update combined Trace2CP scoring text so the direction term is defined as
  current-point plus candidate-point direction evidence for all combined
  tracing, not only z-search.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP combined-scoring description.
- Update `planning/changelog.md`.
- Replace `planning/task_log.md` with this task's notes and validation.

## Tests

- Add a non-z combined regression where current-only scoring would choose the
  straight candidate but candidate-point direction evidence chooses an off-axis
  candidate.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run syntax and diff checks.
