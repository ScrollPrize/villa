# Task Log: Lasagna-Fallback Segment Metadata Cleanup

## 2026-07-31 - Findings

- `meeting_error_ratio` is an observed diagnostic computed as meeting gap over
  combined partial trace length. It can validly exceed one for a failed native
  attempt.
- The loader incorrectly applied an at-most-one constraint to this observed
  diagnostic. Only the configured `meeting_accept_max_error_ratio` is a bounded
  acceptance fraction.
- Rejected native meeting diagnostics describe the discarded native fusion,
  while the stored segment geometry is the Lasagna fallback. Failure code and
  detail are the appropriate persisted diagnostics for that outcome.

## Review

- Accepted native segments continue to own their meeting diagnostics and use
  them for strip display; fallback segments continue to own failure code/detail.
- Segment protection depends on `outcome`, not on meeting diagnostics.
- No separate review agent is used because higher-priority collaboration
  instructions prohibit delegation unless explicitly requested. The plan was
  reviewed directly against the current specs and implementation.

## Deviations

- None.

## Implementation

- Native trace result conversion now retains meeting error/ratio/source only
  for accepted results. Rejected attempts retain failure code/detail.
- VC3D serialization forcibly clears fallback meeting diagnostics, and VC3D
  loading skips those JSON values before type/range validation.
- The shared native C++ validator and Python parser apply the same
  outcome-specific rule. The observed accepted ratio is non-negative but not
  capped; the configured acceptance ratio remains capped at one.
- The sync/merge validator accepts earlier fallback records without inspecting
  their meeting values, and every merge/link-refresh output canonicalizes those
  fields to null/null/empty source.
- Specs and implementation docs now distinguish accepted meeting diagnostics
  from fallback failure diagnostics.

## Validation

- `test_line_annotation_generated_views`: all 51 cases passed after a `-j32`
  build. Coverage includes VC3D construction/serialization/loading and the
  shared native C++ reader.
- `vesuvius/tests/neural_tracing/test_fiber_trace.py`: all 52 cases passed with
  repository-local imports and third-party pytest plugin autoload disabled.
- `volume-cartographer/scripts/tests/test_fiber_merge.py`: all 57 cases passed
  with third-party pytest plugin autoload disabled.
- `python -m py_compile` passed for both changed Python modules.
- Built production `VC3D` with `-j32` successfully.
- `git diff --check` passed. Final review confirmed fallback meeting fields are
  skipped before parsing in every reader and cleared by both writing paths,
  while accepted diagnostics and fallback failure reasons remain intact.
- All tracked task changes were staged; unrelated untracked artifacts were not
  touched.
