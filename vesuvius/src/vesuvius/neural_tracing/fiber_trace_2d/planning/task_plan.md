# Trace2CP Refined Strip Off-Strip Clipping Plan

## Implementation

- Update `build_trace2cp_refined_segment_source` so source-grid sampling keeps
  only valid traced points instead of raising when interior points are outside
  the current source strip.
- If the original start or target trace endpoint is clipped, use the first and
  last remaining valid trace points as the regenerated strip endpoints.
- Preserve strict validation for malformed/non-finite trace inputs and for
  refined traces with fewer than two valid source-grid samples after clipping.
- Preserve existing duplicate-point removal and endpoint extension behavior for
  the remaining valid trace section.
- Do not change native 3D tracing, scoring, candidate search, model inference,
  or the initial side/top strip rendering.

## Spec Update

- Update `planning/specs.md` so refined/regenerated Trace2CP strip
  visualization treats off-strip trace points as clipped display data, not as a
  fatal trace failure.

## Docs Updates

- Update durable planning docs only; no separate user-facing docs are needed
  for this narrow visualization robustness fix.

## Tests

- Add a focused regression test that a refined trace with an off-strip interior
  point still builds a refined segment source.
- Add a focused regression test that off-strip trace endpoints are clipped
  instead of causing a lost-CP-endpoint error.
- Keep a failure test for cases where clipping leaves fewer than two valid
  trace points.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run `git diff --check` over touched files.

## Non-Goals

- Do not silently ignore invalid trace formats or non-finite coordinates.
