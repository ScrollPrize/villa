# Native Diagnostic Refresh And Edge-Truncated Extrapolation Plan

## 1. Diagnostic Refresh

- Keep `setGeneratedBranchOverlayData()` conservative: it may clear stale labels
  while replacing control geometry.
- In the controller's central `refreshBranchLineViews()` path, immediately
  repopulate labels from `generatedSpanAlignmentMetricsForSession()` after the
  branch/control overlay replacement.
- Preserve accepted segment metadata when the user presses Reoptimize while
  already in Lasagna mode so the existing protected-span reinitializer can keep
  it. Retain metadata clearing on an explicit mode transition to Lasagna.

## 2. Partial Extrapolation Retention

- Preserve the best beam from the last nonempty generation in the shared
  one-way tracer.
- When candidate pruning produces an empty frontier, return that last valid
  beam with `reason=no_valid_candidates` rather than returning the initial CP.
- Do not mark the requested distance plane as reached.

## 3. VC3D Tail Selection

- Accept a nonempty extrapolation path as native when it reached the distance
  plane or terminated with `no_valid_candidates`.
- Convert and splice a retained path of at least two points exactly as for a
  completed native tail.
- Continue counting all other incomplete or exceptional results as Lasagna
  fallback extrapolations.

## 4. Tests

- Add a core tracer regression proving invalid directions after one valid step
  return the partial path and `no_valid_candidates`.
- Add a VC3D fiber-mode regression proving one completed tail and one
  edge-truncated tail are both native and that the truncated endpoint is the
  last valid sample rather than the Lasagna endpoint.
- Run `test_fiber_trace3d`, `test_line_annotation_generated_views`, and build
  `VC3D` with `-j32`.

## 5. Spec Update

- Specify diagnostic repopulation after generated branch refresh and protection
  of accepted spans during same-mode Reoptimize.
- Specify `no_valid_candidates` as successful edge truncation for extrapolation
  only, without changing CP-pair or whole-fiber acceptance.

## 6. Docs Update

- Update the VC3D section of `docs/code_structure.md` and
  `volume-cartographer/docs/line_annotation_fibers.md` with refresh and
  edge-truncation behavior.

## 7. Changelog

- Add a concise 2026-07-30 bug-fix entry.

## 8. Review Risks

- Do not reinterpret `no_valid_candidates` as success for CP-pair tracing.
- Do not accept `max_step_factor` as edge truncation.
- Do not retain stale label geometry after CP edits; repopulate labels from the
  current session after replacing control overlays.
- Do not clear accepted metadata on ordinary Reoptimize.
