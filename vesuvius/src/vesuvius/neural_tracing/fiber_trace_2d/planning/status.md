# Native Diagnostic Refresh And Edge Extrapolation Status

## Planning

- [x] Reproduce the diagnostic-clearing call sequence by inspection.
- [x] Locate one-way invalid-candidate termination and VC3D tail fallback.
- [x] Replace task, plan, status, and task log for the follow-up.
- [x] Review the plan against the active specs and implementation.

## Implementation

- [x] Repopulate current span diagnostics after branch-overlay refresh.
- [x] Preserve accepted spans during same-mode Reoptimize.
- [x] Retain the last valid one-way path on invalid candidates.
- [x] Accept invalid-candidate extrapolation as a truncated native tail.
- [x] Add focused core and VC3D regressions.

## Validation And Documentation

- [x] Run focused tests and build VC3D with `-j32`.
- [x] Update specs, docs, changelog, and task log.
- [x] Review and stage the follow-up with the existing task changes.
