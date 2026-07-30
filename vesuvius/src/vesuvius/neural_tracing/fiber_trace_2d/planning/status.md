# Hard Native Directions For Lasagna Fallback Status

## Planning

- [x] Read the user request and nested repository instructions.
- [x] Trace current direction derivation, candidate selection, span ordering,
  tail growth, and final global solve.
- [x] Replace the active task, plan, status, and task log.
- [x] Obtain independent-agent review against task, specs, and current solver.
- [x] Incorporate review findings.

## Implementation

- [x] Add shared structured hard endpoint constraints.
- [x] Add exact fixed-proxy direction constraints using the regular solver.
- [x] Enforce constraints in span initialization and span solves.
- [x] Enforce constraints in open tails and the final global solve.
- [x] Derive constraints for every native endpoint in the VC3D mixed helper.
- [x] Remove candidate/seed-order direction dependence.
- [x] Remove the custom manifold and use fixed proxy points in regular Ceres.
- [x] Surface actual Ceres messages instead of legacy direction dots.

## Validation And Documentation

- [x] Add adversarial, ordering, dual-ended, tail, and invalid-input tests.
- [x] Rebuild affected tests and VC3D with `-j32` after proxy correction.
- [x] Rerun focused regression suites after proxy correction.
- [x] Update specs, docs, changelog, and task log.
- [x] Perform final diff and consistency review.

## Follow-up Correction

- [x] Make degenerate tangent-plane projection perpendicular to the normal.
- [x] Remove previous-line candidates from full reinitialization.
- [x] Make solved-neighbor continuation replace same-side generic rollout.
- [x] Add focused projection and candidate-suppression regressions.
- [x] Rebuild and rerun affected suites and VC3D with `-j32`.
- [x] Update specifications, docs, changelog, and task log.
