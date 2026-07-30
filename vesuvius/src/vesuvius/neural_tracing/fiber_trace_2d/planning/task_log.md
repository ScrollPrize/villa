# Task Log: Hard Native Directions For Lasagna Fallback

## 2026-07-30 - Discovery

- The mixed VC3D helper protects successful native spans but currently chooses
  the first native span as the Lasagna reinitialization seed to propagate
  endpoint directions.
- Lasagna constructs directed continuation candidates when a neighbor direction
  is available, but candidate selectability ignores the computed direction dot
  products and selection is based on normal-alignment score.
- Sequential span processing means a fallback can be solved before a native
  neighbor, so not every native-adjacent CP receives even the optional
  continuation direction.
- The final stitched global solve fixes protected native points and CPs but has
  no endpoint-tangent constraint on adjacent Lasagna samples.
- Lasagna open extensions project their initial direction through the normal
  field, so a retained Lasagna tail can also rotate away from an adjoining
  native endpoint direction.

## Prior Staged Fix

- The preceding one-seed and extrapolation-distance reoptimization correction
  remains staged and is intentionally preserved. Its durable summary is in the
  2026-07-30 changelog before this task replaced the active task log.

## Planned Contract

- Every successful native span supplies exact endpoint tangents from its CP and
  immediate adjacent dense point.
- Every Lasagna span or retained tail on the opposite side receives the
  corresponding outgoing direction as a hard positive-ray constraint.
- Constraint coverage is derived before Lasagna runs and is independent of
  seed choice, solve order, candidate selection, and normal alignment.

## Independent Plan Review

- The review confirmed the manifold approach but required an explicit
  log-distance positive-ray retraction, Jacobians, numerical bounds, and clear
  per-block Ceres ownership.
- Replaced the initial span-pair key with an original-CP-index plus side key so
  the same API represents internal spans and open tails.
- Added minimum sample cardinality for one- and two-ended constraints, because
  a two-point span has no movable adjacent point and a three-point span cannot
  carry two different ray manifolds.
- Required every candidate solve, the `optimizeExistingLine` wrapper path, the
  final global solve, and final VC3D splicing to receive or validate the same
  constraints.
- Added explicit normal-transport bypass for the first constrained tail point,
  unsorted-index mapping, conflict validation, manifold algebra/Jacobian tests,
  and adversarial fresh/stored native geometry tests.

## Implementation Finding

- A successful fused native span may retain repeated endpoint samples. These
  do not define a direction, so constraint derivation uses the first distinct
  dense native point when walking inward from each CP. It fails explicitly if
  the native span contains no distinct point; it never substitutes a normal-
  derived direction.

## Validation Iteration

- The first expanded C++ test build failed only in new assertions because the
  available normalization helper is namespaced as
  `vc3d::fiber_slice::normalizedOrZero`. Qualified the test calls; production
  sources had compiled successfully in the same build.

## Implementation

- Added CP-index/side hard-direction inputs to the shared Lasagna line
  reinitializer and point-index/anchor/direction inputs to the existing-line
  optimizer.
- Added an anchored positive-ray Ceres manifold with log-distance updates,
  analytic Plus/Minus Jacobians, positive-radius bounds, conflict checks, and
  post-solve validation.
- Applied the hard direction to every eligible span candidate, two-ended spans,
  constructed tails, and the final stitched global solve. Protected native
  samples remain fixed.
- The VC3D mixed helper derives both endpoint tangents for every successful
  native span before Lasagna runs and no longer changes the Lasagna seed to
  communicate a direction.
- Added constrained side/vector diagnostics to the reinitialization report.

## Tests And Results

- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_lasagna_line_optimizer test_line_annotation_generated_views test_fiber_trace3d -j32`
  passed.
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_lasagna_line_optimizer`
  passed: 33 test cases.
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_line_annotation_generated_views`
  passed: 49 test cases.
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_fiber_trace3d`
  passed: 39 test cases.
- `cmake --build volume-cartographer/build --target VC3D -j32` passed.
- `git diff --check` passed.

## Explicit Test-Scope Deviation

- The private Ceres manifold is validated through end-to-end constrained
  solves, positive orientation checks, two-ended cardinality, final-solve
  persistence, and invalid/fixed/off-ray inputs. Its analytic Jacobians were
  not exposed as a public or test-only API solely for standalone finite-
  difference tests; adding such an API would expand the production interface
  without changing the requested behavior.

## User-Corrected Solver Design

- Real VC3D data exposed four Ceres failures during initial residual/Jacobian
  evaluation in the custom positive-ray manifold path. The surfaced fallback
  error incorrectly printed legacy continuation-reference dots even though
  those dots were no longer selection gates.
- The user clarified that a fiber-adjacent CP supplies exactly one rollout
  direction from that side and that the regular Lasagna Ceres solve should use
  an additional fixed proxy point to encode it.
- Removed the custom manifold, point-level ray API, Jacobians, and all related
  solver plumbing. Each constrained side now fixes the CP and one adjacent
  proxy point while the existing spacing and smoothness residuals optimize the
  remaining points.
- A constrained side creates one fiber-directed rollout and suppresses the
  normal and continuation rollout variants from that same side. Candidates
  originating from an unconstrained opposite side or existing interior geometry
  share the same fixed proxies.
- Candidate failure diagnostics now report `ceres::Solver::Summary::message`
  for each unusable solve and no longer present legacy direction dots as the
  cause.
- The first shared-rollout refactor build found the new preserve-direction flag
  on the generic rollout signature instead of the directed-rollout signature.
  Moved the flag to the correct existing helper; no solver behavior from that
  failed build was executed.
- The new diagnostic regression initially expected `Residual` with an uppercase
  first letter. Ceres reports `Initial residual and Jacobian evaluation failed.`;
  corrected the exact case-sensitive assertion after confirming the GUI-facing
  failure string contains that message for every rejected candidate.

## Corrected Design Validation

- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_lasagna_line_optimizer test_line_annotation_generated_views test_fiber_trace3d -j32`
  passed after the fixed-proxy correction.
- `test_lasagna_line_optimizer` passed 34 cases, including exact proxy
  positions after the final solve, constrained-side candidate suppression, and
  actual Ceres failure-message propagation.
- `test_line_annotation_generated_views` passed 49 cases.
- `test_fiber_trace3d` passed 39 cases.
- `cmake --build volume-cartographer/build --target VC3D -j32` passed. The
  existing Qt `-Wsfinae-incomplete` warnings remain unchanged.
- `git diff --check` and the production search for `SetManifold`,
  `AnchoredPositive`, and point-level ray constraint types passed with no
  remaining custom manifold code.

## Follow-up Findings

- `projectDirectionToNormalPlane` returns the normalized input when its plane
  projection is degenerate. For a normal-parallel input this preserves exactly
  the forbidden normal direction during rollout transport.
- Full reinitialization still builds, solves, and scores an `existing`
  candidate from the previous Lasagna span.
- A continuation direction from an already solved neighboring span currently
  creates an additional `continue-left` or `continue-right` candidate. Generic
  chord/normal rollouts from the same side remain eligible, so the propagated
  direction can lose candidate selection.
- Native hard directions are derived directly from protected dense native
  geometry and are not obtained from the old Lasagna line. Non-hard seed
  directions can still come from old adjacent line samples, and therefore can
  be wrong when that old line is wrong.

## Follow-up Correction

- Degenerate normal-plane projection now returns a deterministic tangent that
  is perpendicular to the sampled normal. It no longer preserves a
  normal-parallel input direction.
- Full reinitialization no longer constructs, solves, reports, or selects the
  previous line geometry as an `existing` candidate.
- Removed seed directions inferred from the previous line around each CP. An
  unconstrained seed span is initialized only by fresh left/right rollouts.
- A hard native direction or direction propagated from a newly solved neighbor
  is authoritative for that endpoint. It replaces that side's generic rollout
  instead of competing with it. A span with one authoritative endpoint has one
  candidate; a span with authoritative directions at both endpoints has one
  candidate from each endpoint.
- Removed the obsolete `continue-left` and `continue-right` report fields and
  line-probe columns.
- Added regressions for normal-parallel projection, removal of previous-line
  candidates, one-sided candidate suppression, two-sided hard directions, and
  solved-neighbor propagation.

## Follow-up Validation

- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_lasagna_line_optimizer test_line_annotation_generated_views test_fiber_trace3d -j32`
  passed.
- `test_lasagna_line_optimizer` passed 35 cases.
- `test_line_annotation_generated_views` passed 49 cases.
- `test_fiber_trace3d` passed 39 cases.
- `cmake --build volume-cartographer/build --target VC3D vc_lasagna_line_probe -j32`
  passed. The existing Qt `-Wsfinae-incomplete` warnings remain unchanged.
- `git diff --check`, `git diff --cached --check`, and the production search
  for old direction extractors, previous-line candidate helpers, and
  `continue-*` report fields passed.
