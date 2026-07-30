# Hard Native Directions For Lasagna Fallback Plan

## 1. Shared Constraint Contract

- Add a structured Lasagna endpoint-direction constraint keyed by original
  control-point index and side (`before` or `after`). The direction points from
  that CP into the generated Lasagna geometry on that side. This single
  contract represents internal spans and both open tails.
- Directions point from the constrained CP into the Lasagna span. Normalize
  and validate them at the shared API boundary; reject non-finite or
  degenerate directions, invalid original indices, duplicate/conflicting
  `(control, side)` entries, and constraints targeting protected native sides.
- Map original indices through the existing stable control sort explicitly;
  do not rely on caller ordering or oriented protected-pair matching.
- Keep this input separate from protected ranges: protected ranges freeze
  native geometry, while endpoint constraints govern adjacent Lasagna
  geometry.

## 2. Exact Solver Constraint

- Use only the existing point-parameter Ceres problem and its spacing,
  tangent-straightness, normal-straightness, and normal-alignment residuals.
- For each constrained CP side, create the adjacent proxy at
  `CP + normalized_direction * proxy_distance` and add its sample index to the
  existing fixed-point set alongside the CP.
- Use a deterministic positive proxy distance bounded by the configured
  Lasagna segment length and available CP-to-CP chord length.
- Carry those fixed proxy indices through each per-span solve and the final
  stitched global solve. Do not add a manifold, custom parameterization, or
  high-weight direction residual.

## 3. Deterministic Span And Tail Construction

- Extend `reinitializeAndOptimizeExistingLine` with the structured hard
  constraints.
- For a constrained fallback endpoint, initialize the adjacent sample directly
  on the required direction and fix it. Generate exactly one rollout from that
  CP side using the fiber direction; do not also generate normal/chord or
  continuation variants from the constrained side.
- Other candidates may originate from an unconstrained opposite CP, but full
  reinitialization never submits existing interior geometry as a candidate.
- When a solved neighboring span supplies a continuation direction, use it as
  the span's sole rollout candidate unless the opposite endpoint also has an
  authoritative solved-neighbor direction. In that two-sided case, generate
  exactly one rollout from each authoritative endpoint.
- Support simultaneous left and right hard constraints for a fallback span
  between two native spans.
- Guarantee distinct movable samples: at least three span points for one hard
  endpoint and at least four for two. Densify deterministically before solving
  or fail explicitly; two constrained sides must not share one proxy point.
- Apply outer-CP constraints while constructing Lasagna open extensions, so a
  retained fallback tail starts on the exact native continuation ray. The
  first generated tail point bypasses normal transport and is initialized
  directly on the fiber direction.
- Translate span-local and tail-local proxy indices to final stitched indices
  and add them to the final solve's existing fixed-point list.
- Leave local non-reinitialization Lasagna optimization and unconstrained tail
  construction behaviorally unchanged.
- Make tangent-plane projection total: if removal of the normal component is
  degenerate, choose a deterministic perpendicular tangent rather than
  returning the input direction.

## 4. VC3D Constraint Derivation

- In `optimizeFiberWithNativeFallback`, derive endpoint directions from each
  successful native dense span after conversion to base coordinates:
  `normalize(first_distinct_native_point - CP)`, walking inward from the CP so
  repeated fused endpoint samples do not create a zero direction.
- For each neighboring unprotected Lasagna span, pass the negated direction as
  its outgoing hard constraint. If the native span touches an outer CP, pass
  the same continuation rule to the possible Lasagna fallback tail.
- Derive constraints for every native span before invoking Lasagna. Do not use
  the first native span or solve order to deliver directions.
- Remove the mixed helper's native-seed direction workaround. Seed choice may
  remain an optimization detail, but it must have no effect on which hard
  constraints are applied.

## 5. Diagnostics And Failure Semantics

- Report the constrained side and normalized direction in span diagnostics.
- Verify the fixed proxies remain exact after each solve and after final
  stitching. Treat violations as failures rather than silently returning
  unconstrained geometry.
- When no candidate is usable, surface each Ceres summary message. Do not print
  legacy continuation-direction dots as though they caused rejection.
- Reject a successful native span with no finite, distinct endpoint-neighbor
  sample relative to its CP.

## 6. Tests

- Add Lasagna-level adversarial-normal tests for native-left/fallback-right,
  fallback-left/native-right, native/fallback/native dual constraints, multiple
  separated protected spans, constrained open tails, final-global persistence,
  and invalid/degenerate constraints.
- Add fixed-proxy position, minimum-cardinality, unsorted-control remap, and
  duplicate/conflict tests.
- Add a normal-parallel transport regression proving the rollout turns onto a
  deterministic tangent instead of continuing along the normal.
- Assert full reinitialization reports no `existing` candidate and that a
  supplied neighbor continuation replaces, rather than supplements, the
  corresponding generic side rollout.
- Add a VC3D mixed-helper regression deriving constraints from actual native
  dense points deliberately non-collinear with the chord and adversarial
  normal, verifying every adjacent fallback/tail direction for both newly
  traced and already-protected native spans.
- Retain existing bit-exact protected-span and unconstrained Lasagna tests.
- Build VC3D and affected tests with `-j32`; run
  `test_lasagna_line_optimizer`, `test_line_annotation_generated_views`, and
  `test_fiber_trace3d`.

## 7. Spec Update

- Replace candidate/seed-based native-neighbor wording with the exact hard
  endpoint-direction contract.
- Specify the CP/adjacent-native-point formula, outgoing sign, all-CP coverage,
  dual-ended fallback behavior, tail behavior, validation, and explicit
  failure semantics.

## 8. Docs Update

- Update `docs/code_structure.md` to describe constraint derivation in the
  mixed helper and fixed-proxy enforcement across the Lasagna span and final
  solves.
- Update the VC3D line-annotation fiber documentation if it currently describes
  fallback direction behavior.

## 9. Changelog

- Add a 2026-07-30 entry for exact native endpoint tangents on all adjacent
  Lasagna fallback spans and tails.

## 10. Review Risks

- Verify span-index remapping after control sorting and stitched-index remapping
  after open-tail insertion.
- Verify a constrained side generates only its fiber-directed rollout and that
  every remaining candidate shares the same fixed proxy.
- Verify constrained endpoints remain exact after the final global solve and
  native samples remain bit-exact.

## 11. Workflow Records

- Update `planning/status.md` incrementally during implementation and
  validation.
- Replace and maintain `planning/task_log.md` with review findings, deviations,
  commands, failures, and final test results.
