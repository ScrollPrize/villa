# Hard Native Directions For Lasagna Fallback

## User Request

- When a native fiber-traced span adjoins a Lasagna fallback at a CP, the
  native span's endpoint direction must be a hard direction constraint for the
  Lasagna geometry.
- The direction is derived directly from the CP and the next dense point in the
  successfully native-traced span.
- Apply this rule at every CP that has native-traced geometry on one side and
  Lasagna-generated geometry on the other side.
- Fiber-derived directions supersede all normal-, chord-, existing-line-, and
  candidate-derived starting directions. They are constraints, not candidates
  or weighted preferences.

## Direction Semantics

- At a native endpoint, `into_native = normalize(next_native_point - CP)`.
- Lasagna on the opposite side must leave the CP along `-into_native`, thereby
  continuing the same tangent through the CP.
- The rule applies independently to both endpoints of every successful native
  span, including a Lasagna span constrained at both ends.
- If native extrapolation fails and the corresponding Lasagna open tail is
  retained, the tail must obey the same hard continuation direction.

## Correctness Constraints

- The constraint must remain exact through span initialization, span solving,
  stitching, and the final global Lasagna solve.
- Do not implement this as a large residual weight, candidate score, seed-span
  ordering rule, or post-solve correction.
- Preserve successful native span samples bit-exactly.
- Use the existing Lasagna/Ceres point optimization and smoothness residuals.
  Add one fixed adjacent proxy point in the fiber-derived direction for each
  constrained side; do not add a custom manifold or solver mechanism.
- A fiber-constrained CP supplies exactly one rollout direction from that CP.
  Do not also generate normal-, chord-, existing-, or continuation-derived
  starting directions from the same side.
- Invalid or degenerate native endpoint geometry is an explicit task failure;
  it must not silently revert to a normal-derived direction.

## Scope

- Shared Lasagna reinitialization and global solve constraint plumbing.
- VC3D mixed native/Lasagna construction of hard endpoint constraints.
- Internal fallback spans and retained Lasagna open tails.
- Focused C++ regression tests, specifications, code documentation, and
  changelog.

## Out Of Scope

- Native tracer scoring, fusion, intersection, or acceptance changes.
- Changing ordinary all-Lasagna behavior when no native hard constraint is
  supplied.
- Persisting redundant direction metadata; directions are derived from the
  authoritative native dense geometry each time.
