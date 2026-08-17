# Plan: Lasagna-oriented replay failure threshold

## Behavior and shared implementation

1. Add one reusable replay-threshold evaluator to `vc_fiber_tracer`. Given an
   evaluator point and its already selected reference match in base coordinates,
   it samples the Lasagna normal at the matched reference point after converting
   that point to the sampler's working scale.
2. For a valid normalized local normal, decompose the base-coordinate error into
   absolute normal magnitude `dn` and tangent-plane magnitude `dt`. Compute the
   normal-equivalent threshold error as
   `sqrt(dn^2 + (dt / 4)^2)` and compare that value with the existing `--fail`
   threshold. This creates an ellipsoid with radii `T` normal and `4T`
   tangential. Normal sign has no effect.
3. If the local normal is missing, invalid, non-finite, or zero-length, use the
   old Euclidean error as the threshold error. Do not grant tangential relaxation
   without valid Lasagna evidence.
4. Keep `matchForwardPolylinePoint()` and its Euclidean closest-point selection
   unchanged. Store `euclidean_error_base_voxels`, optional
   `normal_error_base_voxels`, optional `tangential_error_base_voxels`,
   `threshold_error_base_voxels`, `threshold_error_ratio`, and
   `local_normal_valid` for every evaluated match. Remove the ambiguous
   unpublished `error_base_voxels`/`error_ratio` keys atomically.
5. The comparison remains strict: `threshold_error > T` fails, so exact pure
   normal error `T` and pure tangent-plane error `4T` are accepted. For `T=0`,
   exact zero error has ratio zero and any nonzero error fails with the finite
   maximum-double ratio used by the current replay code.

## Tracer integration

1. Make a normal sampler and its positive finite working-to-base scale explicit
   mandatory API arguments for both replay evaluators. Greedy replay reuses its
   trace Lasagna sampler and passes `traceToBaseScale`; the shared evaluator
   samples `matched_reference_point_base / normalWorkingToBaseScale`.
2. Extend fiberlet graph replay with the canonical Lasagna normal sampler and
   its working-to-base scale. Use the shared evaluator for every route sample.
3. Apply the same anisotropic decision to fiberlet seed selection. Retain a
   cheap inclusive Euclidean `4T` broad-phase before sampling normals (including
   the exact-zero-only behavior at `T=0`), then rank usable seeds by reference
   arc, normalized threshold error, and node index so seed ordering uses the
   same metric as acceptance. The initial seed match stores the full shared
   measurement.
4. Keep graph costs, beam routing, trace losses, reset advancement, reference
   matching, and all non-distance failure reasons unchanged.

## Output and CLI

1. Clarify that `--fail N` is the normal-direction radius in base voxels and
   that the fixed tangent-plane radius is `4N`.
2. Persist one authoritative aggregate threshold descriptor containing shape,
   normal radius, tangential factor/radius, comparison, and invalid-normal
   policy. Generate nested greedy/fiberlet descriptors from the same shared
   serializer and validate that both engine thresholds equal the root value.
3. Emit all explicit component diagnostics for greedy/fiberlet matches and
   failures, and include them in command-line distance-failure output.
   Graph-exhausted failures preserve their existing last-match diagnostics;
   failures without an evaluated point retain null diagnostics.
4. Keep current replay artifact versions because this is an unpublished
   diagnostic format and update its strict writer atomically; add no repair or
   compatibility path.

## Tests and validation

1. Unit-test the shared evaluator with pure normal, pure tangential, mixed,
   reversed-normal, scaled-sampler-coordinate, and invalid-normal inputs.
2. Update greedy replay tests to supply a normal sampler and verify a displacement
   between `T` and `4T` survives tangentially but fails normally.
3. Update fiberlet graph replay tests for the same decisions, including a seed
   that is accepted only by the tangent-plane allowance.
4. Verify serialized matches, failures, and threshold descriptors contain
   finite consistent component values and normalized ratios. Strict producer
   validation must recompute `dn^2 + dt^2 ~= euclidean^2`, the ellipse formula,
   the ratio and invalid-normal fallback, and prove distance failures exactly
   copy their terminal match measurement.
5. Test exact-boundary and zero-threshold behavior, invalid, zero, NaN and
   infinite normals, and clarify that tangent means the full 2D plane normal to
   the Lasagna surface normal rather than the learned fiber tangent.
6. Build `test_fiber_trace3d`, `test_fiberlet_paths`, `test_fiber_replay`, and
   `vc_fiberlets` with `-j32`; run the three focused suites and `git diff --check`.
7. Run a bounded Paris4 replay and compare failure counts/output against the
   prior isotropic run, recording the exact command and result.

## Spec update

- Change the replay failure-distance contract from an isotropic Euclidean ball
  to the Lasagna-normal ellipsoid above, including invalid-normal fallback,
  unchanged matching, exact boundary behavior, full tangent-plane terminology,
  seed behavior, strict producer validation, and persisted diagnostics.

## Documentation update

- Update `volume-cartographer/docs/fiberlets.md` with the formula, `--fail`
  semantics, normal sampling coordinate conversion, seed behavior, and output
  fields.

## Changelog update

- Record the shared anisotropic failure evaluator and its use by both replay
  tracers.
