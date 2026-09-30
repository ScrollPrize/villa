# Plan: anisotropic Gaussian fiber-presence supervision

1. Add explicit, validated target settings for legacy binary versus
   anisotropic Gaussian presence supervision. Treat the configured presence
   radius as Gaussian thickness sigma; configure width/thickness ratio and a
   finite sigma cutoff.
2. Extend the existing compact segment target metadata with local Lasagna
   surface normals through every sample/shared-slot/batch/slice/prepass path.
   Reuse the native normal sampler and transform normal covectors with the
   local augmentation Jacobian inverse-transpose, including smooth deformation;
   reject invalid/singular samples through the existing deterministic retry.
3. Materialize the soft tube on the training device from exact point-to-segment
   distances in the local tangent / in-sheet transverse / surface-normal frame.
   Evaluate output index `j` at input coordinate `j*output_stride`, merge
   overlapping segment contributions by maximum probability. Assign every
   nonzero-support voxel the tangent of its winning segment and use a
   Gaussian-weighted direction mean. Bound temporaries by cutoff-expanded
   per-segment boxes rather than segment-by-full-volume tensors.
4. Generalize ordinary presence loss statistics to apply soft-target BCE over
   the nonzero tube and zero-target BCE outside it, retaining per-patch
   positive/negative balancing. Update the target-only macro-batch prepass to
   compute exactly the same nonempty support/background group counts.
5. Add focused CPU tests for orientation, 4:1 sigma ratio, cutoff, overlap,
   augmentation metadata, loss gradients, microbatch equivalence, and unchanged
   legacy targets. Run the relevant existing target/refinement/single-backward
   tests through the approved test launcher.
6. Benchmark target materialization and an end-to-end canonical high-resolution
   microbatch before/after, recording the exact command/input and distribution
   summary without bypassing the cooperative GPU manager.
7. Add a production config based on the high-resolution refinement config. Use
   `training.initialize_from` for the best EMA-0.999 weights (EMA snapshots are
   not optimizer-resumable), remove the inherited `training.resume`, preserve
   active online refinement, and record source path/hash/new run name. Launch
   through the approved cooperative experiment runner on physical GPUs 0-5 and
   verify manager allocation plus initial training/test progress. Permit
   automatic startup calibration on an explicit nonempty managed GPU subset.
8. Stop the initial run cleanly, then resume its latest full checkpoint with
   thickness sigma halved from 2 to 1 voxel (and therefore width sigma from 8
   to 4 voxels). Give the continuation a distinct run name, request physical
   GPUs 0-6, and verify restored update/optimizer/refinement state and progress.
9. Represent distribution orientation as a sign-invariant rotation around the
   local fiber tangent, relative to the native Lasagna normal sampled at the
   immutable baseline material position. Persist `(cos(2 theta), sin(2 theta))`
   on the same material-arc lattice as each refined curve so interpolation is
   continuous modulo pi. Include independent test patches and archives; restore
   old geometry-only archives with the explicit `(1,0)` zero-rotation state.
10. During refinement, use fixed symmetric probes at the same physical radius
    along the candidate width and normal axes, scoring width-axis presence
    minus normal-axis presence through the exact nonlinear base-to-patch
    transform.
    Estimate the angular derivative by central finite differences, reject
    incomplete/out-of-patch probe sets, and apply an explicit capped angular
    step plus a weak periodic `sin²(theta)` tether to zero rotation. Convert
    configured output-voxel sigmas through each probe's local augmentation
    Jacobian before evaluating exact transformed query positions.
    Commit curve and orientation together under the existing optimistic lock;
    stale proposals update neither. Reinterpolate doubled-angle vectors on the
    returned material coordinates whenever curve geometry is resampled.
11. Apply the interpolated refined rotation to native segment normals before
    the existing augmentation covector transform and Gaussian rasterization.
    Keep binary targets and refinement without Gaussian orientation unchanged.
    Add archive compatibility, cap/tether, transform, target-orientation, and
    stale-commit regression tests before launching the half-sigma continuation.
    Preserve material identities through clipping so every emitted segment uses
    the matching persisted orientation and immutable baseline normal.

## Spec update

No `planning/spec.md` exists in this checkout. The behavioral contract is:
legacy binary targets remain the default; Gaussian mode requires dense labels
and native valid Lasagna normals and explicitly records thickness sigma, width
ratio, and cutoff in the training config/checkpoint. The field is
`exp(-0.5*(d_normal^2/sigma_normal^2 + d_width^2/sigma_width^2))`, with finite
segment caps using width sigma, maximum composition across segments, and zero
outside the cutoff. Normal sign is irrelevant.

## Docs updates

Document Gaussian tube geometry, soft-loss semantics, configuration, native
normal reuse, and the exact production run in `docs/training_collection_20260909.md`
and/or `docs/training_refinement.md` as appropriate.

## Testing

Use deterministic synthetic segments/normals for numerical target assertions,
then shared transport, stride-1/4, augmentation transform, target-prepass count,
microbatch-equivalence, and existing CPU target/arc/refinement/single-backward
suites. Include rotation/reflection/scale/smooth-warp normal transforms,
sign-invariance, support tails, clipped edges, complete direction support, and
Gaussian-weighted direction normalization. Start the real managed run only
after those pass.

## Changelog

Add one concise 2026-09-29 entry describing opt-in anisotropic Gaussian
presence supervision and the new high-resolution refinement continuation.
