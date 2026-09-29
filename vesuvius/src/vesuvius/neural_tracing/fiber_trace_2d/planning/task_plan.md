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
