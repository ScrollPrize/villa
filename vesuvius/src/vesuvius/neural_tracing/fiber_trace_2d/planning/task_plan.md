# Local Fiber3D embedding implementation plan

1. Discover/reuse normal sampling and tracer exclusion geometry. Require explicit
   normals for embedding-enabled training; never silently change to a spherical
   exclusion. Evaluate exclusion in original-volume coordinates using existing
   inverse augmentation maps, not a rotated-radius approximation. Invalid normal
   data is an explicit error for this training feature.
   Match the Euclidean-nearest continuous GT segment point first, sample its
   surface normal there, then exclude threshold error <=20 (boundary included).
   The tangent plane is the full plane perpendicular to the surface normal.
2. Add opt-in embedding channels to the regular U-Net output head, preserving
   legacy prediction channels and their sigmoid. Embeddings are signed and L2
   normalized when sampled. Keep checkpoint configuration explicit. Conditioned
   decoder combinations require explicit validation rather than reinterpretation.
3. Carry embedding sampling geometry through raw/sample/shared-pool collation,
   device transfer, batch splitting, and deferred GPU augmentation. Reuse the
   existing GT trace and normals loader; bound exclusion work and temporary memory.
4. Sample 65,536 pairs at width 128, linearly scaled by maximum patch dimension,
   equally positive/negative, with replacement. GT points are arc-length sampled.
   Use per-stream deterministic RNG, independent of rank/microbatch placement.
   Negative output-grid weights are detached presence after exclusion. No uniform
   fallback for zero support; report zero negative support. Use separate sums/counts
   with equal component weights and configurable overall embedding weight.
5. Integrate both ordinary and bounded/cooperative training, evaluation and scalar
   logging; end-effective-batch reductions only. Preserve inactive feature behavior.
6. Test geometry scale/rotation/warp and 20/80 boundary; pair counts, sampling
   probabilities and zero support; gradients, model/checkpoint compatibility,
   sample transport, microbatch equivalence, and CPU training smoke tests. Measure
   pair-loss overhead without claiming end-to-end GPU throughput.

## Implementation decisions

- Default model forward keeps the legacy 7*K tensor. An explicit training
  forward option returns prediction and embedding tensors separately; branch
  inference must never reinterpret the embedding channels as extra branches.
- Five numerator-gradient components when embeddings are enabled, with separate
  positive/negative support counts and no per-microbatch communication.
- Build a bounded output-grid eligibility tensor in loader workers using inverse
  geometric maps and shared normal sampling. No dense pairwise embedding matrix.
- The native C++ replay metric has no Python binding: the vectorized NumPy
  training evaluator will be a separate backend of its documented formula,
  checked against analytic boundary cases. This is an explicit cross-language
  implementation deviation, not a new ellipse or fallback convention.

## Spec update

Document opt-in channels, normals requirement, original-coordinate exclusion,
pair budget, detached sampling, RNG, equal means, zero-support behavior, and
additional effective-batch sufficient statistics. Preserve existing defaults.

## Docs updates

Add embedding configuration and objective documentation, test commands/results,
and explicit unsupported combinations if any. Update task/status/log and changelog.

## Review

Independent agent review required before implementation. Any departures from
shared geometry or planned coverage must be recorded before proceeding.
