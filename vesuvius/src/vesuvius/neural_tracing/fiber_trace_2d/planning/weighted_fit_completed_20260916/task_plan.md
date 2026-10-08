# Weighted-fit anchor experiment plan

1. Archive completed anchor benchmark planning. Inspect existing cell fitter, batched sampler, metadata and support/NMS stages.
2. Add opt-in weighted-fit mode to FiberAnchorConfig and shared cell fitter. Reuse owned-cell observations, presence validity/floor, tensor eigensolver, sampler and extraction/report/cache plumbing. Fixed Gaussian sigma2 and local sigma1.5 prediction voxels, damping0.5. Initialize at cell center with principal direction of fixed-Gaussian presence^gamma tensor. For iterations, compute fixedGaussian*localGaussian*presence^gamma*abs(dot)^q, weighted centroid and tensor direction, damp position. Final support uses fixedGaussian*presence*cos² / sum fixedGaussian, same minimumAlignedSupport gate. No fabricated anchors for empty/unsupported cells; no cross-cell NMS in experimental mode, at most one retained anchor per owned cell. Support stays inside cell (convex centroid); no gradient computation or broad fitting halo needed.
3. Experimental fields enter construction identity only when enabled, preserving default legacy cache fingerprints. Expose strict anchor_fit config in native eval. Both anchor-only and subsequent fiberlet modes use same fitter. Canonical default untouched.
4. Add optional fitted axis to anchor coverage inputs and compare absolute cosine angle with GT tangent. Report per-anchor angle and per-piece/pooled median/P95, fractionswithin10/20/30degrees. Placement only remains unchanged.
5. Reusable approved benchmark-full sweep orchestrator: baseline plus20unique experimental combinations (zero iterations independent of q, one row per gamma). Same200pieces, model/predictions/ellipse. Retain all reports and summary matrix; no full graph reruns in initial placement sweep. Can use shared inference caches and demand-only GPU manager. Reuse per-variant completed outputs if matching workload/binary provenance; don't overwrite.
6. Tests: sign invariance, one-per-cell, empty/invalid support, deterministic orientation ties, iteration0, finite/within-cell positions, stronger cluster attraction; cache separation/defaultidentity; angle metric; native integration no paths. Approved build/tests then sweep and report all rows and baseline.

## Spec update
Opt-in owned-cell weighted fit formula, fixed sigmas/damping/support threshold, no NMS, identity isolation; defaults unchanged. Placement direction metric modulo sign.

## Docs updates
Algorithm/commands/sweep results, task status/log, changelog; retain previous benchmark results. Disclose owned-cell-only support and no adaptive covariance in this first experiment.

## Review
Independent plan review before implementation.

Review passed: denominator covers all owned voxels; unusable samples contribute
zero numerator. Degenerate initialization uses strongest observed direction
(stable observation-order ties), later degenerate tensors retain previous axis;
zero local mass retains previous data-supported state. No NMS/context expansion.
Support definition is fixed-Gaussian presence*cos², not a local presence test at
the new position. The comparison changes fitter, multiplicity and NMS together.
Stored fitted axes (not resampled prediction direction) feed angle statistics.
