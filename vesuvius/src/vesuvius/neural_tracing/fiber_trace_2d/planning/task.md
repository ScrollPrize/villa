# Task: staged fiberlet anchor acceleration

Continue accelerating anchor and fiberlet extraction from commit `73fe64e09`.
Implement the remaining optimization options one at a time. After every option,
run the canonical 5,000-base-voxel replay, report wall/CPU phase measurements,
anchor and fiberlet populations, DP work, and replay failures, then stop for
user review before beginning the next option.

The ordered options are:

1. Build compact normalized observations once per extraction tile and let each
   cell reference them instead of rebuilding expanded observation records.
2. Reuse robust-proposal assignments, Gaussian/alignment state, baseline
   objective, and final membership across adjacent fitting phases.
3. Batch peak candidates or use a contiguous spatial index so repeated peak
   responses share observation loads without the locality regression of the
   rejected linked-bin experiment.
4. Evaluate reducing robust refinement from two passes to one, with explicit
   visual and replay-quality review.
5. Reduce duplicated prediction sampling across overlapping anchor-tile halos
   while preserving enough independent jobs for 32-worker load balance.

Small floating-point differences are acceptable. Persistent geometry and file
formats must remain valid, and each checkpoint must retain acceptable replay
quality before it can become the next baseline.

The current continuation targets the remaining fiberlet dynamic-programming
cost. Measure separately: caching decoded static data for every retained node,
reusing each reached node's outgoing-edge descriptors across incoming states,
fully pre-generating all retained-node transitions, and compacting DP state by
deriving incoming edges and predecessor nodes from the node/transition key.
Retain the fastest acceptable composition and record rejected variants.

The next measured continuation is lazy node scoring materialization. Keep the
global deduplicated source-voxel sampling and immutable scoring index, but
interpolate an interior node only when search first requests it. Cache the
result by candidate-local node index and preserve the existing compact
quantization, strict gates, scoring arithmetic, transition order, and endpoint
handling.

The next measured continuation is exact anchor support-stencil reuse. For
complete interior cells with a full sampling halo, construct the canonical
owned-or-radius support offsets once and translate them into each tile instead
of rescanning the full sample cube. Preserve canonical observation order,
gradient-halo eligibility, profile population semantics, and the existing
clipped construction for partial or volume-boundary cells.

The next measured continuation removes robust-membership materialization.
Keep each observation's component assignment and residual histogram bin plus
the two component cutoff bins, and evaluate retained membership inline in the
existing centroid, objective, peak, and final-support scans. Preserve the exact
membership predicate, observation traversal and accumulation order, fitting
arithmetic, and all acceptance decisions.

The next measured continuation removes redundant owned-cell discovery during
anchor initialization. Production extraction already has a dense tile and the
exact cell bounds, so expose the owned Z/Y/X cube as a direct zero-allocation
range for seed initialization while retaining the existing support range for
refinement. Preserve canonical owned-observation order and the public vector
API's validation behavior.

The next measured continuation replaces the direction-conditioned peak
search's ordered-map response cache with bounded contiguous grid storage. The
peak domain, hill-climb traversal, response evaluation order, tie-breaking,
subpixel acceptance, and response arithmetic remain unchanged. Precompute the
grid's feasible points once and use direct shifted indices for response-cache
hits and misses.
