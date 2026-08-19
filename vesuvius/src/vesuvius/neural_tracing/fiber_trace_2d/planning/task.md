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

The next measured continuation splits direction-conditioned peak observations
into a compact hot response stream and a sparse retained-evidence stream. The
dominant response scan must no longer load gradient/alignment fields for every
spatially relevant observation. Exact floating-point identity and accumulation
order are not required; deterministic repeatability, acceptable anchor/replay
quality, and measured speed are the retention gates.

A later peak-search option should measure radial-cutoff survival and unique
observation use across evaluated neighboring candidates. Based on that result,
test either spatially coherent contiguous-block rejection for observations that
no visited candidate needs, or reuse observation loads across neighboring
candidate responses. Do not restore the previously rejected pointer-heavy CSR
or counting-sort implementations unchanged.

The measured checkpoint-17 continuation tested eliminating duplicate
transverse-Gaussian evaluation in retained spatial objectives and final
refined-state evaluation.
Each observation's per-component Gaussian is already required for the
denominator; retain the active-component local values and reuse the assigned component's
value in the numerator. Keep observation/component traversal, equations,
candidate decisions, and public behavior unchanged. Exact numeric identity is
not a retention requirement; deterministic replay quality and measured speed
remain the gates. The explicit local-value storage regressed both targeted
worker phases and was removed; the committed implementation remains active.

The next measured continuation targets the remaining direction-conditioned
peak-response scans. First measure exact radial-cutoff survival, per-component
unique record use, and repeated use across computed responses. If many records
are never needed, test conservative contiguous-block AABB rejection while
retaining sequential record storage. If most records are needed repeatedly by
neighboring candidates, instead batch only already-demanded neighboring
responses so one record load updates several response accumulators. Do not
restore pointer-heavy per-candidate indices or eagerly evaluate unused grid
responses.

Checkpoint 18 measured both strategies and tested full, four-candidate, and
two-candidate demanded cohorts. All batching widths regressed peak-search time
because the compact record stream was already cache-resident while multiple
compensated accumulator sets increased register and spill cost. The experiment
was removed; the committed scalar response cache remains active.

The next measured continuation targets compensated tensor accumulation in
`robustDirectionProposal()`. Replace the six compensated-double tensor
histogram entries with ordinary float32 accumulation, retaining the existing
assignment, robust cutoff, component selection, and eigensolver. Exact numeric
identity is not required; deterministic repeatability and similar anchor and
replay quality are the acceptance gates. Keep this checkpoint separate from
centroid and objective accumulator changes so its effect is measurable.

Checkpoint 19 tested both ordinary float32 tensor bins and ordinary double
tensor bins, but the machine was not idle and the timing results are invalid.
Both partial variants were removed without a retention decision.

The next implementation keeps the production compact-observation robust
proposal in float32 end to end: position and direction access, component and
pivot copies, Gaussian and assignment arithmetic, residual/mass histograms,
and tensor bins. Convert only the fixed-size histogram summaries to double for
the existing robust-cutoff policy and the final six tensor entries for the
existing eigensolver. The public double-observation path remains double through
the same scalar-generic implementation. Build and validate now, but do not run
the canonical performance benchmark until the user confirms the computer is
free.

Checkpoint 21 tested float32 fixed-direction spatial objectives used by anchor
position backtracking. Although the target kernel improved by 10.9%, the added
specialization consistently perturbed code generation enough to regress the
separate robust tensor proposal and total runtime. The experiment was removed;
checkpoint 20 remains the production baseline.

Checkpoint 22 retests that demonstrated compact-float objective kernel in a
separate translation unit so its code generation cannot perturb the robust
tensor-proposal implementation. Keep the expanded public fitter on its existing
double path and retain the experiment only if enclosing runtime improves.

Checkpoint 23 moves final refined-anchor support evaluation to float32 for both
compact production and expanded public observations. Do not preserve double
arithmetic merely for historical identity; safely narrow public values, compute
support/coherence/objective in float, and retain the change only if canonical
quality and enclosing performance remain acceptable.
