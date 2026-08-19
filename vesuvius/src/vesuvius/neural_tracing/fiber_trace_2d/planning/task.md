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

Checkpoint 24 removes the remaining historical double-precision representation
from anchor and fiberlet extraction. Use float32 for observations, configuration
used by extraction math, component/refinement state, retained anchors,
diagnostics, fiberlet candidate geometry, path costs, path points, and graph
geometry. Reference polyline, normal-sampler, and replay APIs may retain their
existing double representation outside this subsystem, but must narrow once at
the extraction boundary instead of propagating doubles through it. Timing and
process-accounting values remain double. Exact numeric identity is not required;
determinism and comparable extraction/replay quality are the retention gates.
Three canonical runs retained deterministic artifacts and replay failures while
improving median wall time by 2.8%, total CPU by 1.7%, and peak RSS by 5.4%
against checkpoint 23. The end-to-end float representation is retained.

## Remaining options after checkpoint 24

Continue from commit `d2229bf6f` and test these options one at a time:

1. Parallelize conversion and sorting of the 32 worker-local interpolation-
   corner sets before their existing deterministic merge. The checkpoint-24
   corner-merge stage costs about 1.05 seconds wall but only 1.36 CPU-seconds,
   exposing a mostly serial wall-time bottleneck. Preserve the final sorted
   unique voxel sequence exactly.
2. If corner sorting remains material, replace worker-local corner hash sets
   with sparse paged bitmaps. The canonical replay performs about 405.7 million
   insertion attempts for only 170,778 unique voxels. Preserve exact corner
   coverage and deterministic ordered output; measure page-directory overhead
   before retaining this larger change.
3. Improve anchor fitting load balance after sampled tile data becomes ready.
   Checkpoint 24 uses roughly 26 effective cores across the anchor phase on a
   32-thread run. Measure per-group and per-cell tails first, then test a queue
   that allows ready cells to be fitted independently without duplicating tile
   samples or exceeding the bounded sample-memory budget.
4. Accelerate direction-conditioned peak-response Gaussian evaluation. Peak
   search still costs about 33.75 worker-seconds over roughly 2.97 billion hot
   observation visits. Evaluate a bounded float lookup or polynomial only with
   explicit approximation-error, anchor-distribution, and replay-quality
   gates. This option intentionally permits small numeric changes.
5. Test eliminating the post-update robust-membership refresh in the default
   one-pass fitter by reusing the accepted step's membership. This removes
   repeated full observation scans but changes fitting semantics and therefore
   requires stronger anchor geometry and visualization review. Do not present
   it as behavior-preserving.
6. Further optimize fiberlet DP through transition SIMD/vectorization or
   largest-candidate-first scheduling. DP costs about 38.7 worker-seconds but
   only 1.23 seconds wall because it already parallelizes well, so prioritize
   it after the serial corner and anchor-tail work.

Do not repeat the rejected peak-response batching/CSR/counting-sort variants,
inline robust-membership materialization removal, or eager candidate-wide edge
generation unchanged. Any revisit must address the measured locality,
register-pressure, or repeated-predicate reason for the prior regression.

The current continuation replaces pair-local anchor-tile overlap reuse with
bounded extraction-wide raw prediction reuse. Partition tiles when the full
union would exceed the existing sample-memory budget; within each partition,
build the exact union of tile sample boxes, sample each prediction voxel once
in bounded parallel batches, and copy contiguous shared row ranges into each
tile before its unchanged gradient, observation, and fitting work. This must
preserve support for extractions larger than the memory budget. Retain the
change only if reduced sampler work improves end-to-end wall time without
unacceptable memory or replay-quality regression.

Checkpoint 25 parallelized conversion and sorting of the worker-local corner
sets while retaining the deterministic pairwise sorted-unique merge. Destination
capacity is allocated on the calling thread before workers populate and sort the
vectors, limiting allocator-arena growth. Three warm canonical runs reduced
median corner-finalization wall time from about 1.05 to 0.268 seconds and total
wall time from 8.96 to 8.23 seconds. Median total CPU increased from 199.47 to
202.91 seconds and peak RSS increased from 2,007,020 to 2,060,332 KiB. Exact
sample order, artifact hash, graph populations, DP work, and replay failures
were unchanged. The wall-time gain is retained with the CPU/RSS tradeoff
recorded.

Checkpoint 26 replaces worker-local interpolation-corner hash sets with sparse
`16^3` bit pages. A small page-pointer cache reduces 405.7 million insertion
attempts to roughly 75 thousand page-directory probes, then corresponding pages
are OR-merged and their exact set bits globally sorted in stored Z/Y/X order.
Three controlled warm runs improved median command wall from 8.23 to 7.77
seconds, total CPU from 202.91 to 196.18 seconds, fiberlet wall from 2.519 to
2.165 seconds, corner finalization from 0.268 to 0.0196 seconds, and peak RSS
from 2,060,332 to 1,697,516 KiB. Exact artifact and replay behavior were
retained. The sparse paged bitmap is retained.

The current continuation accelerates the transverse Gaussian in the scalar
direction-conditioned peak-response scan. Test a small process-wide float
lookup with linear interpolation over a bounded normalized-exponent interval,
with the existing `expf` calculation retained outside that interval. Keep the
radial cutoff, response traversal, compensated accumulation, evidence handling,
cache, hill climb, acceptance checks, and tie policy unchanged. Record direct
approximation error and retain the change only when canonical peak work and
end-to-end runtime improve with comparable anchor and replay quality.

Checkpoint 29 tested both the bounded lookup and arithmetic-only polynomial.
The lookup had no end-to-end gain and slightly increased peak-search work in a
paired run. The polynomial met the error and replay-quality gates but clearly
regressed peak, anchor, and command time. Both variants were removed; the
committed `expf` implementation remains the production baseline.

Checkpoint 35 retained lazy isotropic-smoothness evaluation in the shared local
scorer. The current measured continuation will prepare the candidate side of
normal-aware smoothness once per outgoing DP edge, then reuse it across every
reached incoming state. The shared local scorer must remain the sole owner of
the equations; public callers prepare on demand, while the DP stores only a
stack-local prepared descriptor. Retain the checkpoint only if exact branch
coverage and byte-identical replay output hold and alternating canonical
measurements show a repeatable enclosing gain.

Checkpoint 37 will prepare the reusable sides of multiplicative alignment
scoring. Current-side prediction orientation and its previous-direction factor
will be prepared once per reached incoming state; candidate-side prediction
orientation, presence, and candidate-direction factor will be prepared once
per valid outgoing edge. Pair-dependent dot products and the original seven-
factor multiplication order remain in the transition loop. The shared local
scorer remains the sole owner of the equations, public callers continue to
prepare on demand, and retention requires exact replay output plus a repeatable
fiberlet-DP gain.
