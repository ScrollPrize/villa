# Plan: staged fiberlet anchor acceleration

## Baseline And Review Gates

1. Use commit `73fe64e09` as the checkpoint baseline. Its canonical replay uses
   32 threads, length 5000, and two robust iterations. The latest run measured
   16.37 seconds total wall, 9.76 seconds anchor wall, 277.38 seconds anchor
   CPU, 2520 anchors, 24,518 accepted fiberlets, and 2 greedy / 1 fiberlet
   replay failures.
2. Implement exactly one option per checkpoint. Run focused GCC and Clang
   tests plus the canonical replay after each checkpoint.
3. Report total/anchor/fiberlet wall and CPU, dominant profile phases,
   populations, sampled voxels, DP nodes, and replay failures. Stop for user
   review before starting the next option.
4. Remove experiments that do not improve measured performance or that cause
   an unacceptable quality regression. Record both successful and rejected
   variants in `task_log.md`.

## Checkpoint 1: Tile-Owned Compact Observations

1. Preserve the public `fitFiberCellAnchors()` vector API for focused fixtures
   and external callers.
2. Add an internal observation-range abstraction used by the common fitter so
   production extraction can reference tile-owned observations without copying
   expanded records into every overlapping cell.
3. Construct one compact tile observation per sampled voxel after prediction
   and gradient sampling. Store float coordinates, pre-normalized float
   directions, float presence and gradient, and validity flags. Invalid input
   remains invalid and cannot become positive evidence.
4. For each cell, build only compact tile indices for the existing owned-or-
   support-domain predicate. The common fitter must see the same observation
   order and logical population as before.
5. Account compact tile storage and per-cell index scratch against the existing
   concurrent sample-memory budget. Do not retain both expanded cell records
   and compact tile records in production.
6. Add focused coverage comparing vector-backed and indexed compact fitting on
   the same fixture within float-appropriate geometric tolerances. Profile
   observation construction and total fitting before accepting the checkpoint.

## Checkpoint 2: Reuse Robust State

1. Extend robust proposals with the baseline Gaussian/alignment quantities
   already calculated during assignment.
2. Derive the baseline spatial objective from that pass; calculate only moved
   candidate state in the following objective pass.
3. Fuse final membership refresh with final support evaluation where the
   required axes and positions are identical.
4. Avoid adding large parallel per-observation streams that repeat the prior
   working-set regression.

Measured outcome: rejected. Fusing the fixed-axis spatial baseline with
centroid accumulation replaced a sparse retained-evidence centroid pass with
an all-site pass. Three canonical runs showed no wall or CPU improvement, so
the implementation was removed. The post-refinement membership refresh cannot
be fused with final support without changing semantics because peak refinement
changes component positions between those phases.

## Checkpoint 3: Batched Peak Responses

1. Evaluate multiple peak candidates per observation pass using compact float
   SoA storage, or build contiguous counting-sort/CSR spatial bins.
2. Do not restore linked per-observation bins; their pointer chasing reduced
   visit counts but increased runtime.
3. Preserve the same candidate domain and acceptance checks. Compare peak CPU,
   response visits, selected peaks, and replay quality.

Measured outcome: rejected. Batched neighborhood responses reduced physical
observation scans but increased peak-search time to 39.18 seconds. A 2D CSR
broad phase reduced candidate visits by 59% but increased peak-search time to
39.20 seconds. A 1D counting sort retained contiguous ranges but increased
peak-search time to 42.13 seconds and total wall to 14.91 seconds. All quality
populations remained unchanged. The compact sequential scan has better
locality than these alternatives, so all checkpoint-3 code was removed.

## Checkpoint 4: One Robust Pass

1. Make the measured pass count explicit and run one-pass versus two-pass
   canonical comparisons.
2. Compare anchor geometry/populations and produce visualization artifacts for
   user inspection. Do not accept this checkpoint solely from replay failure
   counts.

Measured and accepted: one pass improved median total wall by 7.1% and anchor
wall by 17.7%, while retaining 3.3% more anchors and 8.1% more accepted
fiberlets. Replay failures stayed at 2 greedy / 1 fiberlet, but local anchor
comparison had material tails and unmatched candidates. One pass is now the
default; document `--maximum-iterations` as the explicit quality/speed knob
for difficult overlapping-fiber data.

## Checkpoint 5: Shared Tile-Halo Sampling

1. Measure repeated prediction voxels between adjacent six-cell tiles.
2. Reuse overlap through bounded shared spatial batches or worker-local spatial
   groups without constructing a whole-volume dense buffer.
3. Preserve cache behavior, sample values, memory limits, and enough jobs for
   effective 32-worker scheduling.

Measured and accepted: the 192 tile boxes submitted 39,701,808 coordinates but
their exact union contained only 6,162,456 voxels. Deterministic
maximum-overlap pairs formed 98 bounded jobs, retained all 32 workers, reused
12,960,096 samples, and reduced actual submissions to 26,741,712. Median total
wall improved from 13.69 to 13.54 seconds and anchor CPU from 184.80 to 174.07
seconds. Anchor/fiberlet populations, DP work, and replay failures were
unchanged.

## Checkpoint 6: Direct Packed-Key DP Index

1. Replace each candidate's `unordered_map` from packed lattice key to node
   index with a bounded direct `uint32_t` table over the already validated key
   range.
2. Preserve node generation, transition order, missing-node behavior, DP
   state, and cost calculations exactly.
3. Account the direct table rather than estimated hash storage in peak search
   memory, and report allocated slots so sparse-lattice overhead is visible.

Measured candidate: 50,718,661 retained nodes occupied 109,469,154 direct
slots (46.3%). Median total wall improved from 13.54 to 13.26 seconds and
median total CPU from 345.87 to 337.69 seconds. Work/quality populations and
the replay artifact were unchanged. The direct index was accepted and retained.

## Checkpoint 7: Paged Scoring Lookup

1. Measure occupied page storage and distinct per-stencil page probes for
   `4^3`, `8^3`, and `16^3` prediction-voxel pages.
2. Replace repeated scoring-voxel hash lookups with dense indices inside
   occupied pages while retaining a sparse page directory.
3. Preserve interpolation corner order, weights, tensor accumulation,
   principal-axis resolution, and node quantization exactly.

Measurement selected `16^3`: 199 pages, 815,104 slots, and 61,144,696 page
directory probes versus 406,380,154 voxel-corner lookups. The implementation
reduced median interpolation materialization wall time from 2.40 to 2.30
seconds and total wall from 13.26 to 13.22 seconds. Exact artifact parity was
retained, and the paged lookup was accepted.

## Checkpoint 8: Prepared Scoring Tensors

1. Sample the remaining interpolation path to separate sparse lookup,
   prediction/normal corner work, and principal-axis resolution.
2. Validate and normalize each unique sampled prediction and normal once.
   Store compact float32 axes and six independent symmetric outer-product
   components per field.
3. Preserve interpolation corner and weight order, double-precision weighted
   accumulation, principal-axis resolution, and final node quantization.

The bounded profile attributed about 16% of sampled time to lookup, 44% to
repeated corner direction/tensor work, and 40% to principal-axis resolution.
All 12,423 sampled points required both principal-axis solves. Three optimized
runs reduced median interpolation materialization from 2.30 to 1.71 seconds
and total wall from 13.22 to 12.51 seconds. Fiberlet geometry, populations,
and replay failures were unchanged; loss values changed around `1e-6` and a
handful of DP work counters changed because of the intentional float32
preparation boundary. The prepared scoring representation was accepted.

## Checkpoint 9: Closed-Form Principal Axis

1. Keep the shared iterative Jacobi resolver unchanged for anchors and other
   existing users.
2. Resolve interpolation tensors with analytic symmetric 3x3 eigenvalues and
   stable row-cross-product eigenvector reconstruction.
3. Reject zero/non-finite tensors and ambiguous top eigenvalues directly.
   Fall back to Jacobi only when the top gap is clear but eigenvector
   reconstruction fails a scale-aware residual check.
4. Count every prediction/normal resolution and iterative fallback, and compare
   complete replay artifacts against checkpoint 8.

Three canonical runs resolved 101,644,450 tensors each with zero iterative
fallbacks. Median interpolation materialization fell from 1.70 to 1.55 seconds,
total wall from 12.51 to 12.42 seconds, and total CPU from 315.81 to 310.98
seconds. Every run was byte-identical to checkpoint 8. The closed-form
resolver was accepted.

## Checkpoint 10: Prepared DP Nodes And Edges

Measure the following variants independently against checkpoint 9. Do not
conflate preparation and search: report both, plus total wall/CPU and peak RSS.

1. Cache each retained node's point, decoded normalized prediction, decoded
   normalized normal, presence, and validity once before dynamic programming.
   Preserve the compact materialized `SearchNode` representation used by the
   earlier parallel stage; the expanded representation is solve-local. Keep
   the current double-precision strict 25-degree feasibility gate during the
   first comparisons.
2. For each reached current node, resolve its at most nine outgoing neighbors,
   edge directions/lengths, destination prediction-deviation check, and
   destination scoring data once. Reuse those descriptors across all reached
   incoming states at that node.
3. Compare the reached-node cache with a candidate-wide pre-generated compact
   edge table. The full table may perform work for unreachable nodes, so retain
   it only if reduced DP work outweighs preparation and memory costs. This
   table covers interior node-to-node edges only; source initialization and
   sink finalization retain their distinct endpoint semantics.
4. Compact `DpState`. The incoming transition state uniquely determines the
   predecessor lattice key `(layer-1,u-du,v-dv)`, incoming direction, and
   incoming length for states 0 through 8. State 9 is source-only. Store only
   cumulative cost, reachability, and the predecessor's state index needed for
   reconstruction; derive the predecessor node from the direct node index,
   which remains alive through reconstruction.
5. If scoring remains dominant, add a prepared normalized local-metric path
   which consumes cached unit vectors directly. Keep the public shared scoring
   implementation authoritative and extract shared arithmetic instead of
   copying it.
6. Keep each variant only when it improves the combined preparation plus DP
   time without unacceptable replay-quality or memory regression. Record all
   rejected variants in `task_log.md`.

### Checkpoint 10 Selected Composition

The retained composition caches solve-local normalized node data, reuses each
reached node's outgoing descriptors across incoming states, derives compact
predecessors from packed keys, rolls float32 cumulative costs over two layers,
and stores global one-byte predecessor states for reconstruction. The eager
candidate-wide edge table was rejected because its unreachable-node work made
combined search slower.

Three final canonical runs measured median total wall/CPU at 11.91/283.73
seconds, fiberlet wall/CPU at 4.52/108.20 seconds, and search wall/CPU at
1.00/31.39 seconds. Against checkpoint 9, median total wall improved 4.1%,
total CPU 8.8%, search wall 46.2%, and search CPU 45.9%. Geometry, populations,
and 2 greedy / 1 fiberlet replay failures remained unchanged; float cumulative
costs changed ten relaxation decisions and serialized costs only.

### Checkpoint 10 Validation

1. Add focused path tests for predecessor reconstruction, missing neighbors,
   source states, and paths crossing every transverse transition.
2. Run focused GCC and Clang `test_fiberlet_paths` and `test_fiber_replay`, plus
   `test_fiber_anchors` as the extraction integration guard.
3. Run three canonical 5,000-base-voxel replays for each serious contender.
   Use the checkpoint-9 one-pass command without the obsolete explicit
   `--maximum-iterations 2`. Compare total, preparation, node/edge preparation,
   DP wall/CPU, peak RSS, populations, DP counters, failures, and artifact
   hashes. When hashes differ, compare per-candidate success/reason, selected
   path geometry, cost components, graph population, and replay failures.
4. Retain only the fastest acceptable composition in production code.
5. Report exact expanded-node, edge-table, direct-index, and state bytes per
   candidate and the peak concurrent search-byte estimate. Count unique reached
   nodes and generated, valid, and reused edge descriptors.

### Checkpoint 10 Spec Update

- Document the retained node/edge representation and state-key invariant in
  `planning/specs.md` after selection.
- Increment the extraction-profile schema only for counters/timings retained
  in the final implementation.

### Checkpoint 10 Documentation Update

- Update `volume-cartographer/docs/fiberlets.md` with the selected DP data flow,
  memory accounting, profile fields, and benchmark result.
- Add the accepted result to `planning/changelog.md`; keep failed variants only
  in `planning/task_log.md`.

## Checkpoint 11: Lazy Node Scoring Materialization

1. Use commit `93bde87a2` and its three-run checkpoint-10 medians as the
   baseline: 11.91 seconds total wall, 283.73 seconds total CPU, 4.52/108.20
   seconds fiberlet wall/CPU, and 1.00/31.39 seconds search wall/CPU.
2. Preserve global native-corner collection, deterministic unique merging,
   batched prediction/normal reads, prepared scoring voxels, and the paged
   scoring index. These establish the complete immutable source data required
   by any node that search may request.
3. Eagerly interpolate only exact candidate endpoints. Retain the immutable
   prepared scoring array and page index through search instead of releasing
   them before dynamic programming.
4. Give each candidate solve a local direct node-to-cache index and a compact
   append-only `DpNodeScoring` cache. On first access, interpolate the node at
   its authoritative stored float position, pass through the existing compact
   encode/decode boundary, and prepare the normalized metric representation.
   Reuse that cached value for every later gate, normal, and scoring access.
   Store cache indices in edges and local variables rather than references so
   append growth cannot invalidate active data; do not reserve the full retained
   node population.
5. Preserve canonical node/state/transition order, source/sink special cases,
   strict double prediction-deviation checks, float local scoring, rolling
   cumulative costs, and predecessor reconstruction. Shared source data is
   immutable and every lazy cache is owned by one solve worker, so no locks or
   cross-candidate writes are introduced.
6. Every candidate owns its mutable page-lookup cache and probe counter. Only
   the underlying paged index and prepared scoring array are shared immutable
   data.
7. Extract one shared compact-scoring conversion helper used at the eager and
   lazy boundaries; do not duplicate compact axis encoding, presence rounding,
   validity, or decode preparation.
8. Count endpoint interpolations, lazy node requests, unique lazy
   materializations/cache misses, and cache hits. Keep
   `interpolatedScoringPoints` as endpoint plus unique lazy interpolations.
   Include shared scoring/index bytes retained through search, the candidate
   node-to-cache map, actual lazy cache capacity, direct node index, rolling
   states, and backpointers in memory diagnostics; retain a conservative
   all-node cache bound for admission accounting.
9. Avoid a timer call on every hot cache access. Retain deterministic sparse
   interpolation subprofiling using a fixed hash of canonical candidate index
   plus packed node key, while exact resolution and materialization counters
   remain complete and candidate-local diagnostics aggregate in canonical
   order.
10. Add regression coverage for endpoint-only interpolation, actual
   interpolation being lower than retained nodes on a pruned fixture, cache
   reuse, serial/parallel determinism, unchanged source-coordinate requests,
   invalid/ambiguous rejection, path geometry/costs, and failures. Run focused
   GCC and Clang tests.
11. Run three identically warmed canonical 5,000-base-voxel replays. Record
   command, inputs, baseline commit, build type, threads, cache state, and
   min/median/max. Compare total, fiberlet,
   endpoint materialization, search wall/CPU, peak RSS, actual interpolated
   nodes, cache hits, DP counters, populations, failures, and artifact hashes.
   Retain the variant only if combined endpoint materialization plus search and
   total runtime improve without unacceptable quality change.

### Checkpoint 11 Spec Update

- Document eager endpoint versus lazy interior-node materialization, immutable
  shared source data, candidate-local cache ownership, actual interpolation
  counter semantics, and the retained compact quantization boundary.

### Checkpoint 11 Documentation Update

- Update `volume-cartographer/docs/fiberlets.md` and profile-schema notes with
  the accepted data flow and measured outcome. Record failed variants only in
  `task_log.md`; add an accepted result to `changelog.md`.

### Checkpoint 11 Measured Outcome

Accepted. Three runs materialized 14,478,750 of 50,718,661 retained nodes and
served 44,815,799 repeated requests from candidate-local caches. Median total
wall/CPU fell from 11.91/283.73 to 10.76/248.84 seconds; fiberlet wall/CPU fell
from 4.52/108.20 to 3.30/69.38 seconds. Search itself rose from 1.00/31.39 to
1.29/40.66 seconds because lazy interpolation moved into search, while eager
interpolation fell from 1.52 seconds wall to about 0.015 seconds. All final
artifacts exactly matched checkpoint 10.

## Spec Update

- Document compact tile observation ownership and float precision boundaries.
- Document any accepted state reuse, batched peak evaluation, iteration-count
  change, or shared-halo behavior only after its checkpoint is measured and
  approved.
- Keep explicit permission for small numeric differences and require geometric
  and replay-quality comparison at every checkpoint.

## Documentation Update

- Update `volume-cartographer/docs/fiberlets.md` after each accepted checkpoint
  with the implemented data flow and benchmark result.
- Keep this task's measured successes and failures in `task_log.md`; add only
  accepted high-level changes to `changelog.md`.

## Validation

1. Build `vc_fiberlets`, `test_fiber_anchors`, `test_fiberlet_paths`, and
   `test_fiber_replay` in the regular GCC tree.
2. Run the three focused CTest targets in GCC and Clang CI-style trees.
3. Run `git diff --check`.
4. Run the exact canonical replay command recorded in `task_log.md` and compare
   against the immediately preceding accepted checkpoint.

## Checkpoint 12: Canonical Anchor Support Stencil

1. Build one canonical support stencil for a complete cell with a full halo.
   Store ordered `(z, y, xBegin, xEnd)` spans relative to the cell sample begin,
   generated under the exact existing owned-or-radius predicate. Do not flatten
   canonical offsets because tile row and plane strides vary.
2. When gradients are enabled, the extra halo voxel guarantees every retained
   full-stencil site is gradient-eligible. Final validity is therefore exactly
   the tile observation's sampled-gradient validity. Keep the explicit halo
   eligibility test in the clipped fallback.
3. Use the stencil only when every axis satisfies `end - begin == cellSize`,
   `begin >= sampleHalo`, and `gridShape - end >= sampleHalo`. Crop and tile
   boundaries do not affect eligibility. Translate each 3D span through the
   current tile shape into compact observation indices. Keep the existing
   clipped scalar scan for every other cell.
4. Preserve profile semantics: `candidateObservations` remains the number of
   sample-cube sites represented, while `retainedObservations`, gradient
   attempts, and valid gradients retain their existing logical populations.
   Add explicit fast-path/fallback cell counters, require their sum to equal
   `workCells`, and advance the emitted profile schema from 14 to 15.
5. Keep the stencil immutable and extraction-local. Do not create a dense
   per-cell cache or increase concurrent sample ownership.
6. Compare the ordered span expansion against a scalar oracle for odd and even
   cell sizes, with and without gradient weighting. Cover multiple full-halo
   cells sharing a tile where tile and cell sample origins differ, exact low/
   high halo eligibility, a crop-edge interior cell, and a partial final cell.
   Assert fast/fallback counters, deterministic serial/parallel populations,
   and unchanged extracted anchors on existing fixtures.
7. Build and run focused GCC and Clang anchor/path/replay tests. Run three
   canonical 32-thread, 5,000-base-voxel replays and compare total/anchor wall
   and CPU, observation-construction work, populations, replay failures, and
   artifact hashes against checkpoint 11.
8. Retain the checkpoint only if it improves total or anchor cost without an
   unacceptable population or replay-quality regression.

### Checkpoint 12 Spec Update

- Document that production anchor extraction reuses a canonical support
  stencil only for complete full-halo cells and retains clipped scalar
  construction elsewhere.
- Define canonical Z/Y/X ordering, exact owned-or-radius membership, and
  gradient-validity equivalence as required behavior.
- Add fast-path/fallback profile counters to the extraction profile schema.
- Define the emitted schema as profile version 15.

### Checkpoint 12 Documentation Update

- Update `volume-cartographer/docs/fiberlets.md` with the support-stencil fast
  path, its boundary fallback, and the new profile fields.
- Record the accepted or rejected benchmark result in `task_log.md`, summarize
  an accepted checkpoint in `changelog.md`, and complete `status.md`.

### Checkpoint 12 Result

Accepted. All 13,027 canonical work cells used the stencil. Three runs reduced
median observation-construction worker time from 18.99 to 11.84 seconds,
anchor CPU from 177.28 to 167.49 seconds, and total wall from 10.76 to 10.37
seconds. Populations, DP work, replay failures, and complete replay artifacts
remained identical to checkpoint 11.

## Checkpoint 13: Inline Robust Membership

1. Use commit `c100de833` and the accepted checkpoint-12 three-run medians as
   the baseline: 10.37 seconds total wall, 238.78 seconds total CPU, 6.42
   seconds anchor wall, and 167.49 seconds anchor CPU.
2. Replace the materialized retained/not-retained byte array with one internal
   robust-membership representation containing canonical component assignments,
   canonical residual histogram bins, and the two selected cutoff bins.
3. Define membership exactly as the existing predicate: an observation is
   retained iff its assignment is below the active component count and its
   residual bin is no greater than that component's cutoff. Keep unassigned
   observations distinguishable through the assignment byte.
4. Remove the second full observation pass in `robustDirectionProposal()` that
   currently overwrites residual bins with booleans. Carry the membership
   representation through local centroid calculation, spatial objectives,
   peak preparation, and final evaluation; do not rematerialize booleans at a
   later phase boundary.
5. Define the predicate in one shared internal/detail membership helper used by
   production and focused tests. Move the final membership object into
   `RefinedFitState` and pass it by const reference through peak and final
   evaluation. Do not replace the removed pass with O(N) membership-vector
   copies. A proposal discarded after component compaction must not transfer
   assignments or cutoffs into the restarted fit.
6. Preserve observation traversal, component traversal, compensated-sum update
   order, robust cutoff selection, tensor construction, line-search decisions,
   component removal, and final support arithmetic. The optimization changes
   only how an already-selected membership predicate is represented.
7. Advance the extraction profile schema to version 16. Keep
   `localTensorObservationVisits` as physical proposal-scan visits and add an
   explicit avoided-membership-materialization visit count. Require their sum
   to equal the version-15 logical tensor-visit count, including the
   `computeAxes=false` final refresh.
8. Add focused tests that compare the shared inline helper against a
   materialized oracle for one/two active components, assigned/unassigned
   observations, arbitrary unassigned residuals, cutoff equality, bin/cutoff
   255, fully retained, and trimmed cases. Force component removal and verify
   the restarted fit does not retain stale membership. Retain final peak use,
   serial/parallel anchor, and replay determinism coverage.
9. Build and run focused GCC and Clang anchor/path/replay tests. Run three
   canonical 32-thread, 5,000-base-voxel replays and compare total/anchor wall
   and CPU, tensor-proposal work, physical and avoided visits, populations, DP
   work, replay failures, and artifact hashes against checkpoint 12.
10. Retain the checkpoint only if it improves total or anchor cost without an
   unacceptable population or replay-quality regression.

### Checkpoint 13 Spec Update

- Document residual-bin/cutoff membership representation and the exact inline
  retained predicate.
- Define `localTensorObservationVisits` as physical proposal scans and document
  the avoided materialization counter.
- Define the emitted extraction-profile schema as version 16.

### Checkpoint 13 Documentation Update

- Update `volume-cartographer/docs/fiberlets.md` with the inline membership data
  flow, profile fields, and measured result.
- Record the benchmark and validation outcome in `task_log.md`, summarize an
  accepted checkpoint in `changelog.md`, and complete `status.md`.

### Checkpoint 13 Result

Rejected and removed. The representation eliminated exactly 809,364,400
materialization visits per run and retained exact artifacts, populations, DP
work, and failures. It replaced one retained-byte load in every later hot scan
with assignment, residual-bin, and cutoff evaluation, however. Across three
runs median command wall rose from 10.37 to 11.01 seconds, anchor wall from
6.416 to 6.653 seconds, and anchor CPU from 167.49 to 169.09 seconds. The
checkpoint-12 materialized retained-byte representation remains the baseline.

## Checkpoint 14: Direct Owned-Cell Initialization Range

1. Use checkpoint 12 as the baseline: 10.37 seconds median command wall,
   6.416 seconds anchor wall, and 167.49 seconds anchor CPU. Recover and compare
   `anchor_fit_setup_work_seconds` from the checkpoint-12 profile because the
   two scans being removed are charged to fit setup; the 11.84-second support
   observation-construction measurement is not their phase baseline.
2. Separate the support observation range used by robust refinement from the
   owned observation range used by initial seed fitting. Keep one shared
   fitting implementation and pass both ranges explicitly.
3. For the production dense-tile path, expose the exact clipped cell cube as a
   zero-allocation row range. Map canonical local Z/Y/X positions
   directly through the tile shape and origin to compact observations; do not
   create an owned-index vector. Validate in O(1) that cell bounds are
   monotonic and contained by the tile sample box, the tile shape product
   equals the observation count, and the owned product equals the clipped cell
   volume before unchecked traversal.
4. Preserve the public vector API by deriving and validating its owned range
   through the existing coordinate-based rules. Preserve its stable input-order
   filter and historical count-only coverage check exactly: do not sort,
   deduplicate, require lattice coordinates, or strengthen duplicate/missing
   validation. Invalid, non-finite, and incomplete public input must retain its
   current behavior.
5. Preserve canonical Z/Y/X owned-observation order, initial denominator and
   seed arithmetic, support-range refinement, gradient-validity semantics,
   profile populations, and all acceptance decisions.
6. Advance the extraction profile schema. Add separate counters for public
   discovery visits, physical owned-initialization visits, and production
   support visits avoided relative to the previous two complete support-range
   scans.
   Keep existing logical counters such as `weightedObservations` stable.
7. Add focused tests for direct-range/index equivalence, partial cells,
   non-zero tile origins, boundary cells, structural range validation, public
   shuffled/off-lattice/duplicate-plus-missing input, invalid flags, NaN
   presence, unusable directions, denominator behavior, and serial/parallel
   extraction parity.
8. Build and run focused GCC and Clang anchor/path/replay tests. Run three
   canonical 32-thread, 5,000-base-voxel replays and compare total/anchor wall
   and CPU, initialization/setup work, populations, DP work, failures, and
   artifact hashes against checkpoint 12.
9. Retain the checkpoint only if total or anchor cost improves without an
   unacceptable population or replay-quality regression.

### Checkpoint 14 `planning/specs.md` Update

- Document the separate owned initialization and support refinement ranges.
- Define canonical direct owned-cell traversal and public validation behavior.
- Document any retained profile fields and schema-version change.

### Checkpoint 14 Documentation Update

- Update `volume-cartographer/docs/fiberlets.md` with the direct owned-range
  production path and its measured result.
- Record the accepted or rejected benchmark result in `task_log.md`; update
  `changelog.md` only for an accepted checkpoint and complete `status.md`.

### Checkpoint 14 Result

Accepted. Production initialization directly visited 833,728 owned voxels and
avoided 858,114,544 visits from the former two support-range scans. Across
three fresh paired runs, median fitter-internal setup time fell from 10.996 to
0.074 seconds, anchor CPU from 169.45 to 162.51 seconds, anchor wall from 6.553 to
6.304 seconds, total CPU from 243.52 to 237.40 seconds, and command wall from
10.76 to 10.65 seconds. Populations, 62,970,689 DP relaxations, 2 greedy / 1
fiberlet failures, and replay SHA-256 remained identical.
The final profile attribution also includes constant-time direct-layout
construction in setup; its validation run reported 0.092 setup worker-seconds.

## Checkpoint 15: Contiguous Peak-Grid Cache

1. Use committed checkpoint 14 (`bc16557f5`) and its fresh paired medians as
   the baseline: 10.65 seconds command wall, 237.40 seconds total CPU, 6.304
   seconds anchor wall, 162.51 seconds anchor CPU, and roughly 43.9 worker
   seconds in direction-conditioned peak search.
2. Replace the per-component `std::map<pair<int,int>, double>` with one bounded
   row-major grid covering `[-extent,+extent]^2`. Extract its checked layout
   and response-cache storage into one reusable internal `detail` helper used
   by production and focused tests. Store computed state separately from
   response values so every response, including NaN or infinity if produced by
   unchanged arithmetic, retains the old cache-hit semantics.
3. Precompute each grid slot's physical point and feasibility once. Use a
   checked shifted-index helper for all hill-climb, neighbor, separable-fit,
   and joint-fit accesses. Do not change candidate enumeration, neighbor order,
   response computation order, response-cache request/miss counters,
   tie-breaking, tolerance calculation, or subpixel acceptance evaluations.
   Preserve the historical `(0,0)` fallback when clipping produces no feasible
   grid slot. Keep physical point construction in double precision and response
   coordinate construction as independent float index/step multiplication;
   never derive one from the other.
4. Keep arbitrary accepted subpixel candidates uncached exactly as before.
   Preserve the existing configuration limit on extent. Validate signed bounds
   before offsetting coordinates, then calculate side and area with checked
   unsigned arithmetic so 32-bit and 64-bit builds reject overflow consistently.
5. Add focused internal tests for shifted index uniqueness/bounds and cached
   response behavior over minimum, default, and maximum supported extents.
   Cover extent zero, exact `+/-extent`, rejected `+/-(extent+1)`, cached NaN
   and infinities, and exact compute-call counts for cache hits. Retain existing
   peak mode, symmetry, subpixel, robust-membership, and serial/parallel
   extraction tests as behavioral coverage. The production extent-zero case
   must verify the single feasible center slot. No direct no-feasible-slot case
   is required because every nonempty fitted cell owns finite lattice positions
   enclosing its center; preserve the dormant center fallback structurally.
   The untouched plateau/lexicographic selection code remains under existing
   peak behavior coverage.
6. Build and run focused GCC and Clang anchor/path/replay tests. Run three
   canonical 32-thread, 5,000-base-voxel replays and compare total/anchor wall
   and CPU, peak-search worker time, peak request/miss/acceptance counters,
   populations, DP work, failures, artifact hashes, and peak RSS against the
   nearest recorded baseline. Peak-cache construction remains included in
   peak-search time.
7. Retain the checkpoint only if it improves peak or enclosing anchor cost and
   preserves peak outputs, deterministic counters, populations, failures, and
   complete replay artifacts exactly.

### Checkpoint 15 Spec Update

- Document that bounded direction-conditioned peak responses use direct
  row-major cache slots while preserving the canonical search and response
  semantics.
- No profile-schema change is planned because existing request, computed, and
  acceptance counters retain their definitions.

### Checkpoint 15 Documentation Update

- Update `volume-cartographer/docs/fiberlets.md` with the bounded peak-grid
  cache and measured result if retained.
- Record the benchmark and validation outcome in `task_log.md`; summarize an
  accepted checkpoint in `changelog.md`, and complete `status.md`.

### Checkpoint 15 Result

Accepted. Three canonical runs reduced median direction-conditioned peak-search
worker time from roughly 43.9 to 42.84 seconds, anchor CPU from 162.51 to
160.30 seconds, anchor wall from 6.304 to 6.221 seconds, total CPU from 237.40
to 234.29 seconds, and command wall from 10.65 to 10.43 seconds. Median peak
RSS was 2.02 GiB. Peak request/miss/acceptance counters, all extraction and DP
populations, 2 greedy / 1 fiberlet failures, and complete replay SHA-256 were
identical to checkpoint 14.

## Checkpoint 16: Split Peak Response And Evidence Streams

1. Use committed checkpoint 15 (`f88aea31e`) and its three-run medians as the
   baseline: 10.43 seconds command wall, 234.29 seconds total CPU, 6.221
   seconds anchor wall, 160.30 seconds anchor CPU, and 42.84 worker-seconds in
   direction-conditioned peak search.
2. Replace the wide all-observation peak record with a compact fixed-width hot
   record containing transverse coordinates, axial Gaussian, and signal. Keep
   one hot record for every spatially relevant observation in canonical order,
   with retained-evidence indices in a parallel fixed-width array so the
   dominant coordinate/kernel stream remains a 16-byte stride.
3. Store direction-alignment and projected-gradient fields only for retained,
   direction-usable evidence whose stored float alignment is positive. Invalid
   gradients remain evidence because their aligned weight contributes to the
   gradient-coverage denominator. Preserve that evidence's relative observation
   order. A response scans the hot records once, computes the
   Gaussian once, accumulates denominator and numerator, and consults the
   compact evidence stream only when the hot record carries a valid index.
4. Small floating-point and accumulation-order changes are acceptable. Do not
   add compatibility arithmetic solely to reproduce legacy bits. Preserve the
   response equation, candidate domain, hill-climb/tie traversal, acceptance
   checks, and deterministic execution.
5. Advance the extraction profile schema and report prepared hot/evidence
   populations plus hot/evidence response visits. Prepared-count ratios expose
   the evidence population fraction. Hot visits count every response-record
   scan; evidence visits count only indexed records actually reached inside the
   radial cutoff, so their ratio also includes spatial rejection. Report actual
   record sizes/capacities and peak RSS.
6. Add focused profile invariants and peak behavior coverage, including a
   mixed/interleaved fixture with denominator-only and retained evidence, plus
   retained evidence with invalid gradients. Assert stream and visit
   relationships without exposing the transient record types as public API.
7. Build and run focused GCC and Clang anchor/path/replay tests. Run three
   canonical 32-thread, 5,000-base-voxel replays and compare total/anchor wall
   and CPU, peak-search worker time, hot/evidence populations and visits, peak
   request counters, peak RSS, extraction/DP populations, replay failures, and
   artifacts against checkpoint 15. If artifacts differ, additionally compare
   deterministic repeats, anchor axis/position distributions, downstream replay
   metrics, and generated visualizations.
8. Retain the checkpoint only if peak or enclosing anchor cost improves and
   deterministic repeatability plus anchor/replay quality remain acceptable.
   Exact artifact identity with checkpoint 15 is explicitly not required.

### Checkpoint 16 Spec Update

- Document the hot response and sparse evidence representations and their
  retained profile counters.
- State explicitly that response regrouping may change floating-point
  accumulation order and is accepted through deterministic quality gates.

### Checkpoint 16 Documentation Update

- Update `volume-cartographer/docs/fiberlets.md` with the split peak data flow,
  profile fields, evidence fraction, and measured outcome if retained.
- Record validation and benchmarks in `task_log.md`; update `changelog.md` only
  for an accepted checkpoint and complete `status.md`.

### Checkpoint 16 Result

Accepted. The retained layout uses a 16-byte response record, parallel 4-byte
evidence indices, and 16-byte evidence records. Only 9,607,554 of 199,261,642
prepared records (4.82%) carried evidence, and only 96,698,222 of 2,974,011,902
hot response visits (3.25%) loaded evidence after radial rejection. Across
three final canonical runs, median command wall fell from 10.43 to 10.25
seconds, total CPU from 234.29 to 229.90 seconds, anchor wall from 6.221 to
6.070 seconds, anchor CPU from 160.30 to 155.82 seconds, and peak-search worker
time from 42.84 to 39.94 seconds. Peak RSS was 2.03 GiB. Populations, DP work,
2 greedy / 1 fiberlet failures, and replay SHA-256 remained identical.

## Future Option: Radial Demand And Neighbor Reuse

1. Add measurement-only counters for total hot records passing the radial
   cutoff and unique prepared records used by at least one evaluated candidate
   per component. Report overlap between adjacent candidate responses.
2. If many prepared records are never used, group the hot stream into spatially
   coherent contiguous blocks with conservative transverse AABBs and reject
   whole blocks before individual distance/Gaussian work.
3. If most records are used but neighboring responses overlap heavily, batch
   only the already-demanded neighboring responses so each hot record load can
   update several accumulators. Do not eagerly evaluate the full peak grid.
4. Compare against the rejected checkpoint-3 results: its 2D CSR removed 59%
   of candidate visits but lost overall because indexing and locality overhead
   exceeded the arithmetic savings. Any new structure must retain sequential
   access and demonstrate lower peak and total CPU, not merely fewer visits.

## Checkpoint 17: Reuse Objective Gaussian Values

1. Use committed checkpoint 16 (`7bb2830fd`) and its three-run medians as the
   baseline: 10.25 seconds command wall, 229.90 seconds total CPU, 6.070
   seconds anchor wall, 155.82 seconds anchor CPU, and 39.94 worker-seconds in
   direction-conditioned peak search.
2. In `retainedSpatialObjective()`, compute all active-component Gaussians once
   per observation while accumulating the denominator, then reuse the assigned
   component value for the retained numerator.
3. Apply the same local reuse in `retainedSpatialObjectivePair()` for both
   candidate states and in `evaluateFinalRefinedState()` for the final support
   numerator. Do not introduce persistent storage or an additional observation
   traversal.
4. Preserve observation/component traversal, denominator and numerator
   accumulation order, membership predicates, response equations, line-search
   decisions, and final support decisions. Exact floating-point identity is
   not required, but deterministic extraction and replay quality are.
5. Use the existing local-state and final-evaluation phase timings as the
   direct work measurement. Do not add a per-observation counter that consumes
   part of the expected saving or advance the profile schema solely for this
   local implementation detail. The nearest checkpoint-16 profile measured
   roughly 23.44 local-state and 13.76 final-evaluation worker-seconds.
6. Run focused GCC and Clang `test_fiber_anchors`, `test_fiberlet_paths`, and
   `test_fiber_replay`. Run one canonical replay for an initial decision; if
   promising, run three canonical 32-thread, 5,000-base-voxel replays and
   compare total/anchor wall and CPU, local-state/final-evaluation worker time,
   populations, DP work, failures, artifact hashes, and peak RSS. If hashes
   change, also compare deterministic repeats, anchor axis/position
   distributions, downstream replay metrics, and generated visualizations.
7. Retain the checkpoint only if the enclosing anchor or objective-evaluation
   cost improves without unacceptable extraction or replay-quality changes.

### Checkpoint 17 Spec Update

- If retained, document that retained objective and final-state kernels reuse
  their already-computed per-component Gaussian values; equations and decision
  semantics remain unchanged.
- Advance the profile schema only if a new deterministic work counter is kept.

### Checkpoint 17 Documentation Update

- Update `volume-cartographer/docs/fiberlets.md` with the reuse path and measured
  result if retained.
- Record implementation, validation, benchmarks, and any rejected variants in
  `task_log.md`; add a concise accepted-checkpoint entry to `changelog.md` and
  complete `status.md`.

### Checkpoint 17 Result

Rejected. Three canonical runs preserved all extraction/DP populations, 2
greedy / 1 fiberlet failures, and byte-identical replay artifacts, but median
local-state evaluation rose from roughly 23.44 to 24.34 worker-seconds and
final evaluation rose from 13.76 to 14.12 worker-seconds. Median command wall
was effectively flat at 10.23 versus 10.25 seconds. The explicit stack arrays
were removed. The likely explanation is compiler common-subexpression
elimination of the pure repeated call in the committed code, while explicit
arrays increased register or spill pressure. No production specification,
user documentation, changelog, or profile-schema change is retained.

## Checkpoint 18: Radial Demand And Neighbor Reuse

1. Use committed checkpoint 16 (`7bb2830fd`) as the production baseline. Its
   three-run medians are 10.25 seconds command wall, 229.90 seconds total CPU,
   6.070 seconds anchor wall, 155.82 seconds anchor CPU, and 39.94
   peak-search worker-seconds.
2. Add temporary deterministic measurement for each peak component: prepared
   response records, total records passing the exact radial cutoff, and unique
   prepared records passing at least once. Track computed grid responses and
   uncached subpixel acceptance responses separately because only grid responses
   participate in demand cohorts.
3. Simulate candidate-by-block rejection for contiguous block sizes 16, 32,
   and 64. Report blocks tested/rejected, record slots surviving block
   rejection, metadata bytes, and component-weighted distributions. Simulate
   the exact conservative rule intended for production: outward-rounded float
   bounds expanded by the cutoff and reject only on axis separation. Do not use
   the simulation to skip production work yet.
4. Measure actual simultaneously batchable grid demand. Before each hill-climb
   neighborhood, separable-interpolation neighborhood, and final 3x3
   neighborhood, collect the feasible unique slots that are still missing from
   the response cache. Report cohort-size histograms and theoretical record
   loads before/after batching by cohort type. Requests reached in different
   iterations are not counted as batchable together.
5. Keep measurement overhead out of the retained implementation. Accumulate
   per-response counts locally and use one compact per-component touch array;
   remove temporary touch/block/cohort tracking after selecting a strategy.
6. Select exactly one implementation from the measurement:
   - If unique radial use is low, partition the existing sequential response
     stream into small contiguous blocks with conservative transverse AABBs.
     Skip a block only when its exact minimum squared distance from the
     candidate exceeds the radial cutoff, then retain the existing per-record
     check inside surviving blocks.
   - If unique radial use is high and repeated use is substantial, batch only
     the currently demanded uncached neighboring grid responses. Traverse the
     sequential response stream once and update candidate-local compensated
     accumulators in deterministic candidate order. Do not precompute the full
     peak grid.
7. Preserve the existing prepared-record layout, response arithmetic,
   per-candidate observation accumulation order, feasibility, hill-climb and
   tie behavior, acceptance checks, and deterministic execution. Small numeric
   changes are acceptable but are not required by either strategy.
8. For block rejection, add scalar-oracle tests including cutoff-boundary and
   randomized fixtures. For batching, add mixed cached/missing and duplicate
   request tests. In either case, verify request/computed/acceptance counters,
   tie/hill-climb behavior, and deterministic repeats; retain existing
   single-slot, ridge, bounded-subpixel, and profile tests. Run focused GCC and
   Clang anchor/path/replay suites.
9. Run one instrumented canonical replay to select the strategy. Run one
   initial optimized replay, then three final canonical 32-thread,
   5,000-base-voxel replays if promising. Compare peak-search worker time,
   total/anchor wall and CPU, peak response counts, populations, DP work,
   failures, artifact hashes, and peak RSS. If hashes change, additionally
   compare deterministic repeats, anchor axis/position distributions,
   downstream metrics, and visualizations.
10. Retain the checkpoint only if peak-search or enclosing anchor cost improves
   with acceptable deterministic replay quality. Remove rejected variants and
   record their measurements.

### Checkpoint 18 Spec Update

- If retained, document the selected exact block broad phase or demanded
  neighboring-response batching and its preserved demand/response semantics.
- Retain no measurement-only profile fields; advance the schema only if a
  generally useful production counter survives the experiment.

### Checkpoint 18 Documentation Update

- Update `volume-cartographer/docs/fiberlets.md` with the selected data flow and
  measured result if retained.
- Record measurement, variant selection, validation, and benchmark results in
  `task_log.md`; add an accepted result to `changelog.md` and complete
  `status.md`.

### Checkpoint 18 Result

Rejected. Measurement showed high simultaneous theoretical reuse, but direct
batching did not convert it into runtime savings. Full-cohort, width-four, and
width-two variants raised peak-search worker time from the 39.94-second
baseline to 46.41, 47.66, and 49.16 seconds respectively. The 16/32/64-record
block simulation could eliminate only 23.6%, 16.5%, and 8.5% of record visits,
so block metadata/check overhead was not implemented after batching failed.
All measured variants retained exact populations, failures, and replay
artifact bytes. Temporary measurement fields, response-cache APIs, tests, and
batching code were removed. No production spec, user-doc, changelog, or profile
schema change remains.

## Checkpoint 19: plain float tensor accumulation

1. Use committed checkpoint 16 (`7bb2830fd`) as the production baseline:
   median 10.25 seconds command wall, 229.90 seconds total CPU, 6.070 seconds
   anchor wall, 155.82 seconds anchor CPU, and approximately 35.5 worker-seconds
   in local tensor proposal work.
2. Change only the optional six-entry symmetric tensor histograms inside
   `robustDirectionProposal()` from compensated double sums to ordinary float32
   sums. Keep residual histograms, total masses, component assignment, robust
   cutoff selection, retained membership, traversal order, and all other
   accumulators unchanged.
3. Merge the at-most-256 retained residual bins into six ordinary double sums,
   then construct the existing double 3x3 matrix at the principal-axis solver
   boundary. Do not change the eigensolver or its uniqueness rules.
4. Exercise the production path through public anchor fitting rather than
   duplicating private tensor construction in tests. Cover representative one-
   and two-component fits plus low-mass, imbalanced, near-isotropic, and nearly
   equal leading-eigenvalue cases. Require finite normalized axes where the
   component remains unique and deterministic component removal otherwise.
5. Run focused GCC and Clang `test_fiber_anchors`, `test_fiberlet_paths`, and
   `test_fiber_replay` suites. Run the canonical 32-thread, 5,000-base-voxel
   replay three times if the first optimized run is promising.
6. Compare local tensor-proposal worker time, total/anchor wall and CPU,
   populations, DP work, failures, artifact hashes, anchor position/axis
   distributions if artifacts differ, and peak RSS. Quality comparison must
   report matched-anchor count and axis-angle and position-delta p50/p95/max,
   plus deterministic repeat agreement. Retain only if the target phase or
   enclosing anchor extraction improves with similar deterministic replay
   quality.

### Checkpoint 19 Spec Update

- If retained, document that robust direction tensors use ordinary float32
  accumulation and that exact floating-point identity is not part of the
  extraction contract. The profile schema remains unchanged.
- If rejected, leave production specifications unchanged and record only the
  measured experiment.

### Checkpoint 19 Documentation Update

- If retained, update `volume-cartographer/docs/fiberlets.md` and
  `planning/changelog.md` with the accumulator boundary and measured result.
- Record implementation, validation, quality comparison, and benchmarks in
  `task_log.md`, then complete `status.md` whether accepted or rejected.

### Checkpoint 19 Result

Inconclusive. The float-bin and ordinary-double experiments were run while the
computer had competing work, so none of their timing measurements are valid.
Both partial variants were removed. Their quality observations may inform test
coverage, but no performance or retention conclusion is drawn.

## Checkpoint 20: scalar-specialized robust proposal

1. Keep `robustDirectionProposal()` shared and scalar-generic. Select `float`
   for production `CompactFiberAnchorObservation` ranges and `double` for the
   public `FiberAnchorObservation` path.
2. In the float specialization, keep observation positions/directions,
   component axes/positions, pivot, Gaussian constants/results, alignments,
   scores, masses, residual histograms, and tensor histograms in float32 for
   the complete per-observation loop. Preserve compact directions' existing
   already-normalized contract; retain public double normalization behavior.
   Build proposal-local scalar component/config/pivot state once before the hot
   loop and use typed literals and scalar OpenCV vectors throughout.
3. Convert the two 256-bin residual histograms and total masses once after the
   hot loop, then call the existing double robust-cutoff implementation. Derive
   each double total mass by summing the converted bins rather than maintaining
   a separate float total. Make residual binning scalar-generic so compact
   residuals are not widened before bin selection. Merge retained float tensor
   bins into six ordinary doubles and retain the existing double eigensolver
   and uniqueness policy.
4. Do not convert centroid, spatial-objective, peak-search, final-evaluation,
   persistent anchor state, or public output arithmetic in this checkpoint.
   This isolates the measured local tensor-proposal phase.
5. Exercise the private compact specialization through the existing public
   `extractFiberAnchors*` sampler fixtures, not by rounding inputs and calling
   the double-only `fitFiberCellAnchors()`. Cover compact and double one-/two-
   component paths plus presence-floor equality, equal component scores,
   residual-bin boundaries, trim-mass limits, low mass, imbalanced weights,
   near-isotropic tensors, and nearly equal leading eigenvalues. Verify
   deterministic repeats, finite normalized retained axes, stable component
   removal, and bounded compact-vs-double axis/position differences rather than
   bit identity.
6. Build `vc_fiberlets` and run focused GCC and Clang anchor/path/replay tests.
   Do not run the canonical replay benchmark until the user explicitly reports
   the computer is free.
7. Once approved, run alternating compensated-baseline and all-float builds on
   the canonical workload, with three valid runs of each. Compare tensor work,
   total/anchor wall and CPU, populations, DP work, failures, deterministic
   hashes, matched-anchor axis/position distributions where available, emitted
   route deltas, and peak RSS.
8. Before retention, require GCC and Clang validation and note that macOS/arm64
   CI compilation remains the portability gate.

### Checkpoint 20 Spec Update

- If retained after valid measurements, document the compact-production float
  robust-proposal boundary and double summary/eigensolver boundary. Do not
  change the profile schema.
- If rejected, remove the implementation and retain only the experiment log.

### Checkpoint 20 Documentation Update

- Record implementation and test results now, then append controlled benchmark
  results when authorized. Update `volume-cartographer/docs/fiberlets.md` and
  `planning/changelog.md` only after retention.

### Checkpoint 20 Result

Accepted. Three alternating runs reduced median tensor-proposal worker time
from 25.63 to 23.74 seconds, anchor CPU from 143.16 to 140.75 seconds, anchor
wall from 5.504 to 5.461 seconds, and command wall from 9.65 to 9.58 seconds.
The optimized runs were deterministic and retained 2,603 anchors, 2,560 graph
nodes, 26,494 edges, and 2 greedy / 1 fiberlet failures. The 352 emitted route
points differed from baseline by at most 1.3764e-6 base voxels.

## Checkpoint 21: compact-float spatial objectives

1. Use committed checkpoint 20 (`397c1cbf3`) as the baseline. Its controlled
   medians are 9.58 seconds command wall, 5.461 seconds anchor wall, 140.75
   seconds anchor CPU, and approximately 22.3 worker-seconds in local state
   evaluation.
2. Keep the shared `retainedSpatialObjective()` and paired variant generic.
   Select float32 only for production `CompactFiberAnchorObservation` ranges;
   preserve the public expanded-observation path's double arithmetic,
   direction normalization, compensated sums, and behavior.
3. For the compact specialization, convert active component axes/positions,
   pivot, Gaussian constants, and presence floor once per objective call. Keep
   observation positions/directions, Gaussian values, alignment, numerator,
   and denominator in ordinary, uncompensated float32 through the complete
   scan. Compact directions retain their existing pre-normalized contract.
   Widen only the final one or two objective ratios to double for unchanged
   tolerance and acceptance logic. The paired function remains one fused scan
   with independent baseline/candidate accumulators; do not add Gaussian reuse
   or any checkpoint-17 behavior.
4. Preserve observation/component traversal, membership bytes, candidate and
   backtracking order, bounds/clamping, accepted persistent component state,
   profile counters, and profile schema. Do not change centroid proposal,
   robust direction proposal, final evaluation, peak search, or serialization.
   Preserve denominator semantics exactly: every finite-position observation
   contributes one Gaussian per active component before any validity, presence,
   direction, assignment, or trimming check gates numerator evidence.
5. Add extraction-level coverage through the private compact path. Compare a
   stable position-refinement fixture with the public double fitter using
   bounded anchor position/axis tolerances, and cover deterministic repeats near
   an objective tie or backtracking decision boundary. Cover denominator-only
   observations with invalid directions, NaN/below-floor presence, unassigned
   or trimmed numerator evidence, zero denominator, axial/transverse cutoff
   equality and adjacent-float cases, and a translated fixture at realistic
   large prediction coordinates. Existing public-path tests must remain exact.
6. Build `vc_fiberlets` plus focused anchor/path/replay tests with GCC and
   compile the touched production path with Clang. Run `git diff --check`.
7. Benchmark three alternating compact-float and checkpoint-20 baseline runs
   on the canonical 32-thread, 5,000-base-voxel replay. Compare local state-
   evaluation work, total/anchor wall and CPU, peak RSS, populations, DP work,
   failures, deterministic hashes, and emitted route displacement. Record
   build type/flags, host, cache state, warmup policy, exact run order, inputs,
   and commits. If artifacts differ materially, additionally compare matched-
   anchor count, axis/position p50/p95/max, accepted backtracking-depth
   distributions, and visualizations.
8. Retain only if local-state or enclosing anchor cost improves without an
   unacceptable deterministic quality change. Remove the experiment otherwise.

### Checkpoint 21 Spec Update

- If retained, extend the compact-production float precision boundary to the
  fixed-direction spatial objective scans while documenting that persistent
  state and acceptance remain double. Do not change the profile schema.
- If rejected, leave production specifications unchanged and retain only the
  measured experiment log.

### Checkpoint 21 Documentation Update

- Record implementation, tests, and controlled results in `task_log.md`.
  Update `volume-cartographer/docs/fiberlets.md` and `planning/changelog.md`
  only if the checkpoint is retained, and close all items in `status.md`.

### Checkpoint 21 Result

Rejected and removed. Across three alternating pairs, median local state-
evaluation worker time improved from 22.48 to 20.03 seconds (10.9%), but median
tensor-proposal work regressed from 23.47 to 31.32 seconds. Anchor CPU rose
from 140.07 to 145.48 seconds, command wall from 9.44 to 9.59 seconds, and total
CPU from 211.40 to 216.80 seconds. All six artifacts were byte-identical and
quality populations were unchanged, identifying a code-generation/locality
regression rather than a numerical-quality failure. Production code and tests
were restored to committed checkpoint 20.

## Checkpoint 22: isolated compact-float spatial objectives

1. Use committed checkpoint 20 (`397c1cbf3`) as the baseline. Preserve its
   shared library before rebuilding so optimized and baseline runs can be
   alternated without a runtime branch in either hot path.
2. Add a source-private fiber-anchor objective module to `vc_fiber_tracer`.
   Move the complete retained spatial-objective equation behind that module's
   internal interface rather than copying a second private implementation into
   `FiberAnchors.cpp`. Move the exact compact record and compensated-sum helper
   once into a source-private shared header; pass a module-specific component
   and scalar-config value instead of exposing fitter state. Add no installed or
   public API.
3. Provide two explicitly selected module paths. Expanded public observations
   retain double arithmetic, direction normalization, compensated accumulation,
   and denominator rules. Indexed compact production observations use the
   checkpoint-21 float32 arithmetic, ordinary float accumulators,
   pre-normalized directions, and a fused paired scan.
4. Borrow observation and index storage through spans. The indexed compact
   kernel iterates logical indices, reads the observation from
   `storage[indices[logical]]`, and reads assignment/membership from
   `[logical]`. Validate logical cardinalities and every underlying index. Do
   not materialize, reorder, or copy per-cell observations. Convert only the
   two component states, pivot, and required scalar configuration once per call.
5. Preserve all-site denominator semantics: each finite-position observation
   contributes to every active denominator before validity, presence,
   direction, assignment, or robust-membership checks gate numerator evidence.
   Preserve candidate/backtracking order, persistent state, counters,
   acceptance, peak search, final evaluation, and output.
6. Add focused coverage through a test-only include of the source-private
   module. Cover expanded and indexed compact inputs, empty input, invalid
   cardinality/index rejection, nonconsecutive and repeated underlying indices,
   denominator-only invalid/NaN/below-floor observations, zero/one/two active
   components, cutoff boundaries, fused paired versus two single evaluations,
   realistic large coordinates, and deterministic repeats. Require exact
   public-double single/paired and fitter behavior; use checkpoint-21 geometric
   tolerances only for compact-float versus double comparisons.
7. Build `vc_fiberlets` and run linked GCC and Clang `test_fiber_anchors`,
   `test_fiberlet_paths`, and `test_fiber_replay`. Add the new source through
   CMake, use standard C++23 `std::span`, and run `git diff --check`;
   macOS/arm64 CI remains the portability gate. If the local Clang tree cannot
   be linked without installation, record that deviation before falling back to
   a production-flag Clang translation-unit compile.
8. Before source edits, rebuild checkpoint 20 from its clean production source,
   save and hash its shared library, and prove the explicit loader path selects
   it. Alternate three isolated and three checkpoint-20 runs through explicit
   isolated library paths on the canonical 32-thread, 5,000-base-voxel replay.
   Record run order, binary/library hashes, compiler/flags, host/load, cache and
   warmup state, exact command and inputs. Compare local-state and tensor-
   proposal worker times, total/anchor wall and CPU, peak RSS, populations, DP
   work, failures, artifact hashes, and emitted route displacement. If artifacts
   differ materially, also compare matched-anchor position/axis and accepted
   backtracking-depth distributions, downstream metrics, and visualizations.
9. Retain only if the compact objective saving survives while tensor-proposal
   work returns to baseline noise and enclosing anchor or total cost improves.
   Otherwise remove the module and retain only the experiment record.

### Checkpoint 22 Spec Update

- If retained, document the compact float objective precision boundary and the
  separate objective module while keeping persistent state and final
  evaluation double. Do not change the profile schema.
- If rejected, leave production specifications unchanged and record only the
  measured experiment.

### Checkpoint 22 Documentation Update

- Record design, independent review, validation, benchmark method, and result
  in `task_log.md`. Update `volume-cartographer/docs/fiberlets.md` and
  `planning/changelog.md` only if retained, then close checkpoint items in
  `status.md`.

### Checkpoint 22 Result

Accepted. Three alternating isolated/baseline pairs measured median command
wall at 9.17/9.56 seconds, total CPU at 201.84/212.93 seconds, anchor wall at
5.050/5.425 seconds, and anchor CPU at 130.74/140.80 seconds. The isolated
objective reduced median local-state work from 22.59 to 13.86 worker-seconds;
tensor-proposal work also remained isolated and improved from 23.72 to 22.78
worker-seconds instead of repeating checkpoint 21's 31.32-second regression.
All six replay artifacts were byte-identical, all work/population counters and
accepted backtracking depths matched, and every run retained 2 greedy / 1
fiberlet failures. Median RSS changed by +0.2%. The module and precision
boundary are retained without a profile-schema change.

## Checkpoint 23: float final anchor evaluation

### Scope and implementation

1. Extend the isolated `FiberAnchorObjectives` module with the complete final
   refined-state reduction: per-component all-site Gaussian denominator,
   retained aligned numerator, retained presence mass, assigned count, and
   combined objective.
2. Keep compact production observations in float32 throughout position,
   direction, presence, Gaussian, dot-product, and accumulation arithmetic.
   Widen only the fixed-size final summary when it crosses into persistent
   `RefinedEvaluation`/anchor output state. Do not preserve double merely for
   historical numeric identity.
3. Run the expanded public final-evaluation entry point through the same
   float32 equation as production. Its double-valued input fields are narrowed
   at observation access and its fixed-size result is widened at return. This
   deliberately tests a broader float boundary; compensated double is not
   retained merely for legacy identity.
4. Preserve all-site denominator semantics, logical assignment/membership
   indexing, component counts, and existing acceptance/serialization code.
   Validate compact source indices and all input cardinalities before scanning.
5. Remove the old templated final-evaluation scan from `FiberAnchors.cpp` once
   both callers dispatch to the shared module. Keep no runtime compatibility
   branch.

### Quality and performance gates

1. Add direct module tests for zero/one/two components, repeated and
   nonconsecutive compact indices, unusable denominator-only evidence, assigned
   counts/presence masses, deterministic repeats, and compact-versus-expanded
   agreement at realistic coordinates.
2. Add threshold-sensitive extraction coverage and compare retained/rejected
   support decisions. Exact floating-point or artifact identity is not a gate;
   stable populations, comparable axes/positions/support values, unchanged
   replay-failure behavior, and deterministic output are required.
3. Run focused GCC and linked Clang suites for `test_fiber_anchors`,
   `test_fiberlet_paths`, and `test_fiber_replay`, plus `git diff --check`.
4. Preserve the checkpoint-22 shared library, verify loader choice, and run
   three alternating canonical replay pairs. Report min/median/max command
   wall, total CPU, anchor wall/CPU, final-evaluation worker time, peak RSS,
   populations, DP work, replay failures, and support-value differences.
5. Retain only if the final kernel and enclosing runtime improve without a
   material quality regression; otherwise remove and record the experiment.

### Measured layout correction

- The initial implementation placed final evaluation beside the checkpoint-22
  objective kernels. Although final-evaluation work improved, it perturbed the
  existing objective code enough to regress enclosing runtime. Split final
  evaluation into its own translation unit and extract common observation,
  Gaussian, and direction primitives into one private shared header before the
  retention measurement. This is a code-generation isolation correction, not
  a second optimization or duplicated implementation.

### Spec update

- If retained, state in `planning/specs.md` that compact production final
  support reduction uses float32 and widens only fixed-size output summaries;
  deterministic quality and acceptance behavior, not numerical identity, are
  the contract.

### Documentation update

- If retained, update `volume-cartographer/docs/fiberlets.md` with the isolated
  final-evaluation ownership and production precision boundary.

### Changelog update

- If retained, add a checkpoint-23 entry with measured performance and quality.
  If rejected, record it only in the active task plan and log.

### Result

Accepted after strict translation-unit isolation. The initial co-located layout
was rejected because it regressed the existing objective kernel despite making
final evaluation faster. Restoring `FiberAnchorObjectives.cpp` byte-for-byte
and placing final evaluation in its own translation unit returned objective and
tensor work to baseline noise. Three fresh alternating pairs measured median
final-evaluation work at 13.11/13.79 worker-seconds, anchor CPU at
130.55/132.40 seconds, total CPU at 202.89/204.33 seconds, and command wall at
9.22/9.26 seconds for candidate/baseline. All six artifacts were byte-identical
with unchanged populations, DP relaxations, and replay failures.

## Checkpoint 24: end-to-end float anchor and fiberlet state

### Scope and implementation

1. Change anchor observations, fitting configuration, peak offsets, robust
   residual/cutoff state, objective components, refinement state, retained
   anchors, support scores, and numeric diagnostics to float32. Remove the
   expanded-double fitter specialization and its checked narrowing layer; both
   direct and indexed inputs use the same float representation.
2. Change fiberlet path configuration, costs, candidate endpoint geometry,
   sampled endpoint state, path points, visual metrics, and graph geometry to
   float32. This includes curved-domain/layer geometry, Hermite construction,
   transported frames, interpolation fractions, scoring voxels, prepared node
   scoring, transition geometry, and DP incoming state. Keep integer
   lattice/cell/index state unchanged.
3. Convert existing external `Vec3d` reference polylines and normal-sampler
   results once at their boundary. Keep replay/reference-distance arithmetic
   and elapsed-time/process-accounting fields double because those are outside
   anchor/fiberlet extraction and are not repeated observation or node state.
   Change the extraction point predicate to `Vec3f` so node enumeration does
   not widen every point. Convert the shared stored-prediction sample itself to
   float if all users permit it; otherwise introduce one float extraction
   sample adapter and never retain the shared double sample in anchor/fiberlet
   state.
4. Serialize float fields directly as JSON numbers and use float-appropriate
   OBJ precision. Existing version-1 and current version-2 artifacts remain
   readable by accepting finite, float-representable JSON numeric values into
   float fields and rejecting out-of-range values. No schema bump, runtime
   compatibility branch, or duplicate double representation is retained.
5. Audit containers and helper return types so no float result is widened to
   double and stored, then narrowed again in a hot loop. Constants and standard
   math overloads used by extraction must be float unless an external boundary
   explicitly requires double.
6. Generalize the shared principal-axis tensor implementation to support float
   without copying its algorithm. Anchor and fiberlet extraction use the float
   instantiation; unrelated existing double callers retain the same shared
   implementation.
7. Split float path/graph costs from replay's double cumulative accumulators.
   Graph geometry and edge/transition costs remain float; replay sums and
   reference-distance results remain double without changing the extraction
   records.
8. Use this precision inventory throughout the audit: extraction configuration,
   geometry, samples, objectives, costs, diagnostics, statistics, and graph
   data are float; timing/process accounting, external reference/replay
   geometry, and cold external scale metadata are double; lattice, indices,
   counts, and flags remain integer.

### Quality and validation gates

1. Add exact compile-time size assertions only for compact trivially-copyable
   hot records and field-type assertions for aggregate records containing
   standard-library objects. Add float serialization/load coverage for legacy
   version-1 and current version-2 artifacts, round trips, and out-of-range
   rejection.
2. Update anchor, fiberlet-path, graph, and replay tests to compare with
   float-appropriate tolerances. Preserve deterministic ordering, integer work
   topology, validity semantics, and threshold strictness.
3. Build and run focused GCC and Clang tests for anchors, paths, graph, and
   replay. Do not run the canonical replay benchmark while the user reports the
   CPUs busy.
4. Once approved, run three canonical 5,000-base-voxel replays and compare
   min/median/max wall and CPU time, memory, anchor/fiberlet/graph populations,
   geometry deltas, DP work, and replay failures against commit `07176ccd6`.
5. Retain only if outputs are deterministic and quality remains comparable.
   Exact artifact hashes and double-rounding identity are not acceptance gates.
6. Audit convergence, degeneracy, eigenvalue uniqueness, owner-boundary,
   support, merge, NMS, and angle tolerances for float semantics. Preserve
   strict comparison direction and add exact/adjacent-float boundary tests;
   do not blindly retain double-scale epsilon constants where they are inert in
   float arithmetic.

### Spec update

- Replace the mixed persistent-double precision rules with one float32 anchor
  and fiberlet extraction contract. Document external double inputs as boundary
  conversions only. Explicitly replace the expanded-public-double,
  persistent-anchor-double, exact-endpoint-double, and final-result-widening
  clauses. Timing/process accounting, external replay/reference geometry, and
  cold scale metadata are the only intentional doubles at subsystem boundaries.

### Documentation update

- Update `volume-cartographer/docs/fiberlets.md` to describe the end-to-end
  float representation and version-1 reader behavior.

### Changelog update

- Checkpoint 24 is retained after three deterministic canonical runs improved
  median wall time by 2.8%, total CPU by 1.7%, and peak RSS by 5.4% while
  preserving comparable replay quality.

## Post-checkpoint-24 optimization queue

Use commit `d2229bf6f` as the new measured baseline. Implement one checkpoint
at a time and stop for review after its focused tests and three canonical
replays. The ordered queue is:

### Checkpoint 25: parallel corner-set finalization

1. Keep candidate preparation and worker-local corner ownership unchanged.
2. Convert each worker's corner set to a vector and sort it concurrently using
   the existing bounded worker count. Keep each vector's `storedVoxelLess`
   order and retain the current deterministic pairwise sorted-unique merge.
3. Account concurrent vector capacity in peak-memory estimates and propagate
   worker exceptions before merging.
4. Add focused coverage comparing serial reference finalization with the
   parallel result for empty, overlapping, duplicate-heavy, and uneven worker
   sets.
5. Compare corner-merge wall/CPU, total wall/CPU, RSS, exact sampled-voxel
   order, graph populations, DP work, artifact hash, and replay failures.

Result: retained. The production path allocates exact destination capacities on
the calling thread, then concurrently fills and sorts worker vectors before the
existing merge tree. GCC and Clang focused suites pass all 49 cases. Three warm
canonical runs produced:

| metric | checkpoint 25 min / median / max | checkpoint 24 min / median / max | median change |
|---|---:|---:|---:|
| command wall | 8.08 / 8.23 / 8.35 s | 8.86 / 8.96 / 8.98 s | -8.1% |
| total CPU | 201.66 / 202.91 / 205.02 s | 197.05 / 199.47 / 200.68 s | +1.7% |
| anchor wall | 5.011 / 5.124 / 5.236 s | 4.963 / 4.984 / 5.032 s | +2.8% |
| anchor CPU | 130.65 / 131.49 / 133.02 s | 128.80 / 129.92 / 130.80 s | +1.2% |
| fiberlet wall | 2.488 / 2.519 / 2.534 s | 3.236 / 3.298 / 3.313 s | -23.6% |
| fiberlet CPU | 68.26 / 69.31 / 69.89 s | 66.01 / 67.33 / 67.68 s | +2.9% |
| corner finalization wall | 0.265 / 0.268 / 0.268 s | about 1.05 s | about -74.5% |
| corner finalization CPU | 2.462 / 2.463 / 2.497 s | about 1.36 s | about +81% |
| peak RSS | 2,060,168 / 2,060,332 / 2,077,188 KiB | 1,994,740 / 2,007,020 / 2,018,200 KiB | +2.7% |

All runs retained 170,778 sampled voxels in the exact prior order and produced
artifact SHA-256
`f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`.
Anchor/graph/DP populations and the 2 greedy / 1 fiberlet failures were
unchanged. The cold first run was excluded after recording 1,670 major faults
and 773,656 filesystem-input blocks.

### Checkpoint 26: sparse paged corner bitmap

Proceed only if checkpoint 25 leaves corner collection or merging material.

1. Measure worker-set sizes, duplicate ratios, occupied spatial pages, and page
   reuse before choosing a page edge.
2. Replace per-worker `unordered_set` insertion with a bounded sparse page
   directory and dense bits inside occupied pages. Cache the last page during
   each candidate's locally coherent insertion stream.
3. Merge corresponding pages with bitwise OR and enumerate set bits in exact
   stored-voxel order. Do not sample conservative extra corners.
4. Compare all 405.7 million insertion attempts, 170,778 canonical unique
   voxels, corner collection/merge time, memory, and complete replay quality.

Result: retained. The worker-local hash sets are replaced by sparse `16^3`
pages with dense 4,096-bit occupancy. An immediate-page fast path plus an
eight-entry worker cache served 405.65 million of 405.73 million insertion
attempts without a page-directory lookup; only about 75 thousand probes
remained. The 32 workers occupied about 6,017 pages and 4.645 million
worker-local unique voxels, which OR-reduced to 199 pages and the exact prior
170,778 globally unique voxels. One final Z/Y/X sort preserves the established
sampler order.

Three warm canonical runs measured command wall at 7.65 / 7.77 / 7.84 seconds,
total CPU at 195.97 / 196.18 / 197.57 seconds, fiberlet wall at
2.151 / 2.165 / 2.181 seconds, corner finalization wall at
0.0190 / 0.0196 / 0.0202 seconds, and peak RSS at
1,692,124 / 1,697,516 / 1,715,028 KiB. Against checkpoint 25 medians, command
wall improved 5.6%, total CPU 3.3%, fiberlet wall 14.0%, corner finalization
92.7%, and peak RSS 17.6%. Worker-local corner collection itself rose to about
8.23 CPU-seconds because bitmap insertion remains in the preparation hot loop;
the much cheaper finalization and lower transient storage still improve the
enclosing workload. All runs retained exact artifact SHA-256
`f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`,
unchanged populations and DP work, and 2 greedy / 1 fiberlet failures.

### Checkpoint 27: ready-cell anchor scheduling

1. Add measurement-only per-sampling-group and per-cell fitting durations to
   distinguish tile sampling latency from fitting-tail imbalance.
2. If fitting tails explain the roughly 26/32 effective-core utilization,
   retain sampled group storage under the existing memory budget and enqueue
   its cells independently once sampling and gradient construction complete.
3. Keep each cell's observation order and fitting arithmetic unchanged. A
   group is released only after all of its cells finish.
4. Compare anchor wall/CPU, worker utilization and tail, sample reuse, peak
   memory, deterministic output, and replay quality.

Result: retained. Measurement-only schema-19 timing found complete group jobs
at 1.013 seconds median, 3.172 seconds p95, and 4.027 seconds maximum, while
individual cells were 8.11 milliseconds median, 11.85 milliseconds p95, and
28.78 milliseconds maximum. The implementation keeps group owners and tile
storage bounded as before, but publishes each prepared tile's cells to a
cooperative queue that every extraction worker can drain. Owners wait for all
dependent cells before releasing observations or advancing overlap reuse. An
existing admission gap was also closed: two tiles are paired only if their
staged peak fits `maximumConcurrentSampleBytes`.

Three warm canonical runs measured command wall at 6.97 / 6.97 / 6.99 seconds,
total CPU at 193.91 / 194.12 / 194.37 seconds, anchor wall at
4.251 / 4.262 / 4.264 seconds, anchor CPU at 126.97 / 126.98 / 127.42 seconds,
and peak RSS at 1,684,328 / 1,687,504 / 1,709,368 KiB. Against checkpoint 26
medians, command wall improved 10.3%, anchor wall 15.9%, total CPU 1.1%, and
anchor CPU 2.2%; fiberlet wall remained effectively flat. Median maximum group
job duration fell to 1.647 seconds. All runs retained exact replay SHA-256
`f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`,
exact fiberlet-route SHA-256
`1ec7df7b8d2417ddc762652be3bf0057eef8b93a329a24d36f02a8837465014b`,
unchanged populations and DP work, and 2 greedy / 1 fiberlet failures.

### Checkpoint 28: extraction-wide raw prediction reuse

1. Replace pair-local tile overlap reuse with bounded exact-union partitions.
   Partition canonical-order tiles conservatively from their dense sample-byte
   upper bound so arbitrarily large extractions retain the current streaming
   behavior. Within each partition, represent the exact union as sorted
   `(z,y)` rows containing merged X intervals and one contiguous float32 raw-
   sample array; do not use a per-voxel hash or sample conservative padding.
   The canonical workload should form one partition; overlap crossing a
   partition boundary may be sampled once per partition.
2. Sample every partition-union coordinate exactly once in deterministic
   bounded batches. Use a separately admitted sampling-worker count, and pass
   lower-level sampler thread count `1` on every call. Allocate the final
   shared vector before workers write disjoint ranges. Record exceptions by
   canonical batch index, join all workers, and propagate the earliest batch
   failure deterministically before any tile reads shared samples.
3. Account row/interval metadata and shared samples plus the maximum concurrent
   coordinate and expanded-sampler scratch during sampling. Separately account
   metadata/shared samples plus `fitWorkerCount * maximumTileWorkingBytes`
   during fitting, including gradients, compact observations, queues, and
   per-worker cell scratch. Reject only when one legal tile partition cannot
   fit, matching the existing minimum supported workload.
4. Materialize each tile's dense raw sample vector by copying contiguous X
   ranges from the immutable shared rows. Preserve canonical Z/Y/X tile order,
   tile-local gradient construction, compact-observation construction, the
   ready-cell scheduler, fitting arithmetic, and canonical error reporting.
5. Keep shared sampling and cell fitting as separate bounded phases within
   each partition for the
   first implementation. Measure the removed sampler work against any lost
   sampling/fitting overlap; do not retain the change merely because submitted
   voxel count falls.
6. Make each tile the fitting-job and ownership unit. A tile owner prepares one
   immutable observation vector, publishes its cells, helps the shared ready-
   cell queue until its own cells complete, then releases the vector and marks
   the tile complete. Idle workers stop only after every tile job in the
   partition is complete. Cell failures remain stored and rethrown by lowest
   canonical cell index after all partitions finish.
7. Bump the extraction profile schema. Replace sampling-group fields with
   partition count/durations and add maximum accounted live bytes, shared-
   union bytes, batch count/size, shared sampling wall/CPU, and tile-copy work.
   `submittedPredictionVoxels` counts union voxels across partitions;
   `reusedPredictionVoxels` is tile occurrences minus submissions; the exact
   whole-extraction union remains a diagnostic and can be lower than submitted
   count only when partition boundaries repeat overlap.
8. Add focused coverage for overlapping and disjoint tiles, merged and split
   row intervals, one- and multi-partition budgets, deterministic single/multi-
   worker sampling, reversed multi-batch failures, wrong sampler result sizes,
   gradient enabled/disabled, clipped and explicit-cell runs, invalid samples,
   progress callback failures, checked arithmetic, and exact output parity.
   Use structured keys and checked `size_t` arithmetic; make no assumptions
   about record padding or signed shifts.
9. Run focused GCC and Clang anchor/path/replay tests, then three warm canonical
   replays with host-load checks. Compare total and anchor wall/CPU, prediction
   sampling and tile-copy work, RSS, populations, DP work, failures, and replay
   hashes against checkpoint 27. Record exact command, dataset, build type,
   min/median/max, maximum accounted live bytes, and union-planning time. Add a
   synthetic budget-pressure validation. Remove the implementation if combined
   wall time or memory is not acceptable.
10. Update `status.md` incrementally and record implementation decisions,
    deviations, failed variants, and measurements in `task_log.md`.

#### Checkpoint 28 spec update

- Replace pair-local sampling-group reuse with the retained bounded exact-union
  partition contract, including deterministic coordinate order, memory
  admission, ownership/termination, and profile counter meanings.

#### Checkpoint 28 documentation update

- Update `volume-cartographer/docs/fiberlets.md` with the shared-row layout,
  phase boundary, memory accounting, and measured benchmark result.
- Record accepted behavior in `planning/changelog.md`; keep rejected variants
  and measurements only in `planning/task_log.md`.

#### Checkpoint 28 result

- Retained schema-20 bounded exact-union sampling after three warm canonical
  QuickBuild runs. Median command wall improved from 6.97 to 6.82 seconds,
  anchor wall from 4.262 to 4.069 seconds, and anchor CPU from 126.98 to 111.61
  seconds. Prediction submissions fell from 26,741,712 to the exact 6,162,456-
  voxel union; median peak RSS changed from 1,687,504 to 1,675,944 KiB.
- All runs retained exact replay SHA-256
  `f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`,
  unchanged populations and DP work, and 2 greedy / 1 fiberlet failures.
- Focused validation passed 83 GCC and 83 Clang anchor cases, 49 GCC path
  cases, and 6 GCC replay cases.

### Checkpoint 29: peak Gaussian acceleration

1. Isolate only the transverse radial Gaussian inside the existing scalar
   sequential `responseAt()` scan. Keep the precomputed axial Gaussian, radial
   cutoff, response/evidence record layouts, traversal, compensated sums,
   response cache, hill-climb order, acceptance checks, and tie policy
   unchanged. Do not restore rejected response batching, CSR, or counting-sort
   layouts.
2. Add one shared private peak-Gaussian helper used by production and direct
   tests. Its input is the nonnegative normalized exponent
   `distanceSquared / (2 * sigma^2)`. Start with a process-wide 512-interval
   float table with 513 entries over `[0, 8]` and linear interpolation. The
   default three-sigma cutoff ends at exponent `4.5`, so it is fully covered
   while the 2,052-byte table remains small. Acquire the immutable table once
   per peak search and use an inline lookup with no per-observation function
   call or static-initialization guard. Use the existing `expf` calculation for
   negative, non-finite, or above-domain inputs; preserve exact `1` at zero and
   use only portable standard C++.
3. Add deterministic dense-grid, interval-midpoint, exact-knot, fixed-seed
   random, and `nextafter` endpoint tests against `expf`. Require maximum
   absolute and relative error no greater than `3.5e-5`, and mean relative error
   no greater than `2.2e-5` over `[0,8]`. Add fallback, monotonicity, positivity,
   and deterministic multi-thread coverage. Preserve the caller's exact radial
   cutoff rather than making the helper return zero. Do not add a production
   runtime selector or duplicate the complete peak implementation for tests;
   compare complete exact/candidate binaries externally.
4. Build and run focused GCC and Clang anchor tests. Run one canonical screening
   replay and compare peak-search worker time, anchor wall/CPU, command wall/CPU,
   RSS, anchor/graph populations, DP work, failures, and artifact/route geometry
   against checkpoint 28. Reuse the discrete, separable, and joint peak
   positions already serialized in diagnostic records. Report changed discrete
   peak count, matched peak-position p50/p95/maximum displacement, unmatched
   diagnostics, matched axis-angle differences, and emitted-route displacement.
   If the table regresses or has no useful signal,
   remove it before considering a polynomial. Do not retain multiple runtime
   implementations or compatibility switches.
5. For a viable table, alternate three warm checkpoint-28 and three candidate
   replays from separate libraries under the same QuickBuild executable,
   command, inputs, thread count, warmed data, and host-load checks. Measure
   min/median/max. Require deterministic candidate artifacts, unchanged anchor
   count and diagnostic identity, peak-position p95 displacement at most `0.05`
   prediction voxels and maximum at most `0.5`, maximum axis-angle difference
   at most `1e-4` radians, unchanged replay failure counts, graph population
   changes below `1%`, DP-work changes below `2%`, and emitted-route p95/maximum
   displacement at most `0.1`/`1.0` base voxels. Report changed discrete peaks
   separately. Exact artifact identity is not required. Retain only a repeatable
   enclosing-runtime improvement; otherwise remove the experiment.
6. Update profile schema only if a new permanent counter is required. Record
   approximation bounds and retained performance/quality in
   `volume-cartographer/docs/fiberlets.md` and `planning/changelog.md`; record
   rejected variants and full measurements only in `planning/task_log.md`.

#### Checkpoint 29 spec update

- If retained, document the bounded interpolated transverse Gaussian, exact
  cutoff ownership, fallback domain, determinism, and permitted numerical
  differences. If rejected, leave production specifications unchanged.

#### Checkpoint 29 documentation update

- Document the direct approximation error and canonical performance/quality
  result in `volume-cartographer/docs/fiberlets.md` only for a retained variant.
- Record the retention decision in `planning/changelog.md`; retain all trial
  details in `planning/task_log.md`.

#### Checkpoint 29 result

- Rejected the 513-entry linear-interpolation lookup. Its measured maximum
  absolute/relative errors were `3.03e-5`/`3.07e-5`, but a direct paired replay
  was tied at 6.84 seconds command wall while peak-search work increased from
  31.11 to 31.43 worker-seconds.
- Rejected the degree-six range-reduced polynomial. It improved approximation
  error to `1.41e-5` maximum absolute and `2.81e-5` maximum relative, and
  reproduced the exact checkpoint-28 artifact, but raised peak-search work to
  35.54 worker-seconds, anchor wall to 4.204 seconds, and command wall to 6.97
  seconds.
- Removed both implementations and their direct tests. Production code,
  specifications, user documentation, and changelog remain unchanged. Final
  validation passes 83 GCC and 83 Clang anchor cases, 49 GCC path cases, and
  6 GCC replay cases.

### Checkpoint 30: one-pass membership reuse experiment

1. Reuse the robust assignments and retained membership computed from the
   geometry at the start of the terminal outer iteration and used to derive its
   accepted geometry update, instead of unconditionally recomputing only
   membership at the moved geometry after the refinement loop. In symbols,
   change `M(S_n) -> S_(n+1) -> M(S_(n+1))` to
   `M(S_n) -> S_(n+1)` for terminal membership. Axes,
   centroids, backtracking, accepted positions, iteration stopping, component
   removal, peak search, and final-support arithmetic remain unchanged.
2. Preserve general multi-iteration behavior: every next outer iteration still
   begins with a fresh robust proposal at the preceding iteration's accepted
   geometry. Only the terminal membership-only refresh is removed. A component
   removed as non-unique continues to restart the same iteration and cannot
   leave stale component indices in the retained membership.
3. Add a focused fixture where the accepted transverse move changes which
   evidence would be assigned or retained by a post-move refresh. Verify that
   final support and peak evidence attribution use the membership that justified
   the accepted move. Also cover `maximumIterations > 1` with a forced
   non-converged update that differs from one iteration, plus early convergence,
   to prove ordinary between-iteration recomputation still occurs. Add explicit
   compaction fixtures where original component zero is removed and component
   one survives as compact index zero, where removal follows an earlier accepted
   iteration, and where both components are removed; verify diagnostic IDs,
   assigned counts, support, and peak evidence remain attached correctly.
4. Use the existing `localTensorProposalWorkSeconds` and
   `localTensorObservationVisits` counters to measure the removed scan; do not
   add a profile field unless those aggregate counters cannot distinguish the
   expected reduction in focused coverage. The removed terminal proposal makes
   two complete observation scans, so expect `2 * observation_count` fewer
   logical visits for every surviving terminal fit. Record that aggregate robust
   mass/outlier counters no longer include a terminal refresh. Keep the
   checkpoint-28 shared library as the external reference. Do not retain a
   runtime selector or compatibility branch.
5. Build and run focused GCC and Clang anchor tests, then one canonical
   screening replay. Compare command/anchor wall and CPU, robust proposal work
   and visits, RSS, matched-anchor position and projective-axis-angle p50/p95/max,
   unmatched anchor counts, anchor/graph/fiberlet populations, DP work, emitted
   route displacement, and greedy/fiberlet failures. Generate baseline and
   candidate replay visualization artifacts for close/crossing fibers, large
   moves, Gaussian/axial and robust-cutoff boundaries, and support-threshold
   decisions. Exact numeric or artifact identity is not required. A screening
   run may reject early; retention requires three interleaved baseline/candidate
   runs with min/median/max. Retain only if quality remains comparable and the
   removed pass yields a repeatable enclosing-runtime improvement; otherwise
   restore the refresh and log rejection.
6. Do not repeat the rejected inline-membership experiment: this checkpoint
   changes when membership is recomputed rather than adding its predicate to
   every downstream hot scan.

#### Checkpoint 30 spec update

- If retained, specify that terminal final evaluation and peak evidence
  attribution use membership computed from the geometry at the start of the
  terminal iteration, while every additional outer iteration still recomputes
  membership. Correct the stale specification default for
  `maximum_iterations` from two to the implemented and documented value one.
  If rejected, leave production semantics unchanged but still correct that
  pre-existing default-value inconsistency.

#### Checkpoint 30 documentation update

- If retained, update `volume-cartographer/docs/fiberlets.md` with terminal
  membership semantics and measured performance/quality, and record the result
  in `planning/changelog.md`. Keep rejected trial detail only in
  `planning/task_log.md`.

#### Checkpoint 30 result

- Rejected terminal membership reuse after one canonical screening replay.
  Command wall improved from the checkpoint-28 median `6.82` to `6.45` seconds,
  anchor wall from `4.069` to `3.734` seconds, and anchor CPU from `111.61` to
  `100.48` seconds. The candidate nevertheless reduced retained anchors from
  2,603 to 2,568, graph nodes from 2,562 to 2,528, graph edges from 26,445 to
  26,082, and DP relaxations from 62,873,000 to 62,214,882. Fiberlet replay
  failures increased from one to two. Greedy output stayed exact, while the
  fiberlet route had 351 points instead of 352.
- Removed the production and focused-test experiment after that decisive quality
  regression; three paired timing runs and visualization review were therefore
  unnecessary. Production fitting semantics, user documentation, and changelog
  remain unchanged. Corrected only the pre-existing specification typo that
  listed two outer iterations as the default instead of one.

### Checkpoint 31: remaining DP throughput

1. Use committed checkpoint 28 (`1675886b7`) as the production baseline. Test
   scheduling independently from transition arithmetic, scoring, and DP data
   representation.
2. After candidate preparation, construct a stable work permutation ordered by
   descending retained-node count. Retained-node count is already available and
   is a cheap deterministic heuristic, but it is not assumed to predict direct
   index initialization or reached-state/transition work exactly. Break ties by
   original search index and measure its correlation with complete candidate
   solve duration.
3. Let search workers claim slots in that permutation while all candidates,
   prepared data, errors, and solve profiles remain indexed by original search
   index. Preserve each candidate's node order, edge order, state order,
   arithmetic, and result placement.
4. Record complete per-candidate solve duration and per-worker busy duration so
   the screening run exposes completion-tail balance rather than only aggregate
   CPU time. Add focused coverage for the pure descending-cost permutation,
   stable ties, empty input, one-worker/multi-worker output equivalence,
   canonical profile/index placement, and deterministic lowest-original-index
   exception selection when failures complete out of order. Do not expose a
   runtime selector or change serialized artifacts.
5. Build and run focused GCC and repository-local Clang fiberlet-path tests.
   Check host load, then run one canonical 32-thread screening replay. Compare
   command wall/CPU, fiberlet search wall/CPU, DP work counters, candidate and
   graph populations, route displacement, replay failures, artifact hash, and
   peak RSS. Largest-first may otherwise hide a transient-memory regression by
   starting the largest searches together.
   Retain only if the enclosing search or command wall time improves beyond
   ordinary run noise without a quality regression; otherwise remove it and
   record the rejection.
6. If scheduling is retained or rejected, profile transition-cost arithmetic
   and candidate completion tails before testing a portable vectorized
   transition kernel with a scalar fallback for Ubuntu/macOS and amd64/arm64.

#### Checkpoint 31 spec update

- If retained, specify only that independent fiberlet candidates may be
  scheduled in estimated descending work order while their externally visible
  order and all within-candidate decisions remain deterministic. If rejected,
  leave production specifications unchanged.

#### Checkpoint 31 documentation update

- If retained, document the scheduling estimate and measured result in
  `volume-cartographer/docs/fiberlets.md`, and add a concise changelog entry.
  Keep rejected experiment details only in `planning/task_log.md`.

#### Checkpoint 31 result

- Rejected largest-candidate-first scheduling after one canonical screening
  replay. Search wall remained `1.226` seconds, while worker busy times were
  already tightly grouped from `1.214` to `1.219` seconds under the existing
  dynamic queue. Retained-node count had only `0.476` Pearson correlation with
  complete solve duration.
- The candidate reproduced the exact checkpoint-28 artifact and replay quality,
  but command wall was `6.92` seconds versus the `6.82`-second checkpoint-28
  median. All scheduling and temporary timing code was removed. Production
  source, profile schema, specifications, user documentation, and changelog
  remain unchanged.

### Checkpoint 32: DP transition-cost profiling

1. Keep checkpoint 28 production behavior and first split existing DP worker
   time with shared boundary timestamps into initialization/source seeding,
   intermediate-layer propagation, final sink evaluation, traceback/result
   materialization, and signed residual. The parent DP timer excludes local
   vector/cache destruction; retain and document that existing boundary.
2. Only for a deterministic hash sample of canonical candidate indices, time
   the two interleaved propagation sections per reached node: outgoing-edge
   construction including lazy scoring/deviation validation, and reached-state
   transition scoring/relaxation. Report sampled propagation residual and work
   counts as sampled evidence, not extrapolated exact totals.
3. Record the existing generated/valid/reused edge, reached-state, transition
   lookup, and relaxation counts beside the new work timings. Actual scored
   transitions are `valid_edges + reused_edges`; transition lookups and
   successful relaxations are different quantities.
4. Add focused finite/nonnegative and signed-residual reconciliation invariants
   for successful, no-path, direct, zero-length, and empty-corridor candidates.
   Build GCC and Clang path tests, check host load, and run one canonical
   profiling replay. Emit temporary diagnostics on a separate versioned profile
   line and remove them if their overhead materially changes search wall time.
5. Use the measured dominant phase to plan one isolated portable optimization.
   Do not introduce SIMD, architecture-specific code, altered transition order,
   or numerical changes during this profiling checkpoint.

#### Checkpoint 32 spec update

- None. This checkpoint measures existing implementation phases and does not
  change search semantics or persistent output.

#### Checkpoint 32 documentation update

- Record profiling findings in `planning/task_log.md`. Do not update user docs
  or the changelog unless a subsequent retained optimization changes the
  implementation materially.

#### Checkpoint 32 result

- The canonical profile attributed `33.59` of `38.93` DP worker-seconds to
  propagation, `3.93` to initialization/source seeding, `1.39` to sink
  evaluation, `0.017` to traceback, and `0.004` to residual work. Search wall
  was `1.230` seconds versus the `1.226`-second uninstrumented screening run, so
  profiler overhead was negligible.
- A deterministic 808-candidate sample covered 103,097 reached nodes. Within
  sampled propagation, outgoing construction/lazy scoring used `0.2372` of
  `0.5709` seconds (41.5%), transition scoring/relaxation used `0.3145` seconds
  (55.1%), and residual control/allocation work used `0.0192` seconds (3.4%).
  The exact checkpoint-28 artifact and replay quality were preserved.

### Checkpoint 33: remove redundant DP direction normalization

1. Directions produced as `delta / length` after positive finite length
   validation are already unit directions to float precision. Add the explicit
   finite checks missing from the current internal edge paths, then pass those
   values directly into prepared metric scoring rather than immediately taking
   a second square root and division through
   `prepareFiberLocalUnitDirection()`.
2. Collapse `DpIncoming`'s duplicate ordinary/metric direction and length
   fields into one float direction and length. Rename the equivalent outgoing
   edge fields if needed for clarity. Keep all geometry, traversal, metric,
   state, accumulation, and tie ordering unchanged.
3. Keep prediction-axis and normal-axis normalization unchanged; they are not
   geometry-created directions. Add focused finite/near-epsilon/oblique geometry
   coverage and multi-layer paths that exercise source and ordinary predecessor
   states. Existing prepared-scoring and serial/parallel deterministic artifact
   tests remain integration gates; comparisons affected by the removed second
   normalization use tight tolerances rather than exact equality.
4. Build GCC and Clang path/scoring tests. Run one canonical screening replay
   with the checkpoint-32 profile retained temporarily, comparing outgoing,
   transition, search, command, populations, artifact determinism, route
   displacement, failures, and RSS. Small floating-point differences are
   allowed, but retain only a measurable enclosing gain with comparable replay
   quality.

#### Checkpoint 33 spec update

- If retained, clarify that DP geometry-created directions are validated once
  and then consumed by prepared metric scoring as unit directions. No file or
  user-facing configuration changes.

#### Checkpoint 33 documentation update

- If retained, record the prepared-direction invariant and measured result in
  `volume-cartographer/docs/fiberlets.md` and add a concise changelog entry.
  Keep rejected details only in `planning/task_log.md`.

#### Checkpoint 33 result

- Rejected. Three instrumented candidate runs measured median search wall
  `1.216` seconds and DP worker time `38.466` seconds, versus
  `1.230`/`38.932` for the identically instrumented checkpoint-28 path. The DP
  worker reduction appeared to be 1.2%, but the required final uninstrumented
  build measured `1.227`/`38.830`, effectively the checkpoint-28 baseline.
- All candidate runs were deterministic. DP relaxations changed by eight and
  accumulated costs changed slightly as expected, but emitted greedy and
  fiberlet route points remained exact, with unchanged populations and 2 greedy
  / 1 fiberlet failures. The direction change, temporary checkpoint-32
  profiling, and proposed docs/spec/changelog updates were removed.

### Checkpoint 34: inline prepared DP metric implementation

1. Preserve the public `FiberLocalScoring` API and extract its alignment,
   smoothness, and prepared-metric implementations into three private inline
   primitives in one source-private header shared by the public wrappers and
   `FiberPaths.cpp`. The private prepared primitive must call the private
   alignment and smoothness primitives. Do not duplicate equations or
   introduce a DP-specific scoring implementation.
2. Route only the already-prepared DP transition path through the private
   inline helper. Keep the generic normalization path, invalid-prediction path,
   normal-aware fallback, arithmetic expression order, transition order,
   accumulation, and tie policy unchanged. The source and sink paths may keep
   using the public API because they are not the measured hot loop.
3. Keep the private header under `core/src/fiber_tracer/` and use only portable
   standard `inline`, without force-inline attributes. Verify with `nm -D` that
   all public scoring symbols retain their signatures and visibility. Inspect
   the optimized propagation call site specifically and require that it no
   longer calls the interposable prepared-metric, alignment, or smoothness
   symbols; calls elsewhere in the library remain legitimate.
4. Extend exact generic/prepared branch coverage for invalid prediction, null
   and invalid current prediction, sign flips, normal-aware and isotropic
   fallback, degenerate directions, nonpositive lengths, and non-finite inputs.
   Run focused GCC and repository-local Clang fiberlet-path tests plus GCC
   replay tests. Require exact artifact hash, populations, DP counters, routes,
   and failures because this is an implementation-placement change with
   identical arithmetic.
5. Build separate uninstrumented baseline and candidate libraries from the same
   compiler configuration. Check host load, screen once, then, if viable, run
   three alternating warm baseline/candidate pairs. Report min/median/max for
   command wall/CPU, anchor wall/CPU, fiberlet wall/CPU, search wall/CPU, DP
   worker time and counters, RSS, populations, routes, and failures. Retain only
   a repeatable enclosing gain outside paired run noise; otherwise restore the
   external-call path and log rejection.

#### Checkpoint 34 spec update

- None. Scoring equations, inputs, precision, and decisions remain unchanged.

#### Checkpoint 34 documentation update

- If retained, document only the shared inline prepared-scoring implementation
  and measured result in `volume-cartographer/docs/fiberlets.md`, with a concise
  changelog entry. Keep rejected details only in `planning/task_log.md`.

#### Checkpoint 34 result

- Retained. The optimized propagation loop has no calls to exported or
  out-of-line private scoring helpers, while all five public scoring symbols
  retain their exported signatures and visibility.
- Three alternating, uninstrumented QuickBuild baseline/candidate pairs reduced
  median search wall from `1.1652` to `1.0509` seconds (9.8%), search CPU from
  `36.4426` to `32.9779` seconds (9.5%), DP worker time from `36.8775` to
  `33.2330` seconds (9.9%), and fiberlet wall from `2.1290` to `2.0077` seconds
  (5.7%). Median command wall improved from `7.87` to `7.75` seconds.
- All six artifacts had SHA-256
  `904c39d08e39c6b7b65ac95fd47d28d50e254a33609201c92aef71c6cc131308`,
  with exact populations, DP counters, routes, and 2 greedy / 1 fiberlet
  failures. Peak RSS medians were `1,611,932` KiB baseline and `1,601,256` KiB
  candidate.

### Checkpoint 35: lazy isotropic smoothness evaluation

1. In the shared source-private smoothness primitive, move isotropic angle and
   penalty evaluation into the invalid-normal fallback. In the normal-aware
   branch, evaluate the isotropic angle only when either projected tangent is
   degenerate and the existing tangent-angle fallback requires it.
2. Preserve the exact equations and arithmetic order for every returned cost.
   Do not add a DP-specific implementation, approximation, runtime option, or
   platform-specific intrinsic. Both exported wrappers and the inlined DP path
   must continue to use the same private implementation.
3. Add an independent test-local implementation of the pre-checkpoint
   equations; do not use generic/prepared comparisons that share the private
   helper as the parity oracle. Compare every returned field exactly for valid
   projected tangents, `normalValid=false`, a valid flag with zero normal, each
   one-sided projected-tangent degeneracy, and two-sided degeneracy.
4. Run focused GCC path/replay tests and repository-local Clang path tests.
   Inspect baseline and candidate GCC DP call-site control flow, and candidate
   Clang control flow where practical, to establish whether the compiler had
   already sunk the eager `acos` and confirm both fallback paths retain it.
5. Build an uninstrumented baseline at commit `0d104426e` with the same CMake
   options as the candidate. Check host CPU load before every run, screen once,
   then use counterbalanced warm order `B/C, C/B, B/C`. Record the exact
   command, dataset, build flags, output directories, and min/median/max.
   Compare command, fiberlet, search, DP-worker and CPU times, RSS, populations,
   DP counters, routes, failures, and artifact hashes. Retain only a repeatable
   enclosing gain with byte-identical output; otherwise remove the source and
   test changes.

#### Checkpoint 35 spec update

- None. This checkpoint changes only when an existing intermediate is
  evaluated; equations, precision, inputs, outputs, and decisions are
  unchanged.

#### Checkpoint 35 documentation update

- If retained, document the shared local-scorer lazy fallback evaluation and
  its fiberlet-DP measurement in
  `volume-cartographer/docs/fiberlets.md`, add a concise changelog entry, and
  record full measurements in `planning/task_log.md`. If rejected, keep only
  the experiment result in the active task records.

#### Checkpoint 35 result

- Retained. GCC baseline code generation kept the isotropic `acos` before
  normal validation, while the candidate partitions it into the two fallback
  paths; a valid-normal/valid-tangent transition now evaluates one angle rather
  than two. Candidate Clang code generation also keeps the fallback calls
  behind their branches.
- Three counterbalanced uninstrumented QuickBuild pairs reduced median search
  wall from `1.0368` to `0.9723` seconds (6.2%), search CPU from `32.5708` to
  `30.5559` seconds (6.2%), DP worker time from `32.7720` to `30.7132` seconds
  (6.3%), and fiberlet wall from `1.9605` to `1.9091` seconds (2.6%). Median
  command wall improved from `7.61` to `7.55` seconds (0.8%).
- All six measured artifacts and the screening artifact had SHA-256
  `904c39d08e39c6b7b65ac95fd47d28d50e254a33609201c92aef71c6cc131308`,
  with exact populations, DP counters, routes, and 2 greedy / 1 fiberlet
  failures.

### Checkpoint 36: prepared outgoing-edge smoothness

1. Extract candidate-side normal-aware smoothness preparation into the shared
   source-private local-scoring implementation. The descriptor will own the
   normal from which it was prepared and retain only the normalized candidate
   tangent, candidate normal angle, and compact branch state needed by the
   existing equations. This makes mismatched descriptor/normal use impossible
   and avoids retaining the redundant candidate normal component.
2. Add a shared metric-cost entry point that accepts an already prepared
   candidate smoothness descriptor. Keep the existing prepared-direction entry
   point as an on-demand wrapper over the same implementation so exported
   public callers, source/sink transitions, and DP transitions cannot diverge.
   Do not copy scoring equations into `FiberPaths.cpp`. Preserve the invalid
   candidate-prediction early return before on-demand preparation so invalid or
   non-finite geometry does not evaluate projection or inverse sine.
3. Prepare and store the descriptor in each valid stack-local `DpEdge`, using
   that edge's direction and destination normal. Reuse it for all reached
   incoming states. Preserve outgoing-edge generation order, transition order,
   accumulation order, tie policy, precision, and returned arithmetic exactly.
4. Include the source-private header directly from the focused path test and
   compare the candidate-prepared metric route against an independent
   test-local implementation of the complete legacy metric equation. Cover
   valid normal-aware scoring, invalid and zero normals, NaN/Inf normals with a
   valid flag, previous-only and candidate-only projected-tangent degeneracy,
   both-sided degeneracy, zero/non-finite step directions, invalid candidate
   prediction, and nonpositive edge lengths. Run focused GCC path/replay tests
   and repository-local Clang path tests. Inspect optimized DP code to verify
   that candidate projection, normalization, and inverse-sine are outside the
   incoming-state reuse loop.
5. Record baseline/candidate `sizeof(DpEdge)`, prepared-descriptor count, valid
   edge count, and reuse count, and inspect generated code for stack growth or
   spills. Use operation counts and disassembly rather than per-transition
   timers that would distort this short hot loop; aggregate DP/search timing
   remains the runtime gate.
6. Build an uninstrumented QuickBuild `VC_TESTING=ON` baseline at commit
   `08b4ea9cb` with CMake options matching the candidate. Use the canonical
   Paris4 5,000-length replay, 32 threads, two maximum iterations, warm cache,
   and distinct output directories. Check host CPU load, screen once, then run
   three counterbalanced warm pairs in order `B/C, C/B, B/C`. Record the exact
   command and dataset paths plus min/median/max command wall/CPU, anchor
   wall/CPU, fiberlet wall/CPU, search wall/CPU, DP worker time, RSS,
   populations, DP counters, routes, failures, and hashes. Retain only a
   repeatable enclosing gain with byte-identical replay output; otherwise
   remove the experiment.

#### Checkpoint 36 spec update

- None. The optimization reuses identical candidate-side intermediates within
  one reached node; scoring equations, precision, inputs, outputs, and
  decisions remain unchanged.

#### Checkpoint 36 documentation update

- If retained, update `volume-cartographer/docs/fiberlets.md` to document
  outgoing-edge candidate smoothness preparation and add a concise changelog
  entry. Record the complete experiment and measurements in
  `planning/task_log.md`. If rejected, retain only the experiment record in the
  active planning files.

#### Checkpoint 36 result

- Retained. The private descriptor owns its destination normal and stores one
  normalized candidate tangent, one candidate normal angle, and compact branch
  state. `DpEdge` grows from 24 to 56 bytes, adding 288 stack bytes for its
  fixed nine-edge array. Optimized code places the candidate `asin` in outgoing
  construction and leaves only the previous-side `asin` in transition reuse.
- Three counterbalanced uninstrumented QuickBuild pairs reduced median search
  wall from `0.9701` to `0.9505` seconds (2.0%), search CPU from `30.2511` to
  `29.8211` seconds (1.4%), DP worker time from `30.6398` to `29.9875` seconds
  (2.1%), and fiberlet wall from `1.8996` to `1.8814` seconds (1.0%). Median
  command wall was effectively flat at `7.53/7.52` seconds because anchor work
  dominates the enclosing run.
- All six artifacts had SHA-256
  `904c39d08e39c6b7b65ac95fd47d28d50e254a33609201c92aef71c6cc131308`,
  with exact populations, DP counters, routes, and 2 greedy / 1 fiberlet
  failures.

### Checkpoint 37: prepared two-sided alignment inputs

1. Extend the shared source-private local-scoring implementation with explicit
   incoming-side and candidate-side alignment descriptors. The incoming
   descriptor owns the previous direction, sign-oriented current prediction
   axis, and clamped previous/current factor. The candidate descriptor owns the
   candidate direction, sign-oriented candidate prediction axis, clamped
   presence, and clamped candidate/candidate-axis factor. Combine candidate
   alignment state with the existing candidate smoothness descriptor so one
   valid `DpEdge` has one atomically constructed, authoritative candidate
   direction. The descriptor owns all data and retains no references into the
   reallocating lazy node-scoring cache.
2. Add one shared metric entry point that consumes both prepared descriptors.
   Preserve the invalid candidate-prediction early return before preparation.
   Keep the four pair-dependent dots and the exact factor sequence: presence,
   previous/candidate, stored previous/current-axis,
   previous/candidate-axis, current-axis/candidate,
   current-axis/candidate-axis, stored candidate/candidate-axis, followed by
   `(1-score)*candidateLength`. Store individual clamped factors rather than a
   pre-multiplied product. Keep metric entry points as on-demand wrappers over
   the same implementation and do not copy equations into `FiberPaths.cpp`.
   The standalone `fiberLocalAlignmentLoss()` API must retain its current raw,
   caller-oriented semantics and must not implicitly orient axes.
3. In DP, prepare incoming alignment once after reconstructing each reached
   state and prepare candidate alignment/smoothness once per valid outgoing
   edge. Reuse both descriptors for all transitions between them. Preserve
   outgoing generation order, reached-state order, transition order,
   accumulation order, tie policy, precision, and returned costs. Restrict the
   prepared route to reused interior transitions; source initialization, sink
   finalization, and the two-layer shortcut retain their existing paths and
   endpoint semantics. `DpIncoming` continues to own its raw direction for sink
   scoring alongside the transient prepared interior state.
4. Add focused tests that compare the private prepared route with the
   independent complete legacy metric oracle. Cover valid and invalid current
   prediction, current/candidate sign flips, null current prediction,
   non-finite presence, exact-zero and negative-zero dots, zero/non-finite
   directions and lengths, valid-but-nonfinite prediction axes, invalid
   candidate prediction with poisonous remaining inputs, and existing normal-
   aware/fallback smoothness branches. Add deterministic randomized bitwise
   comparison and a DP fixture with differently oriented incoming states
   sharing outgoing edges. Run GCC path/replay tests and repository-local Clang
   path tests. Verify public local-scoring symbols remain exported.
5. Inspect optimized GCC and Clang DP code to confirm sign orientation and
   side-only clamps are outside the transition reuse loop. Record descriptor
   sizes and stack growth; reject or revise the layout if the extra state
   causes spills that erase the targeted gain.
6. Establish one generic invariant local fiberlet-replay benchmark launcher
   command under the build tree, reusable by all future checkpoints. Switch
   checkpoint, baseline/candidate executable, commit, output root, and run label
   only through launcher configuration, never by changing the approved command.
   Build a matching QuickBuild `VC_TESTING=ON` baseline at commit `d9cebed3f`.
   Check host CPU load before and after every run, screen once, then run
   counterbalanced warm pairs `B/C, C/B, B/C` on the canonical Paris4
   5,000-length replay with 32
   threads and two maximum iterations. Record command, dataset, build options,
   min/median/max command/CPU/anchor/fiberlet/search/DP/RSS values, populations,
   counters, failures, and hashes. The launcher creates fresh output
   directories and records executable, commit, configuration, and output path.
   Retain only a repeatable targeted and enclosing gain with byte-identical
   replay output.

#### Checkpoint 37 spec update

- None. Preparation reuses values within one DP node/edge combination while
  preserving the exact equations, precision, multiplication order, decisions,
  and output.

#### Checkpoint 37 documentation update

- If retained, update `volume-cartographer/docs/fiberlets.md` with two-sided
  alignment preparation, add a concise `planning/changelog.md` entry, and
  record complete measurements in `planning/task_log.md`. If rejected, remove
  source/test/documentation changes and retain only the experiment result in
  the active task records.

#### Checkpoint 37 result

- Retained. Candidate-side preparation is owned by one combined 64-byte metric
  descriptor; replacing the old separate direction and smoothness fields grows
  `DpEdge` from 56 to 76 bytes. A 28-byte incoming descriptor is transient per
  reached state. Optimized code places side-only orientation and clamping at
  edge/state preparation, outside the pairwise transition scan.
- Three counterbalanced warm QuickBuild pairs reduced median search wall from
  `0.9495` to `0.9279` seconds (2.3%), search CPU from `29.9074` to `29.0604`
  seconds (2.8%), DP worker time from `29.9843` to `29.2899` seconds (2.3%),
  and fiberlet wall from `1.8768` to `1.8693` seconds (0.4%). Median total CPU
  was effectively flat/slightly lower at `200.63/200.48` seconds. Command wall
  was `7.52/7.58` seconds because the untouched anchor phase was slower in all
  candidate slots.
- Every measured artifact retained exact populations, DP counters, routes,
  2 greedy / 1 fiberlet failures, and SHA-256
  `904c39d08e39c6b7b65ac95fd47d28d50e254a33609201c92aef71c6cc131308`.
