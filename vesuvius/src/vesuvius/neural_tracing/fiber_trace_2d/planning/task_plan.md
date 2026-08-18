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
