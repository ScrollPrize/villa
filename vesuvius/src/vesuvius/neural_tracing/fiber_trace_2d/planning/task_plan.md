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
