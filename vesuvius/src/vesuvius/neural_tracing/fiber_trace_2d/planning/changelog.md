# 2026-08-25: durable Fiberlet crop trace artifacts

- Made a sparse Fiberlet Zarr `traces` dataset the authoritative crop-tracing
  output. It losslessly stores complete float64 base-coordinate paths, seed
  data, global selected-route cost, and traced length, with strict inventory,
  ownership, ordinal, and atomic-publication validation.
- Split `vc_fiber_trace_chunk` into explicit `trace` and `visualize` commands.
  Trace mode reopens the published artifact before generating line outputs;
  visualization-only mode needs no source Fiberlets or normal volume.
- Added deterministic cost-density histograms and ten rank-decile OBJ subsets
  while retaining byte-identical all/direction/anchor OBJ geometry.

# 2026-08-25: deterministic parallel Fiberlet crop tracing

- Parallelized independent anchor-seed graph traversal while retaining serial
  strongest-first result integration, coverage suppression, counters, and OBJ
  order. Speculative traces covered by an earlier result are discarded.
- Extracted a reusable immutable, storage-key-based replay source and extended
  the existing chunk-route bulk materializer with complete route geometry,
  cost profiles, and stored joins. Chunk reads are prefetched in batches;
  tracing performs no cache queries.
- Removed the four-worker cache-contention workaround. Host-CPU tracing reduced
  the 16-seed trace phase from 17.57 s to about 3-4 s on the Paris4 1024-base
  crop while preserving the original OBJ byte-for-byte.
- Replaced synchronized seed batches with bounded continuous computation and
  ordered ticket commits. Converted the materialized graph to contiguous
  records with borrowed directional adjacency/route views, eliminating route
  copies during rejected lookahead. The 500-attempt Paris4 crop remains
  byte-identical and drops from 86.80 s to 79.66 s wall time.
- Replaced per-branch committed-set and route-vector copies with compact
  parent-linked lookahead ancestry. Terminal candidates retain indices and use
  exact linear minimum selection instead of materializing and sorting every
  route. The same 500-attempt crop remains byte-identical; three Release runs
  take 46.81-47.01 s wall with 34.24-34.54 s tracing.

# 2026-08-25: Fiberlet producer cache contract v3

- Moved anchor and Fiberlet generation to one new producer-contract namespace
  after detecting different stale/current records under the unpublished v2
  identity. Default runs regenerate into v3 directories; explicit incompatible
  roots remain strict errors and old directories are not modified.
- Added compiler identity/version and build configuration to the producer
  fingerprint after GCC/Clang generated 2,526 versus 2,528 focused Fiberlets at
  hard thresholds. Mixed-toolchain chunks can no longer share a dataset.
- Preserved strict endpoint-evidence validation while allowing only bounded
  finite float32 reconstruction roundoff across GCC/Clang; scoring and stored
  payloads are unchanged. Scheduled failures now report the owner key,
  terminal cache status, and nested generator cause.
- Raised the exact chunk-route per-entry state guard from one million to five
  million, the smallest tested bound that completes the 1024-base staged
  Paris4 workload without partial results.

# 2026-08-24: memory-first staged Fiberlet overlays

- Added a shared bounded write-back LRU for invocation-local reduction layers.
  Later boxes and stages read serialized anchor and paired prefix/route chunks
  directly from memory; asynchronous atomic writes occur only under shared
  cache pressure, with pending buffers still charged to the budget.
- Logical stage hashing now combines metadata, spilled chunks, and latest
  in-memory chunks without forcing a flush. Exact stage ID/payload hashes and
  final populations remain unchanged.
- The hot four-stage Paris4 Release workload improved from a 3.93 s median to
  2.90 s; its 5.56 MB of temporary payloads required no disk spills.

# 2026-08-24: arbitrary staged Fiberlet graph reduction

- Replaced point-query graph construction with chunk-granular anchor,
  prefix, and route loading, cache-free parallel transition construction, and
  reusable worker pools. On the hot Paris4 512-base two-stage workload,
  Release wall time fell from an 8.09 s median to 2.46 s while retained-ID and
  complete overlay-payload hashes remained exact.
- Exact entry searches use fixed worker-owned partitions and reusable
  thread-local scratch, with no synchronization in the per-trace loop. The
  stage-one search alone improved from 0.852 s at one thread to 0.108 s at 32.
- Corrected stage tables to report inside anchors, all incident Fiberlets, and
  per-box interior Fiberlets in that stage's complete box union, with separate
  original, inherited input, and output counts. The joint table remains scoped
  to the full selected bbox.
- Replaced the unpublished fixed two-stage driver with repeatable
  `--stage SIDE,OFFSET_X,OFFSET_Y,OFFSET_Z` analysis passes over a selected
  base-coordinate bbox.
- Added same-layout sparse anchor/Fiberlet overlays with missing-chunk
  fall-through, explicit empty overrides, strict prefix/route pairing, and
  record-exact monotone replacement checks.
- Made overlapping boxes consume earlier removals in deterministic order and
  report per-stage plus joint anchor/all/interior reductions. Derived layers
  are temporary; initial on-demand caches remain reusable and unchanged.

# 2026-08-24: two-stage regional Fiberlet reduction diagnostic

- Added globally reusable aligned stage-one reduction chunks and centered
  half-offset stage-two analysis without changing the authoritative anchor or
  Fiberlet caches.
- Bounded stage two to exactly the selected stage-one owners; incident-halo
  requests outside that set are in-memory empty views and cannot start or
  publish extra preprocessing.
- A hot Paris4 512/256 run reused all eight stage-one chunks, completed stage
  two in 1.4 s, and reduced 13,750 incident Fiberlets to 4,168 (69.69%) and
  5,730 internal Fiberlets to 618 (89.21%).
- Added conservative directed reachability removal, complete unused-anchor
  removal, exact bidirectional physical macros, disjoint directed-chain
  detection, and atomic forced-continuation descriptors for each centered
  stage-two box. Distinct routes are retained; macros remain in-memory and
  reference their original ordered Fiberlet geometry and objective scalars.
- On the same hot crop, post-stage-two simplification removed 51/1,515 anchors,
  merged 73/4,168 physical Fiberlets into 4,095 macros, and found 1,041
  one-successor directed states with 2.19-macro mean forced rollout length.

# 2026-08-24: chunk-local optimal Fiberlet-route statistics

- Added a shared on-demand cached-graph diagnostic that finds every exact cheapest
  directed entry-to-first-exit route through a half-open base-coordinate box.
  It shares regular replay's strict join-angle and normal/tangent-aware join
  scoring, rejects all revisited anchors, generates missing canonical chunks
  through the existing preprocessor/cache/LRU path, and reports the retained
  union and route distributions without pruning graph payloads.
- In a populated 128-base Paris4 box, exact optima used 55/62 anchors and
  1,135/1,369 incident fiberlets. Only 45/252 internal Fiberlets survived,
  giving an 82.14% internal reduction after boundary-crossing edges are
  excluded. The bounded cold shared-cache fill took 183.71 s and the hot
  analysis took 0.29 s.

# 2026-08-22: geometrically weighted fiberlet substep costs

- Restored stored whole-fiberlet and join scoring as the default replay cost
  evaluator. The slower subsegment/grid evaluator is available explicitly as
  `--cost-mode stepped`; its profile-weight default remains 1. Aggregate mode
  does not load cost profiles, and cache identity is unchanged.
- Added replay-only stepped-mode `--cost-profile-weight` in `[0,1]` to blend
  each fiberlet's decoded average density with its decoded subsegment densities
  before geometric weighting.
- On the full Paris4 radius-768 hot-cache corridor, delayed `W=0.99` falloff
  produced 5/5/3/2/2 failures at profile weights 0/0.25/0.5/0.75/1. With
  `W=1`, profile weight zero exactly reproduced the two aggregate-baseline
  failure arcs.
- A matched-terminal-weight lookahead sweep found no zero-or-one-failure
  configuration. H384/T0.25 retained two failures; every H512 setting produced
  four, H768 produced three or four, and H768/T0.5 hit the route-state limit.
- Removed two long-route quadratic paths discovered after the 5k validation.
  Score initialization now reads cumulative prefix scalars and visits only the
  checkpoint/lookahead suffix, while logical-route cleanup uses a bounded
  persistent cursor instead of rescanning the complete registry.
- Made `fiberlet-replay --threads` one shared budget for its concurrently
  running greedy and fiberlet evaluators. This removes the 32-logical-CPU
  oversubscription cliff without changing search ordering.
- A hot-cache full Paris4 radius-768 replay crossed the former 65-percent
  slowdown point at 12 seconds and completed in 22.34 seconds wall time, with
  14 greedy and 5 fiberlet failures.
- Repaired the scorer after the first profile integration changed more than the
  checkpoint-forward term. Ranking now preserves the authoritative unweighted
  prefix through the checkpoint and integrates decoded subsegment costs only
  from the checkpoint through the horizon. Decoded profiles are not rescaled to
  the separately stored whole-edge cost, and one integrator handles every
  weight, including `W=1`.
- Restored conservative relaxed cost-to-go pruning and incremental scalar score
  state. Ordinary cached arc queries no longer load route payloads; profiles use
  a bounded decision-local cache. On the Paris4 5k hot-cache benchmark, repaired
  `W=1` had zero fiberlet failures and median wall/CPU ratios of 1.18x/1.21x
  versus pinned revision `64e5341`.
- Added opt-in delayed geometric falloff through replay-only `--cost-delay`;
  zero preserves immediate decay and 192 base voxels covers half the default
  lookahead before decay begins.
- Persisted fixed-sqrt `uint16` cost density for every selected DP route
  segment and changed exact and bounded graph lookahead to integrate those
  profiles on a checkpoint-rooted base-distance grid.
- Added replay-only `--cost-weight` and `--cost-step` controls. Weighted ranking
  remains separate from the unchanged five-component committed diagnostics.
- On the Paris4 5,000-base-voxel interval, `W=0.99` at both 16- and
  32-base-voxel spacing completed with zero fiberlet failures, matching the
  unweighted stored-profile run.
- Added bounded decision windows and common-start `fiberlet-replay --arc`
  diagnostics. Two Paris4 wide-radius failures now have constrained-radius
  proxy routes and ranked alternative cost decompositions without full-fiber
  reruns; the proxy is explicitly not treated as a same-graph oracle.
- Clipped retained diagnostic polylines exactly at their checkpoint and
  lookahead boundaries. The diagnostic-filter regression now also proves that
  filtering leaves selected geometry and failure counts unchanged.

# 2026-08-22: fractional fiberlet endpoint evaluation

- Replaced the unpublished combined q1/u8 matrix row with a 0.125-base-voxel
  endpoint, compact-direction, fixed-sqrt `uint16` scenario. Fractional
  evaluation quanta use shared eager/cache rounding while canonical float
  anchors and the persistent compact storage schema remain unchanged.

# 2026-08-22: fixed nonlinear uint16 fiberlet cost view

- Added fixed global square-root encoding of cost per prediction voxel with a
  ceiling of 256. It reconstructs edge totals with the stored path
  length without adaptive chunk ranges, cache identity changes, or persisted
  geometry changes.
- On the full Paris4 radius-768 replay it matched the compact-float control's
  two failure arcs and reference-distance distribution while reusing the same
  unchanged geometry cache.

# 2026-08-22: globally ranked incremental exact fiberlet lookahead

- Replaced one-best-continuation-per-checkpoint-prefix exact replay with one
  multi-source frontier and globally ranked full-horizon beam population.
  Winning routes now retain their lookahead suffix across checkpoint advances;
  multiple winners may share a checkpoint prefix.
- Added a single raw-loss cutoff shared by all source beams, deterministic
  fixed-batch parallel expansion and state accounting, and selected-prefix-only
  reference evaluation.

# 2026-08-22: exact full-width fiberlet replay default

- Restored exact cost-bounded lookahead as the replay default after incremental
  prefix processing removed the actual long-route slowdown. Finite search
  widths remain available only as an explicit approximate experiment.
- On the full Paris4 radius-768 replay, exact search completed in 82.66 seconds
  with five fiberlet failures versus 58.34 seconds and seven failures for width
  128, using the same hot cache and unchanged one-million-state cap.

# 2026-08-22: explicit fiber replay phases

- Replaced the misleading weighted replay percentage with independent
  cache/preprocessing and actual reference-trace progress. Visualization and
  publication now report as a separate output phase.
- Compact output now removes completed cache progress, shows elapsed time once,
  adds a rolling current-speed ETA, and reports the latest bounded lookahead's
  state-expansion count plus its minimum applied local cutoff loss per
  prediction voxel.

# 2026-08-21: bounded intermediate fiberlet lookahead

- Added deterministic equal-distance intermediate pruning for graph replay,
  defaulting to a 128-route working frontier at 48-base-voxel intervals over
  the existing 192-base-voxel lookahead. Exact A* remains selectable with
  `--search-width 0` for focused comparisons.
- Preserved exact logical-front cost semantics: the terminal fiberlet remains
  whole in graph state while only its in-horizon edge fraction is scored, with
  its entering join charged once. Replay JSON now records search mode and
  per-front population/pruning diagnostics.

# 2026-08-20: concise fiberlet replay progress

- Replaced default per-stage, per-chunk, and per-evaluator replay chatter with
  one monotone overall reference-progress bar and ETA. Detailed profiling rows
  remain available through `fiberlet-replay --stats`.

# 2026-08-20: on-demand fiberlet storage and replay

- Replaced the unpublished cache payload with strict version 2. Float-cache
  anchors now retain canonical endpoint prediction/normal samples, and
  prefixes retain all five cost components, authoritative path length, and
  exact base-space endpoint steps. Cached joins no longer resample source
  volumes, and committed route reconstruction matches DP duplicate suppression.
- Canonicalized replay diagnostic indices across eager and cached graph
  adapters. On the 5,000-base-voxel Paris4 corridor, cold-cache, warm-cache,
  and eager replay produce byte-identical 342-point/65-edge fiberlet JSON and
  OBJ, total loss `91.7058129850775`, and loss density
  `0.14645713824432563`.
- Added strict version-1 float32 and compact fiberlet structure-of-arrays
  codecs with independently compressed fields, checksummed headers, stable
  anchor/edge identities, and separate connectivity and route payloads.
- Added local sparse Zarr-v2-layout anchor/fiberlet datasets with atomic chunk
  publication and on-demand float32 generation through the shared chunk cache.
- Extended `ChunkCache` with variable-length opaque payloads, independent
  scheduler lanes, lease-aware eviction, and byte-accurate per-cache/shared
  accounting.
- Replaced default corridor-wide replay graph materialization with stable-ID,
  cache-backed adjacency and route queries. The beam retains no permanent graph
  copy, and evicted connectivity/geometry reloads transparently. The old eager
  path remains available only through `--eager-graph` for diagnostics.
- Fixed a lost completion wakeup in parallel anchor fitting by publishing tile
  and ready-cell completion under the condition predicate mutex. No polling or
  timeout recovery was added.
- Added globally referenced replay progress, stable chunk schedule diagnostics,
  internal fiberlet phases, and independent greedy/fiberlet terminal rows.
- Parallelized canonical source-anchor candidate generation and released large
  prepared candidates on their search workers. On the measured Paris4 chunk
  `107,34,45`, generation wall fell from 34.795 to 4.344 seconds while prefix
  and route cache payloads remained byte-identical.
- Stored decoded anchor, prefix, and route objects directly in the existing
  lease-aware `ChunkCache` LRU. Prefix chunks build one charged incident index,
  lookahead uses exact endpoint descriptors, and route geometry is loaded only
  for committed edges. A cache-warm 100-base-voxel Paris4 replay fell from a
  17.13-second median to 0.20 seconds with byte-identical output.
# 2026-08-20: shared anchor observations and proposal evidence

- Built compact anchor observations and presence gradients once per bounded
  exact-union sample, replacing overlapping tile copies with uint32 index maps.
- Canonical median anchor wall and CPU improved by 11.7% and 11.4%, total wall
  by 6.7%, and peak RSS by 7.1%, with byte-identical replay artifacts.
- Added extraction-profile version 24 with shared-observation and tile-map work
  counters plus complete raw/compact coexistence memory accounting.
- Materialized eligible robust-proposal evidence once per cell and reused its
  contiguous 32-byte records across axis and final-membership passes. Median
  local-refinement work improved 2.5% and anchor CPU 1.3% with byte-identical
  replay artifacts; profile version 25 reports record preparation and storage.
- Restricted production fiberlet corridor admission to the two centerline
  segments incident to each curved-domain layer. Canonical segment tests fell
  72%, node-enumeration work about 6.6%, and fiberlet CPU about 1.5%, while the
  complete replay artifact remained byte-identical.
- Inserted common eight-corner interpolation cells into one resolved sparse
  bitmap page. Corner-collection work fell about 65%, preparation CPU about
  28%, and command wall about 4%, with unchanged sampled voxels and byte-
  identical replay output.

# 2026-08-19: float final anchor support reduction

- Isolated final refined-anchor evaluation in its own translation unit and kept
  both compact and expanded observation paths in checked float32 arithmetic.
- Reduced canonical final-evaluation worker time by 4.9%, anchor CPU by 1.4%,
  and total wall by 0.4% with byte-identical replay artifacts.
- Prepared candidate-side normal-aware smoothness once per outgoing fiberlet DP
  edge and reused it across incoming states. Canonical search wall and DP worker
  time improved about 2.0% and 2.1%, respectively, with byte-identical replay
  artifacts.
- Prepared current-side and candidate-side multiplicative alignment inputs at
  their natural DP reuse boundaries. Canonical search wall and DP worker time
  improved another 2.3% each with byte-identical replay artifacts.

# 2026-08-18: solve-local fiberlet DP reuse

- Cached normalized scoring inputs once per retained node and reused each
  reached node's outgoing descriptors across its incoming DP states.
- Replaced all-node cumulative state with rolling two-layer float32 costs and
  packed-key-derived predecessors, cutting canonical search CPU by 45.9% and
  total CPU by 8.8% while preserving selected geometry and replay failures.
- Added extraction-profile version 13 with solve-local memory, node/edge reuse,
  and node-preparation diagnostics.
- Deferred interior scoring interpolation until first search access and cached
  it per candidate. This reduced canonical fiberlet CPU by another 35.9%, total
  wall by 9.7%, and peak RSS by 14.2% with byte-identical replay artifacts.
- Added extraction-profile version 14 with endpoint, lazy request/miss/hit, and
  shared/local scoring-memory diagnostics.
- Reused an exact ordered support-span stencil for complete anchor cells,
  reducing canonical anchor CPU by 5.5% and total wall by 3.6% with
  byte-identical replay artifacts.
- Added extraction-profile version 15 with support-stencil and clipped-cell
  counts.
- Traversed the exact owned-cell cube directly during anchor initialization,
  reducing canonical fit-setup worker time from 11.00 to about 0.09 seconds and
  anchor CPU by 4.1% with byte-identical replay artifacts.
- Added extraction-profile version 16 with public discovery, direct owned, and
  avoided support-visit counts.

# 2026-08-17: robust sampled-direction anchor refinement

- Replaced joint angular line search with deterministic competitive robust
  sampled-direction aggregation and position-only bounded refinement.
- Removed pre-refinement close-direction merging, preserved robust membership
  through peak fitting, and added strict configuration/artifact diagnostics.
- Reduced the default alternating robust update budget to one measured pass;
  `--maximum-iterations` remains the documented quality/speed knob for
  difficult overlapping-fiber fits, and legacy numeric identity is
  intentionally not required for this fitter.
- Fused robust histogram/tensor work, reused tile gradients, paired spatial
  objectives, and reduced peak-response geometry to transverse coordinates.
- Increased anchor tiles from four to six cells per axis after measured tile
  sweeps; canonical anchor wall time fell from 11.59 to 10.34 seconds.
- Stored transient peak-search observations and evaluated transverse peak
  responses in float32, reducing canonical anchor wall time to 9.76 seconds.
- Paired overlapping anchor tiles into bounded sampling jobs and reused raw
  prediction samples across pair halos. This reduced canonical sampler
  submissions by 32.6% with unchanged extraction output.
- Replaced per-fiberlet packed-node-key hash maps with bounded direct indexes,
  reducing canonical replay wall time by 2.1% with unchanged path work and
  replay quality.
- Replaced repeated scoring-voxel hash lookups with sparse `16^3` pages and
  dense page-local indices, reducing interpolation materialization by 4.2%
  while preserving exact replay artifacts.
- Prepared compact float32 prediction/normal axes and symmetric tensors once
  per sampled voxel, reducing interpolation materialization by 26.1% while
  preserving fiberlet geometry and replay outcomes.
- Replaced fiberlet interpolation's iterative principal-axis solve with a
  guarded closed form, reducing resolver materialization by another 8.8% with
  zero fallbacks and byte-identical replay artifacts.

# 2026-08-17: fiberlet extraction profiling

- Added deterministic anchor and fiberlet workload counters and finer-grained
  wall, CPU, and summed-worker timings without changing extraction decisions.
- Added one shared versioned extraction-profile row to `vc_fiberlets benchmark`
  and full `fiberlet-replay` extraction.
- Documented a reproducible, exact-output performance protocol and recorded the
  initial optimization candidates for the fiberlet tracing pipeline.
- Added extraction-profile version 2 with exclusive anchor-fitting subphases,
  repeated observation-visit counters, and exact peak-cache/backtracking
  diagnostics. Measurement identified local component refinement, not
  seed-pair fitting, as the next dominant extraction hotspot.

# 2026-08-13: fiberlet graph replay

- Replaced the quantized world-axis half-grid DP with a deterministic curved
  cubic-Hermite domain using 2-voxel arclength planes, parallel-transported
  frames, 0.5-voxel transverse offsets, and floating-point interpolation.
- Removed the fiberlet lattice's 45-degree smoothness dead zone and scored
  graph joins with the same shared local alignment and Lasagna-normal
  tangent/normal objective, including exactly-once replay accounting.
- Expanded integer-DP candidate generation from the radius-four shell to every
  shorter cell offset within the same outer bound and added a strict 25-degree
  sampled-fiber-direction feasibility constraint.
- Added deterministic bidirectional fiberlet graphs with strict 45-degree
  anchor joins and loss-density beam routing with anchor-level lookahead.
- Added `fiberlet-replay`, shared variable-step reference matching, strict
  graph/route artifacts, and an independent reloadable napari route layer.
- Unified reference, greedy, and fiberlet longitudinal comparison scope under
  `--along`; greedy postroll and exact display cropping now derive from it.
- Fiberlet graph replay now completes the edge containing its first distance
  failure and continues for anchor-bounded `--along` postroll, with explicit
  complete/truncated distances, overshoot/shortfall, and whole-edge identities.

# 2026-08-12: narrower independent anchor NMS

- Decoupled transverse anchor NMS from the peak-refinement window and reduced
  defaults to 2 transverse and 1 longitudinal prediction voxels.
- Added the transverse radius to strict experimental anchor artifacts and path
  loading while retaining conservative external crop context.

# 2026-08-12: fiberlet replay distance filtering

- Added an independent napari fiberlet-radius control that physically removes
  paths wholly outside the exact reference/failed-trace polyline distance.
- Set display defaults to 32 base voxels for presence and anchors and 16 for
  fiberlets, with all three values preserved across artifact reload.

# 2026-08-12: parameter-independent anchor visualization

- Decoupled napari anchor-stage visualization compatibility from extractor
  parameters. Old, absent, or extended parameter metadata now renders without
  affecting coordinate, geometry, lineage, or final-output consistency checks.

# 2026-08-12: full 2D anchor subpixel experiment

- Replaced independent 1D peak parabolas with a complete 3x3 least-squares 2D
  quadratic including cross-coupled curvature and conservative Hessian,
  half-step, owner-domain, and real-response guards.
- On the fixed David Paris4 benchmark, the 2D fit reproduced the discrete
  population exactly but scored below the 1D baseline: 29.07%/43.90% versus
  29.78%/44.88% anchor/cell hits at 4 vx, and 54.09%/80.42% versus
  54.59%/81.06% at 8 vx.
- Extending the signed-gradient sweep above its former maximum found a
  4-voxel optimum near weight `1.0` and an 8-voxel optimum near `1.3-1.5`;
  weight `1.2` gave the best equal-weight aggregate for joint-2D.
- Added matched `discrete`, `separable_1d`, and `joint_2d` benchmark positions
  from one extraction. The old `0.2` 1D result reproduced exactly. Across the
  complete `0.0-2.0` sweep, separable-1D beat joint-2D on every 4/8 anchor/cell
  hit rate; separable weight `1.1` scored 31.14%/46.89% at 4 vx and
  55.97%/82.64% at 8 vx.
- Selected separable-1D for production anchor placement and gradient weight
  `1.0` as the default. Joint-2D remains transient benchmark provenance only.

# 2026-08-12: discrete-versus-subpixel anchor benchmark

- Split refined-anchor localization output into equal-population `discrete`
  and `subpixel` reports while keeping discrete provenance transient and strict
  version-1 artifacts unchanged.
- Selected `0.2` as the default signed gradient weight from the fixed David
  Paris4 sweep. Explicit CLI and artifact values remain authoritative.
- At `0.2`, subpixel fitting improved 4-base-voxel anchor/cell hit rates from
  27.79%/41.94% to 29.78%/44.88% and 8-base-voxel rates from 53.76%/79.86% to
  54.59%/81.06% relative to the discrete peaks.

# 2026-08-12: signed presence-gradient anchor centering

- Added deterministic presence-only 3D Sobel gradients and reliability-gated
  inward/outward normal-plane voting to the refined-anchor peak objective.
- Added strict gradient weight/reliability artifact parameters and
  `--gradient-weight`, with exact weight-zero fallback.
- On the supplied David Paris4 reference, the default raised 4-base-voxel
  anchor/cell hits from 1,002/982 to 1,021/1,000 and 8-base-voxel hits from
  1,884/1,823 to 1,896/1,832 while reducing mean, median, and p95 distance.

# 2026-08-12: outer-parallel anchor extraction

- Parallelized anchor sampling and fitting over canonical cell jobs while
  forcing every job's lower-level prediction sampler to one thread.
- Added aggregate concurrent-halo memory bounding, deterministic indexed result
  and error assembly, and serialized progress/retain handling.

# 2026-08-12: refined-anchor localization benchmark

- Added `vc_fiberlets anchor-benchmark` to extract only geometric refined
  anchors in exact reference-intersecting cells and measure their exact
  base-coordinate distance to a strict fiber's dense line.
- Added stable distribution output and inclusive 4/8-base-voxel anchor and
  reference-cell hit rates, with empty refined populations kept explicit.
- Shared exact polyline-to-cell selection with failed-trace replay while
  preserving replay's positive-radius tube behavior.

# 2026-08-12: physical replay anchor filtering

- Changed the Napari replay anchor-radius cutoff to physically subset final and
  staged anchors, cell centers, and refinement offsets instead of leaving
  transparent depth-occluding geometry in the rendered layers.
- Retained defensive full geometry/features for reversible slider updates and
  transactional artifact reload/rollback.

# 2026-08-12: local-maximum fiberlet anchors

- Added deterministic direction-conditioned normal-plane peak search after the
  existing non-orthogonal anchor direction fit.
- Integrated the peak response with a `1.5`-voxel transverse Gaussian and a
  longer `1.5`-cell-side along-direction Gaussian, with complete rotated-kernel
  halo sizing.
- Added bounded non-decreasing subvoxel peak fitting, continuous owner-cell
  constraints, normalized edge response, strict transverse/axial peak fields,
  and base-voxel CLI controls.

# 2026-08-12: faster replay viewer startup

- Kept detailed anchor-stage layers enabled by default so replay diagnostics are
  immediately available; `--no-anchor-stages` provides an explicit fast-start
  opt-out and avoids hashing unused stage artifacts.
- Preserved strict stage schemas and selected-stage topology across artifact
  reloads.

# 2026-08-11: dense-fiber failure replay

- Added C++ greedy native replay against dense VC3D fiber geometry with bounded
  monotone matching, typed failure statuses, and exact postroll handling.
- Added exact tube-selected anchors, tube-constrained integer fiberlets, sparse
  replay scoring preload, and content-addressed atomic diagnostic bundles.
- Added strict napari replay loading with external-Zarr validation and separate
  reference, trace, failure, anchor, and quality-colored fiberlet layers.
- Kept replay failure-marker styling compatible across napari Points API
  versions by avoiding the renamed outline-color keyword.
- Added a replay presence-tube mask using a one-time reference rasterization and
  base-voxel distance transform, with a runtime radius slider defaulted from the
  anchor/fiberlet extraction tube.
- Added replay stage timing plus per-second anchor and fiberlet progress/ETA;
  anchor progress distinguishes selected-cell fitting from NMS context.
- Expanded the replay presence mask to the union of the reference fiber and the
  complete failed greedy trace, including postroll, for direct comparison.
- Added `anchor_cells.obj` with every selected cell center and retained-anchor
  displacement lines, exposed as separate clipped napari diagnostic layers.
- Added strict initialized/refined/support/selection/NMS anchor-stage JSON
  diagnostics with stable merge lineage, exact filter causes, and actual NMS
  suppressors, plus independently toggleable napari comparison layers.
- Added an independent base-voxel anchor-radius slider that filters every replay
  anchor diagnostic against the exact reference/failed-trace union distance.
- Added transactional in-process replay-artifact reload with stable empty
  layers, strict prediction/crop compatibility, derived-distance refresh, and
  no presence-Zarr reopening.

# 2026-08-11: fiber-centered anchor refinement

- Replaced fixed cell-centroid anchor positions with bounded halo-backed joint
  direction/position refinement on a rotating plane through each cell pivot.
- Added deterministic direction-compatible local-maximum NMS across cells,
  exact crop-boundary context, strict refinement metadata, and focused
  convergence, locality, determinism, and artifact validation.

# 2026-08-11: fiberlet loss/quality visualization

- Added length-normalized per-fiberlet trace loss, report-relative visual
  quality, strict napari metric parsing, per-shape features, a runtime napari
  colormap selector, and density CLI statistics. Path MTL/material output was
  removed in favor of napari-owned display color.

# 2026-08-10: integer-DP fiberlet paths

- Added strict anchor-artifact loading, radius-four cell-shell pairing, and
  deterministic integer prediction-voxel DP with exact virtual endpoints.
- Added the native tracer's shared multiplicative local alignment loss with
  lattice-edge integration, finite invalid-prediction bridges, and shared
  native Lasagna-aware direct curvature.
- Added `vc_fiberlets paths`, versioned JSON/OBJ output, focused regression
  tests, and real-crop deterministic validation. Global graph construction,
  deduplication, extension, and H/V/winding assignment remain deferred.
- Made base-volume coordinates the only spatial contract exposed by the
  unshipped fiberlet CLI and JSON/OBJ artifacts; no compatibility aliases or
  legacy coordinate fields are retained.
- Added `paths --stats`, explicit score-presence semantics, and two-index OBJ
  path edges.
- Added independently loadable central XY/XZ/YZ fiber-presence textured-quad
  OBJ/MTL/PNG context, with direct presence sampling and `--no-slices` opt-out.
- Materialized each small crop's fiber/normal scoring volume once and moved
  exact independent DP searches to a deterministic fixed worker pool, removing
  repeated sampling and nested thread teams.
- Added monotonic rate-limited path progress with throughput and ETA on stderr.

# 2026-08-10: C++ fiberlet cell anchors

- Added deterministic cell-based extraction of zero, one, or two
  non-orthogonal unoriented anchors from canonical Fiber Lasagna
  `presence/nx/ny`, plus the cache-aware `vc_fiberlets anchors` command and
  sparse JSON/base-coordinate OBJ artifacts. Connection and path stages remain
  deferred.
- Merged near-duplicate fitted directions using an angle-plus-objective test
  and a joint PCA refit, with strict diagnostics and configurable thresholds.
- Kept the complete anchor OBJ and added separate deterministic component-zero
  and component-one OBJ layers for inspection.

# 2026-08-07

- Simplified manager S3 staging to marker-only resumable rclone transfer and
  restored lean existing-schema Atlas Lasagna entries without duplicated
  portable provenance.

# 2026-08-06

- Added path-aware per-user Bash completion installation for `las_manager`,
  with isolated providers that coexist across multiple virtual environments.
- Added longest-prefix contextual help and shared argument-aware Bash/Zsh
  completion, including exact cache-local OME scale proposals.
- Made manager catalog indexing tolerate volumes whose optional shape is null.
- Replaced repeated `volume ls` labels with a grouped table and added exact
  chunk-backed prefetched-scale reporting.
- Refined the volume tree so each scroll shares its first volume row and
  branches additional volumes below, removed the duplicate ID column, and
  aligned depth/height/width components to widths 6/5/5.

# 2026-08-03

- Added experimental float16 shared raw-product accumulator rings while
  retaining the measured-faster float32 default, float32 weights, and float32
  flush arithmetic.
- Moved shared inference integer-to-FP32 normalization onto CUDA after compact
  H2D and made Fiber model autocast default to checkpoint training policy.
- Added opt-in shared Fiber/Lasagna multi-device loader and worker stage
  profiling with direct reader-concurrency diagnostics.
- Overlapped shared Lasagna/Fiber output flushing with inference using one
  enlarged bounded circular mmap and one runner-wide asynchronous flush,
  without band-sized RAM snapshots or overlap copies.
- Made downloader negative-remote caches tolerant and atomic, and exposed the
  automatic S3 transfer worker count in Fiber and Lasagna inference CLIs.
- Added shared bounded streaming multi-GPU whole-volume inference for Lasagna
  and Fiber 3D, with separate CPU/Zarr prefetch, persistent per-device workers,
  shared-memory input/results, and canonical single-writer accumulation.
- Made Fiber 3D per-rank hang logs, manual dumps, test watchdog, resource
  polling, and diagnostic CUDA synchronization explicitly opt-in through config
  or CLI.

# 2026-08-02

- Fixed cropped shared 3D inference treating globally positioned output chunks
  as unsupported when their origins exceeded the crop-local accumulator shape.
- Fixed circular accumulator depth underplanning across initial chunk-aligned
  no-op flushes in cropped, downscaled shared inference.
- Distributed dense Fiber 3D tests deterministically across DDP ranks with
  persistent process-worker prefetch, exact ordered metric reconstruction, and
  stdout/TensorBoard total-test timing.

# 2026-08-01

- Added persistent per-rank Fiber 3D training diagnostics and an eight-minute
  rank-0 test watchdog with per-batch phase/resource markers and manual
  `SIGUSR2` stack dumps.
- Fixed Vesuvius Python CI collection by exposing the monorepo's shared
  `lasagna` source namespace and triggering the workflow for Lasagna changes;
  repaired the Zarr 3.2.1 matrix with explicit cross-version v2 fixtures and
  shared Zarr 2/3 chunk-key and raw-store access.

# 2026-07-31

- Repaired Atlas pred-snap tests for strict per-channel 3D Lasagna arrays,
  made Fiber 3D inference include ceil-sized odd OME edge planes, and made VC3D
  reject incompatible independently attached Lasagna source volumes.
- Made native mode effective during new-fiber seed creation: configured fiber
  inference now chains the internal Lasagna reference solve directly into the
  existing single-CP native extrapolator without displaying or saving the
  intermediate Lasagna geometry.
- Made every v3 fiber reader reject missing or malformed mode/segment metadata
  without repair, completed v3 geometry consumption in Atlas, Lasagna, and
  Spiral, made sync route invalid v3 inputs to manual conflict handling, fixed
  NML/direct Python construction, and made Lasagna probe optimization outputs
  carry coherent regenerated v3 span metadata.
- Merged the main VC3D line-annotation toolbar, tag, schematic-overview,
  in-place refresh, and pane-lifecycle work while retaining both interactive
  rendered fiber strips and their per-span mode/metric/message labels.
- Removed the unpublished top-level `vc3d_fiber` version 2 and its obsolete
  descriptor migrations from VC3D, shared C++ readers, Python training input,
  and sync validation. Legacy file version 1 and current version 3 remain;
  version 3's `tracer_version: 2` is unchanged.
- Made fiber-aware sync merge version-3 dense span geometry and CP-owned
  descriptors atomically, preserve separated local/remote edits across an
  unchanged span, merge the global mode base-aware, and route every adjacent,
  overlapping, or unalignable result to the existing manual conflict workflow.
- Persisted the actual Lasagna and fiber-inference manifest identities consulted
  per interpolation span, using public manifest URLs instead of local cache
  paths for open-data catalogue Lasagna.
- Changed newly created VC3D fibers to default their global interpolation and
  extrapolation mode to the native fiber tracer while retaining Lasagna for
  older files with no persisted mode.
- Added `vc3d_fiber` v3 per-span interpolation goals and actual modes, grouped
  cubic-spline interpolation, trace-to-Lasagna-to-spline fallback, the
  global-only 100-base-voxel shortcut, persisted metrics/messages, a checked
  Ctrl-right-click goal selector, and viewport-packed `C`/`L`/`T` span labels.
- Restricted persisted meeting diagnostics to accepted native spans; fallback
  records now retain only failure code/detail, and all readers ignore stale
  fallback meeting values so earlier project files load cleanly.
- Made native VC3D fiber extrapolation completion depend only on its nominal
  `ceil(distance / step)` generation budget, without target planes,
  `max_step_factor`, or measured arc-length acceptance.
- Added a 10-base-voxel floor to native CP-pair meeting acceptance, which now
  uses `max(10 base voxels, 10% of combined partial traced length)`.
- Added VC3D terminal warnings whenever native line-annotation tail
  extrapolation retains its Lasagna fallback, including the side, full
  trace/exception reason, returned point count, and failure source.

# 2026-07-30

- Preserved native span error labels through Reoptimize/branch refresh and made
  native extrapolation stop at the last valid prediction before a volume edge
  instead of restoring the Lasagna tail.
- Replaced VC3D's endpoint-only native CP-pair acceptance with symmetric
  moving-plane trace intersection, 10%-of-traced-length acceptance, and
  arc-length-warped fusion; persisted accepted/fallback outcomes now restore
  meeting errors or stable failure labels after reload.
- Hard-constrained every Lasagna fallback span and retained tail adjacent to
  native fiber geometry to continue the fitted dense native endpoint tangent,
  using fixed proxy points in the regular Ceres solve, independent of normals,
  candidate selection, seed choice, or solve order.
- Removed previous Lasagna geometry and endpoint directions from full
  reinitialization candidates, made solved-neighbor directions exclusive
  rollout sources, and made degenerate normal-plane transport choose a
  perpendicular tangent instead of preserving a normal-parallel direction.
- Fixed Fiber-model mode and extrapolation-distance changes to rebuild both
  tails on newly seeded one-control-point fibers when Auto-reoptimize is active.
- Added persisted fiber-global Lasagna/native modes to VC3D line annotation,
  including full rebuilds on mode changes, invalid-span native retracing with
  per-span Lasagna fallback, trained-neighbor continuation directions, and
  configurable native/Lasagna tail extrapolation.
- Ported Python's target-local multi-plane intersection behavior to the shared
  C++ tracer used by VC3D and `vc_fiber_trace_metric`, including per-beam
  recrossing state, all-plane termination, threshold-aware whole-fiber
  continuation, and selected-crossing diagnostics.
- Further reduced the approved native precomputed whole-fiber workload from
  1.869s wall / 8.222s CPU to a three-run median of 0.986s / 5.134s with
  deterministic partial parent ordering, compact lazy frontiers, unique-cube
  corner reuse, unit-vector invariant math, unique-key start sampling, indexed
  worker batches, and compact candidate task storage. All final runs retained
  7 restarts and the exact 6,910,839-candidate / 4,318-generation workload.
- Reduced the approved native precomputed whole-fiber workload from 21.155s
  to 1.869s with exact lower-bound lookahead ordering, a measured default
  32-parent final-lookahead cap, and fused pinned-corner decode/scoring. The
  retained result has 7 restarts versus the 8-restart baseline; exact lazy and
  exhaustive controls remain available.
- Added shared requested-level VC3D eight-corner batch sampling for native
  fiber prediction and Lasagna normal volumes, including mixed physical chunk
  grids, one decoded cache per physical scalar volume, boundary-aware retained
  chunk lookup, and caller-side orientation-tensor normal interpolation.
- Fused persisted prediction/normal corner decoding into native candidate
  scoring, eliminating full decoded candidate arrays and reducing the approved
  warm-cache whole-fiber workload from 37.277s to 27.291s wall time. The
  measured 8-restart post-float result is unchanged.
- Reduced that workload to 21.155s wall / 619.366s CPU with compact
  reconstruct-on-selection frontier records and corner-batch improvements.
  The accepted post-float result remained at 8 restarts.
- Converted native fiber tracing's internal geometry and candidate math to
  float while retaining double persisted/public coordinate boundaries.
- Added persistent CP-owned native fiber segment metadata in strict
  `vc3d_fiber` v2 files, Lasagna protection for traced spans, scoped CP-edit
  invalidation, finalized trace auto-save, and Ctrl-right-click transactional
  reversion of traced spans to Lasagna optimization.
- Standardized Python CLI, native C++ CLI, and VC3D fiber endpoint acceptance
  on a `20` base-voxel threshold; physical voxel size is now optional and used
  only for physical-unit reporting.
- Replaced opaque Lasagna project/cache identities with actual local or remote
  Zarr source paths, readable remote-source cache paths, and path-bearing
  manifest ownership tags without duplicating manifest-derived group, channel,
  or spacing metadata.
- Corrected VC3D native segment tracing to keep stored fibers in base
  coordinates while prediction and normal sampling run on the derived sd2
  trace grid with correctly scaled physical units.
- Added generic exact-byte arbitrary remote-file caching, persistent direct
  remote Lasagna manifests, and VC3D local/remote manifest attachment with
  canonical role tags and automatically reconciled ordinary 3D project
  volumes.
- Restricted VC3D Lasagna attachment and sampling to per-channel 3D ZYX arrays;
  older flat CZYX preprocessing/fit artifacts now require conversion to
  per-channel OME-Zarr.
- Reduced native `vc_fiber_trace_metric` warm-cache runtime on the remote
  fiber manifest workload from roughly 314s to roughly 86s with profiled
  direct chunk-resolution sampling, bounded deterministic beam pruning, and
  lightweight final-lookahead frontier records while preserving the measured
  5-restart result.
- Added opt-in native C++ Trace2CP parallel candidate scoring through batched
  persisted prediction materialization, batched Lasagna normals, and
  `vc_fiber_trace_metric --threads`, while preserving deterministic beam output
  order.
- Reduced native Trace2CP beam expansion overhead by keeping internal trace
  paths parent-linked and caching per-trace cone offsets instead of rebuilding
  them per beam.
- Matched native C++ Trace2CP beam pruning/reached-target selection and compact
  normal principal-axis decoding more closely to the Python tracer, with
  focused `test_fiber_trace3d` regression coverage for beam order, reached
  ties, normal eigensolver behavior, and all-pairs candidate loss.

# 2026-07-29

- Native 3D Trace2CP refined/fused/regenerated presence panels now display
  presence modulated by predicted direction alignment to the strip plane, while
  original presence panels remain raw presence.
- Python native 3D Trace2CP whole-fiber tracing now keeps tracing after early
  far target-plane crossings until the best selected crossing is within the
  configured error threshold or the trace budget is exhausted.
- Python native 3D Trace2CP whole-fiber tracing now preserves live trace point,
  previous direction, sampled-current direction, and smoothing history across
  successful CP crossings instead of reinitializing at each accepted CP.
- Native 3D Trace2CP target-plane termination no longer uses the CP-to-CP
  chord. Python and VC3D native tracers now require target-local line-neighbor
  and inferred-direction planes, wait until all configured planes are crossed,
  and score the lowest in-plane CP crossing error.
- Added native `vc_fiber_trace_metric --inference-scaledown-power`, defaulting
  to 2, so existing fiber prediction manifests derive trace scale from the
  persisted prediction scale without adding manifest fields.
- Brought the native C++ `vc_fiber_tracer`/`vc_fiber_trace_metric` Trace2CP
  search controls into parity with the Python native tracer for persisted
  inference products: circular default cone candidates, presence-weighted
  current branch choice, angle-squared split smoothness, cumulative tangent
  smoothness, target-plane crossing interpolation, spatial beam pruning, and
  matching CLI flags/defaults.
- Python native 3D Trace2CP now accepts multiple `--fiber-json` paths, runs
  them sequentially with a shared loaded model, writes indexed per-fiber
  summaries/visualizations, and reports an accumulated restart-rate score.
- Python native 3D whole-fiber Trace2CP visualization now page-splits wide
  JPG output around restart boundaries and before the JPEG dimension limit.
- Python native 3D whole-fiber Trace2CP CP labels now render at the bottom of
  each strip and include CP indices for explicit trace selection.
- Python native 3D whole-fiber Trace2CP can now start metric tracing at a
  selected CP with `--whole-fiber-start-cp-index`.
- Improved remote Lasagna manifest fetch errors with resolved URL, HTTP
  response metadata/body excerpts, and S3 region/auth diagnostics.
- Required Lasagna normal samplers for native Trace2CP normal-aware smoothing
  and made `vc_fiber_trace_metric` require explicit `--normal-manifest`.
- Changed Python native Trace2CP and the native C++ fiber metric default beam
  lookahead to 2 so their relevant trace-control defaults match.
- Changed native `vc_fiber_trace_metric` to infer its tracer working scale from
  the precomputed fiber inference manifest channels instead of a CLI scale
  argument.
- Extended native VC3D Lasagna dataset opening so `vc_fiber_trace_metric` can
  stream precomputed fiber inference manifests directly from HTTP/S3 locations
  with an explicit remote cache directory, including relative and absolute
  group Zarr paths.
- Prevented pyramid multiprocessing from multiplying every worker by a full
  OpenBLAS/OpenMP thread pool while retaining automatic process parallelism.
- Added the first native VC3D 3D fiber Trace2CP segment tracer: shared
  Lasagna compact-channel helpers, project-level fiber inference dataset
  selection, a Qt-free `vc_fiber_tracer` core target, and a Ctrl-right-click
  generated-line segment optimization action in the line annotation GUI.
- Added native `vc_fiber_trace_metric`, a no-visualization C++ whole-fiber
  restart-rate runner for precomputed 3D fiber inference `.lasagna.json`
  products and `vc3d_fiber` JSON files.

# 2026-08-06: Lasagna inference manager foundations

- Added the installed `las_manager` command with XDG configuration, atomic
  initialization, unique command prefixes, and read-only shell completion.
- Added conditional one-hour open-data catalog caching and deterministic volume
  discovery that preserves full Atlas identity/origin/license metadata.
- Added safe cached Fiber checkpoint discovery with stable backend/run/snapshot
  selectors, checkpoint hashes, test metrics, model/output/precision metadata,
  and optional Atlas model identity.

# 2026-07-28

- Defaulted Fiber whole-volume output to filtered 0.25x inference, changed
  Fiber and Lasagna OME chunks to 64 cubed, and made the shared circular
  runner touch, flush, and clear only supported contribution-dirty chunks.
- Native 3D Trace2CP inference blocks now use a shared VC3D-backed
  requested-level axis-aligned block-read API and batch missing block forwards,
  avoiding dense coordinate-grid sampling for regular inference cubes.
- Native 3D Trace2CP live inference now uses the shared fiber 3D inference
  adapter plus shared Lasagna normal encoding and raw fiber prediction decode
  helpers.
- Native 3D Trace2CP now defaults to metric-only output; `--vis` explicitly
  enables JPG rendering and partial image updates.
- Native 3D whole-fiber Trace2CP now reports restarts per 1000 reference
  voxels, with optional restarts per meter in summaries and progress output
  when VC3D `volume.metadata["voxelsize"]` is available.
- Native 3D whole-fiber Trace2CP human metric output now uses one decimal
  place and includes mean successful run length in millimeters beside `err/m`
  when physical units are available.
- Native 3D Trace2CP can now box-downsample raw inferred fields with
  `--inference-scaledown-power` before direction/presence sampling.
- VC3D remote volume metadata normalization now always discovers public
  `scan/tomo/acquisition/detector/samplePixelSize` as `voxelsize` when no
  explicit positive `voxelsize` exists.
- Replaced full-Z predict3d scratch mappings with fixed-depth circular mmap
  rings and consolidated Lasagna/Fiber neural inference onto one multi-scale,
  chunk-flushing runner.
# 2026-08-03: process-parallel shared inference flush

- Replaced the single Python flush thread with bounded persistent spawn workers
  that read frozen rolling-accumulator mmaps by absolute path.
- Added shared `flush_workers` control and `--flush-workers` to Fiber and
  Lasagna inference (automatic CPU-count default capped at 64, synchronous
  baseline 0).
- Added process failure/hard-exit cleanup, overlap, multi-process execution,
  and CLI forwarding coverage.
- Motivation: the prior threaded implementation regressed the representative
  eight-GPU inference phase from 178.8 s to 305.4 s.
- Reused the pyramid pool's pre-spawn native-runtime guard so NumPy/OpenBLAS is
  single-threaded during child module import, including for GPU workers.
# 2026-08-03: TensorStore whole-volume inference prefetch

- Shared Fiber/Lasagna inference now defaults to asynchronous TensorStore Zarr
  bbox reads with read-ahead capacity independent of GPU/result slots.
- Added bounded cache/I/O/copy and per-GPU prefetch controls, single-device
  read-ahead, Python-Zarr fallback, exact reader-equivalence tests, and input
  starvation/high-water diagnostics.

# 2026-08-03: process-parallel native accumulation

- Added deterministic process-owned chunk accumulation shared by Fiber and
  Lasagna multi-device inference, with bounded queues and retained result-slot
  lifetimes.
- Added a portable native accumulator extension with runtime AVX-512F+F16C
  dispatch and restored float16 product rings as the default.
- Added `--accumulator-workers`, backend/throughput diagnostics, native
  numerical coverage, and process-vs-synchronous output coverage.
# 2026-08-13: half-voxel fiberlet DP

- Halved anchor-to-anchor fiberlet DP spacing to 0.5 stored prediction voxels,
  with cached sign-invariant prediction/normal interpolation and unchanged
  base-coordinate artifacts.

# 2026-08-14: full-reference dual fiber replay

- Added independent reset-capable greedy and fiberlet whole-reference replay,
  strict version-2 segmented artifacts, optional indexed failure
  visualizations, and strict napari selection/reload.
- Replaced whole-graph scoring preloads with deterministic on-demand curve
  batches through the existing volume samplers and added a reproducible local
  anchor/fiberlet extraction benchmark with sampling, DP, and peak-memory
  diagnostics.
- Replaced curve-batched fiberlet preparation with a global stage-parallel
  pipeline: each candidate geometry is built once, native corners are globally
  deduplicated independent of sampler batch size, and materialized nodes are
  reused by parallel DP. Added actual per-stage process-CPU utilization and
  coordinate-call diagnostics.
- Compacted retained fiberlet DP nodes to 24 bytes with checked 32-bit local
  keys and Lasagna-style axis/presence bytes, removed retained interpolation
  stencils, and separated prepared-geometry from transient-search memory
  reporting.
- Replaced overlapping per-cell anchor reads and the second NMS-context pass
  with deterministic shared dense tiles over a conservative context population;
  on the fixed 4096-base-voxel Paris4 workload this reduced anchor time by 42%,
  total time by 22%, and peak RSS by about 75%.
- Defaulted `vc_fiberlets benchmark` to the complete reference interval after
  the first control point; explicit `--along` still requests a shorter interval.
- Added replay-only `--length` in base voxels so extraction, both evaluators,
  failure reporting, persisted geometry, and visualizations share one bounded
  interval; omission retains complete-reference replay.
- Restored direct replay visualization manifests without a viewer index,
  published stable per-tracer failure aliases for reload, and restored loading
  of strict version-1 single-visualization replay artifacts.

# 2026-08-17: CT-rendered replay trace strips

- Added three disconnected, sheet-aligned replay strip OBJ artifacts per
  failure using the unchanged default `buildLineViewSurfaces()` geometry.
- Extracted the existing `vc_lasagna_line_probe` fine-to-coarse surface texture,
  textured-mesh, and TIFF helpers for shared use without changing their sampler
  behavior.
- Made `vc_fiberlets fiberlet-replay --vis` require `--volume` to name one
  concrete CT OME-Zarr array/group. Base trace coordinates are mapped through
  the parent `multiscales` transform before the unchanged renderer samples that
  group. Each trace publishes a hashed OBJ/MTL/uncompressed-TIFF triple; the
  selected group transform is persisted and napari reads only those artifacts.
- Removed the unpublished replay-only ribbon, mask, uint16, and PNG paths.
- Accepted equivalent concrete-group paths with or without a trailing directory
  separator when resolving OME-Zarr multiscales metadata.
- Replaced fixed replay strip supersampling with automatic endpoint-inclusive
  source-group voxel resolution. Napari now tessellates the validated textured
  OBJ to every stored TIFF texel instead of displaying only coarse OBJ-vertex
  samples.
- Added `fiber_replay.jpg`, a full-selected-interval top/side CT overview with
  yellow reference, red greedy, and cyan fiberlet overlays. It uses the same
  default line-view surfaces and shared renderer, honors `--length`, is emitted
  even for zero-failure visual runs, and is stored both immutably and at a
  stable direct-inspection path.
- Increased only the direct-inspection replay overview to 8x source-group
  sampling, added full-height pre-reset failure bands, and wrapped long strips
  into deterministic fraction-aligned panels without resampling. Per-failure
  OBJ/MTL/TIFF strips remain at native selected-group sampling.
- Replaced replay's isotropic failure cutoff with one shared Lasagna-normal
  ellipsoid for greedy samples, fiberlet samples, and fiberlet reseeding: the
  configured base-voxel radius remains unchanged normal to the sheet and is 4x
  in the tangent plane. Replay output now stores explicit component diagnostics
  and rejects inconsistent measurements before publication.
- Extended the full replay overview with fiberlet-centered top/side CT strips,
  preserved reset components with explicit raster placement, and changed long
  output to complete four-strip blocks packed across indexed JPEGs capped at
  65,000 pixels per dimension.

# 2026-08-17: profiled and accelerated fiberlet extraction

- Added versioned anchor/fiberlet phase profiles to benchmark and replay output,
  including deterministic work counters and process-CPU measurements.
- Replaced repeated replay-tube scans with an immutable float32 Boost segment
  R-tree and prioritized each local corridor's layer-adjacent segment. On the
  5,000-base-voxel Paris4 replay workload, median wall time fell from 58.61 to
  25.32 seconds while complete published artifacts remained byte-identical.
- Added a conservative exact support-sphere broad phase to repeated anchor
  refined-state evaluation. On the same 5,000-base-voxel replay, median wall
  time fell from 24.92 to 22.09 seconds and complete artifacts remained
  byte-identical.
- Added version-3 local-refinement subphase profiling, identifying repeated
  refined-state evaluation as roughly 89-90% of remaining local-refinement
  worker time on the canonical replay.

# 2026-08-18: contiguous anchor peak-grid caching

- Replaced direction-conditioned peak-search ordered maps and repeated grid
  geometry with checked contiguous row-major caches. The canonical replay's
  median peak-search worker time improved from roughly 43.9 to 42.84 seconds
  with byte-identical artifacts and deterministic work counters.
- Split peak responses into a 16-byte hot stream and sparse retained-evidence
  storage. Canonical median peak-search worker time improved from 42.84 to
  39.94 seconds and command wall from 10.43 to 10.25 seconds with unchanged
  replay artifacts and quality populations.
- Kept production compact robust direction proposals in float32 through their
  per-observation loop, widening only fixed-size summaries for the existing
  double cutoff and eigensolver. Canonical median proposal work improved from
  25.63 to 23.74 worker-seconds with unchanged populations/failures and a
  maximum emitted-route displacement of 1.38e-6 base voxels.
- Isolated retained spatial objectives in a private translation unit and kept
  production compact objective arithmetic float32 while preserving the public
  compensated-double path. Canonical median objective work improved from 22.59
  to 13.86 worker-seconds and command wall from 9.56 to 9.17 seconds with
  byte-identical replay artifacts.

# 2026-08-19: cooperative ready-cell anchor scheduling

- Work-balanced prepared anchor cells across the bounded sampling-group worker
  pool while retaining tile-local observation ownership and overlap reuse.
  Canonical median anchor wall time improved from 5.067 to 4.262 seconds and
  total wall time from 7.77 to 6.97 seconds with exact replay artifacts.
- Replaced pair-local raw prediction reuse with bounded exact-union sampling
  partitions. The canonical replay submitted 77% fewer prediction voxels;
  median anchor CPU improved 12.1% and command wall 2.2% with exact artifacts.
- Shared one source-private inline implementation between exported local
  scoring and the prepared fiberlet DP loop. Canonical median DP search wall
  improved 9.8% and fiberlet wall 5.7% with exact replay artifacts and work
  counters.
- Deferred isotropic-angle evaluation in shared local scoring until an invalid
  normal or degenerate projected tangent requires it. Canonical median DP
  search wall improved another 6.2% with byte-identical replay artifacts.
- Batched the four pair-dependent interior-DP alignment dots across compact
  valid outgoing lanes while retaining scalar transition relaxation order.
  Matching optimized runs improved median DP search wall by 5.9% and fiberlet
  wall by 5.7% with byte-identical replay artifacts.
- Pre-indexed immutable robust-proposal eligibility during compact anchor
  fitting. Matching optimized runs reduced median proposal work by 20.4%,
  anchor CPU by 2.2%, and command wall by 0.9% with byte-identical replay
  artifacts.
- Removed redundant robust-membership state copies while retaining the original
  proposal kernel. Matching optimized runs improved median fitting work 1.6%
  and command wall 1.1% with byte-identical replay artifacts.
- Moved direction-conditioned peak signal from the dense response stream into
  sparse evidence, reducing hot records from 16 to 12 bytes. Matching optimized
  runs improved median peak-search work 5.0% and anchor CPU 1.5% with byte-
  identical replay artifacts.

# 2026-08-19: fiberlet storage quantization experiment

- Added a one-extraction C++ benchmark for the proposed anchor-position,
  compact-axis, and per-chunk cost encodings. Endpoint quantization now feeds
  the regular curved-plane DP rather than attempting to reuse a baseline route.
- On the canonical Paris4 interval, no quantization added a tracing failure.
  Maximum Euclidean line separation was 10.15/9.07/11.46 base voxels for
  position quanta `1/2/4`, 3.56 for compact axes, 5.35 for `uint8` costs, and
  zero for `uint16` costs; no persistent encoding was selected.

# 2026-08-20: hoisted compact robust-proposal geometry

- Materialized eligible compact proposal observations once per cell and reused
  their absolute positions, normalized directions, presence, and logical
  destinations while retaining the original floating-point geometry order.
- Published selected anchor cells ahead of context-only cells in each prepared
  tile's cooperative queue, modestly reducing the measured anchor wall tail
  while preserving byte-identical output.
- Enforced finite positions once while preparing compact robust-proposal
  records, eliminating redundant finite checks in three hot passes and reducing
  proposal worker time by about 9% with byte-identical replay output.
- Reused each final-evaluation component Gaussian for its denominator and
  retained numerator, preserving byte-identical replay output while removing
  duplicate geometry and exponential evaluation.
- Reused robust-cutoff membership as sparse per-component centroid indices.
  This removed 98.8% of centroid visits and reduced centroid worker time by
  64% while preserving a byte-identical canonical replay artifact.
- Deferred peak direction validation until after retained-component membership
  succeeds, reducing peak worker time by another 12% with byte-identical replay
  output.
- Reused refinement's exact observed support bounds for peak ownership,
  removing a complete support scan and another 1.04 worker-seconds from peak
  work with byte-identical replay output.
- Reused the generated compact range's finite-position invariant in robust
  preparation, reducing preparation worker time by 11% with byte-identical
  replay output.
- Carried full-halo support bounds from the fixed stencil into compact
  refinement, reducing robust preparation another 12% with byte-identical
  replay output.
- Cached configured direction eligibility once per unique compact observation,
  reducing robust preparation another 33% and anchor CPU by 2.5% with byte-
  identical replay output and no compact-record size increase.
- Split dense peak-denominator traversal from sparse positive evidence and
  removed the dense evidence-index stream, reducing peak and enclosing wall
  time with byte-identical replay output.

# 2026-08-20: chunk-native replay corridor selection

- Removed the serial full-corridor anchor-cell enumeration performed before
  cache-backed replay could start. Reference scheduling now selects storage
  chunks directly, while requested anchor chunks perform the canonical exact
  segment-to-cell test locally and retain existing NMS and path containment.
- On the Paris4 full-fiber radius-768 workload, cache work began after 0.006
  seconds instead of remaining in serial setup beyond 20 seconds. A bounded
  20.06-second cold-cache run consumed 560.16 CPU seconds (27.9 effective
  cores). The 5,000-base-voxel radius-64 replay remained byte-identical and
  completed in 5.90 seconds.

# 2026-08-20: live cached replay progress

- Added generated-or-persisted chunk-resolution accounting to the concise
  replay progress estimate and a bounded timer repaint so elapsed time, an
  estimated fraction, and ETA remain live during long cache preprocessing.
- Isolated progress observer failures from cache results and made ticker and
  resolution-state shutdown safe for late worker completion. Radius-64 replay
  output remained byte-identical.

# 2026-08-22: incremental fiberlet replay prefixes

- Replaced repeated logical-route vector construction and visited-set copying
  with exact canonical logical histories and immutable Patricia cycle state.
- Made selected-route reference matching, normal-threshold evaluation, and
  diagnostic index assignment process only newly selected history suffixes;
  complete public route output is assembled once at segment termination.
- Preserved the canonical hot-cache replay bundle byte-for-byte while reducing
  its median wall time from 7.57 to 3.87 seconds and peak RSS from 202,872 KiB
  to 102,424-104,900 KiB.

# 2026-08-20: cache-backed quantization comparison

- Replaced the eager full-population quantization CLI with sequential baseline
  and selected-scenario replays over persistent, fingerprinted, bounded caches.
- Applied endpoint position/direction quantization before fresh sampling and DP,
  projected compact logical IDs for ordering, and decoded per-owner-chunk costs.
- Added indexed line comparison with Euclidean, normal, and tangential
  distributions for baseline-to-scenario and both replay-to-reference paths.
- Kept quantized anchor identity cell-local after a full-corridor run showed
  that anchors from several adjacent cells can round to one Q4 coordinate.
- Added explicit exact-baseline cache reuse and collision-resistant atomic
  temporary writes so long interrupted comparisons resume safely.

# 2026-08-21: geometry-cache reuse across cost experiments

- Split anchor/fiberlet geometry quantization from replay-only cost decoding,
  allowing float, uint8, and uint16 cost comparisons to share persistent
  geometry and DP caches.
- Added `--scenario all`, which runs one baseline and all 17 non-baseline
  scenarios while creating at most eight geometry cache groups.
- Retained the historical u8-tagged namespace only as opaque compatibility
  metadata so the completed radius-768 Q4+axis cache remains reusable.
- Made anchor extraction canonical across all geometry scenarios. Quantized
  endpoint views are now derived once per loaded anchor chunk under a bounded
  single-flight LRU, while only fiberlet geometry remains scenario-specific.
- Added producer-group cache cancellation and draining for batch replay so
  completed full-corridor quantization results no longer hang or crash during
  process worker-pool teardown.
- Added float-position compact-axis `uint8` and `uint16` cost scenarios. Both
  are graph-only views over the existing compact-axis geometry cache.
- Full radius-768 validation confirmed the shared namespace. Because the
  existing compact-axis cache was only partially populated, its per-owner cost
  range scan completed missing geometry chunks in place; it did not create a
  cost-specific cache.

# 2026-08-21: focused quantization failure replay

- Added atomic baseline/scenario replay artifacts and exact failure-window
  reporting to the cache-backed quantization benchmark.
- Added base-voxel `--arc`/`--length` focused intervals and an original
  `--seed-key` override for deterministic replay of later failure segments.
- Replaced focused corridor pre-generation with demand-only access to the
  completed full-corridor cache and added ranked beam-frontier/cost/route
  comparison artifacts.
- Added full-route committed-fiberlet cost distributions plus configurable
  base-voxel exclusion around replay failures.

# 2026-08-21: fixed-distance fiberlet lookahead experiment

- Added an initial local base-voxel lookahead experiment with proportional
  active-cost scoring and clipped final-fiberlet geometry. It was not shipped
  and has now been replaced by the persistent search below.
- On the focused Q1 failure section, 192 base voxels approximates the measured
  three-edge median (189.1 base voxels). A hot-cache float/Q1 comparison
  completed with zero failures in both runs, no committed first-edge
  divergence, and 0.443/0.305/3.689 base-voxel mean/median/maximum line
  separation. Float/Q1 replay wall times were 2.05/2.26 seconds.

# 2026-08-21: rolling whole-fiberlet beam search

- Replaced both unpublished local lookahead implementations with one persistent
  beam of up to 16 whole-fiberlet histories from each uninterrupted segment
  seed.
- Added a shared logical checkpoint, 192-base-voxel lookahead, and default
  48-base-voxel checkpoint advance. Every retained history expands through all
  valid branches to the common logical horizon; the final edge remains whole.
- Added one deterministic global top-16 prune, whole-fiberlet commitment through
  the next checkpoint, shared immutable parent histories, explicit state bounds,
  clipped final-output materialization, and decision diagnostics.
- Parallelized independent retained-beam expansion under the existing
  `--threads` setting. The focused 600-base-voxel radius-768 hot-cache replay
  fell from 17.30 to 6.97 seconds while preserving the same failure and distance
  results.
- Added bounded equal-distance fronts and deterministic uniform-cost label
  search. Reconvergent histories now retain only the best logical-incoming-arc
  and front-offset label, while exact proportional front scoring and the exact
  A* oracle remain available for comparison.
- Restored immediate `fiber_replay_failure` output in compact-progress mode;
  failure lines now interrupt and redraw the progress bar instead of being
  suppressed unless `--stats` is enabled.

# 2026-08-22: configurable exact replay beam width

- Removed the unpublished fixed maximum of 16 graph-replay beams. `--beam`
  now accepts any positive width while retaining 16 as its default and the
  generated-state limit as the exact-search work bound.
- Changed exact completion retention to ordered insertion so wider experiments
  do not sort the complete retained set after every discovered route.
- Changed the graph replay default lookahead from 192 to 384 base voxels while
  retaining the independent default checkpoint advance of 48 base voxels.
- Failure-window diagnostics now use the available match search span when a
  replay reset fails without advancing near the reference end.
# 2026-08-22: compact float-position fiberlet default

- Adopted exact float endpoint positions, compact fitted directions, and fixed
  sqrt-density `uint16` edge costs with ceiling 256 as the default cache-backed
  fiberlet replay profile. Preserved an explicit all-float correctness oracle,
  canonical float anchors, and the unchanged unpublished compact serializer.
- Added fractional base-voxel endpoint evaluation and replaced the unpublished
  q1/u8 matrix row with the q1/8 compact-direction sqrt-u16 experiment.
- Full Paris4 radius-768 evaluation produced two failures for both the new
  default and exact oracle; q1/8 produced three.

# 2026-08-22: sparse whole-volume fiberlet preprocessing

- Added presence-only sparse scanning and deterministic output-chunk mapping.
- Added a retained intermediate anchor cache and a combined compact final Zarr
  with anchors, prefixes, and routes. Expected chunks and completeness are
  reconstructed directly from source presence, intermediate anchors, and final
  payload tuples without persisted activity or completion markers.
- Added read-only combined-dataset cache facets configured from a fresh source
  scan, atomic partial-tuple recovery, and locked stale-temp cleanup.
- Added `vc_fiberlets preprocess-volume` with Z/Y/X scheduling, resume, and
  one-second progress, ETA, and compressed-size projection with persistent
  minute checkpoints.
- Replaced the whole-volume anchor barrier with a dependency-driven Z-slab
  pipeline. Ready fiberlets dynamically consume the shared worker budget first;
  remaining workers generate deduplicated anchor lookahead.

# 2026-08-06: Lasagna manager durable Fiber runs

- Added scale-specific open-data prefetch through the existing downloader,
  backend-neutral immutable run records, detached tmux execution, authoritative
  exit/state logging, durable/live listings, and contextual tmux attachment.

# 2026-08-06: checkpoint-driven Fiber inference provenance

- Made embedded checkpoint configuration authoritative for Fiber inference,
  retaining an explicit positional config only for legacy snapshots.
- Added direct portable `inference.json` output with exact scale/settings,
  checkpoint and catalog identity, failure state, and bounded structural OME
  inventory; Lasagna manifests now preserve its relative reference and unknown
  forward-compatible fields.
- Connected manager run/catalog context to the direct inference writer without
  leaking host paths, command logs, or tmux identity into `artifacts/`.

# 2026-08-06: shared inference OME-Zarr compression

- Defaulted newly created Fiber and Lasagna inference pyramids to exact
  Zarr-v2 Blosc/Zstd level-3 byte-shuffle compression through their shared
  output-group creator.
- Added the shared `--ome-compressor` compatibility override and preserved
  existing per-level codecs on resume with an explicit mismatch warning.

## 2026-08-06 — Lasagna manager Phase 5 integration

- Made Bash and Zsh completion registry-derived and added cached dynamic
  snapshot, volume, inference, and live-run selectors without completion-time
  refresh or mutation.
- Made completed portable provenance part of the manager success contract and
  added bounded moved-bundle validation shared across artifact kinds.
- Validated a Fiber provenance mapping against the checked-out Atlas Pydantic
  `DataEntry` model and exercised a synthetic Lasagna artifact fixture.

## 2026-08-06 — Atlas staging and prediction ingestion

- Added shared Fiber/Lasagna portable-bundle validation and atomic, idempotent
  run-UUID staging uploads with commit markers and content manifests.
- Added Atlas provenance parsing, explicit model registration, and browser
  bundle registration while mapping both Fiber and Lasagna output to the
  existing Lasagna artifact and CC BY-NC publication rule.

## 2026-08-06 — Lasagna manager backend

- Added structure-based Lasagna checkpoint indexing, namespaced selectors, and
  shared manager/tmux launch dispatch through `predict3d`.
- Made direct Lasagna inference author the shared portable provenance envelope,
  including its manifest decoding fields and structural Zarr inventory.
- Reused the existing catalog, prefetch, run lifecycle, completion, staging,
  and Atlas Lasagna ingestion paths without a second orchestration workflow.
# 2026-08-06: self-contained detached manager inference

- Packaged sibling Fiber/Vesuvius and canonical Lasagna modules so inference
  needs no ambient `PYTHONPATH`.
- Moved automatic open-data prefetch into the tmux runner with an explicit
  lifecycle, making launch return immediately while retaining strict
  prefetch-before-GPU ordering.
- Replaced provenance-heavy paths with concise, collision-safe human labels
  while retaining canonical Atlas identity in metadata.
# 2026-08-06: manager defaults and stable tmux attachment

- Added initialized global inference params for 512-voxel tiles, 32-voxel
  borders, 96-voxel overlap, and all visible GPUs, with per-run overrides.
- Made managed tmux identity use atomically captured, run-UUID-tagged stable
  window IDs and distinguish orphan inference processes from attachable runs.
- Made attached inference panes show the same live byte stream retained in the
  durable run log.
# 2026-08-06: manager no-prefetch download delegation

- Made `inference run --no-prefetch` retain backend on-demand downloads while
  the default prefetch-first workflow continues to disable concurrent fetching.

# 2026-08-07: provenance-driven Atlas model registration

- Added shared direct/managed Fiber and Lasagna inference commit provenance.
- Replaced manual upload model selection with fresh checkpoint-hash resolution
  and automatic minimal Atlas model registration.
- Standardized Fiber Atlas models as `fiber3d/unet` Lasagna models with numeric
  references, relative snapshot path, and snapshot SHA-256.

## 2026-08-11 — Shared live selected-scale inference cache

- Added opt-in Fiber/Lasagna full-volume live S3 materialization with a bounded
  lazy tile window, authoritative active-plane inventory, atomic transfer, and
  conservative whole-Z-plane eviction behind the canonical commit frontier.
- Added selected-level reader/mutator locking, manager launch/provenance support,
  and 10 TiB / 10,000-tile defaults without changing the normal prefetch path.
- Made live progress cadence independent of TTY detection and report unique
  remote-missing chunks per listed Z plane.

## 2026-08-23 — Managed whole-volume Fiberlet artifacts

- Added durable `las_manager fiberlet` processing with shared tmux lifecycle,
  configurable native thread budget, strict upstream prediction validation,
  resumable local anchors, and portable final provenance.
- Made combined Fiberlet Zarr metadata self-describing and independent of
  runtime paths, with canonical processing/source fingerprints.
- Added bounded-memory staging and minimal Atlas `fiberlets` copy-first public
  representation support without registering a synthetic model.
# 2026-08-23: catalogue-backed managed Fiberlet normals

- Made the normal selector optional for managed Fiberlet jobs. The manager now
  distinguishes published regular Lasagna normals from Fiber Lasagna outputs
  through exact Atlas model channels, caches the remote manifest with the VC
  read-through marker, and lazily reuses normal chunks.
- Added explicit Atlas normal selectors, cache-only completion, stable remote
  dependency provenance/resume, and one-voxel base-shape compatibility.
- Reused VC3D's canonical open-data Lasagna cache URL, identity directory, and
  marker schema; manifest SHA-256 is now integrity metadata only.
- Invalid public origins, cache markers, coordinate metadata, and remote group
  descriptors now fail explicitly instead of selecting or constructing an
  alternate cache dependency.

## 2026-08-25 - Anchor-seeded Fiberlet crop tracing

- Added standalone bidirectional crop filling from combined Fiberlet datasets,
  with deterministic anchor seeds, first-exit clipping, cycle rejection, and
  ordinary stored edge/join costs.
- Added replay-compatible 20/80-base-voxel anisotropic anchor coverage with
  direction discrimination, while intentionally deferring output-fiber
  deduplication.
- Added combined sparse-dataset consumption without a source Fiber manifest:
  absent tuples decode as empty and partial tuples fail.
- Kept whole-volume production sparse as well: missing/all-zero presence input
  chunks are not scheduled, and generated empty anchor or Fiberlet chunks are
  not written.
- Added line OBJ and optional six-face CT texture visualization through the
  existing VC3D fine-to-coarse sampler.

## 2026-08-25 - Structural Fiberlet normal compatibility

- Replaced the crop tracer's exact Lasagna manifest byte-hash requirement with
  base-frame, shape, scale, and normal-channel compatibility validation.
- Reused the established Lasagna one-chunk padding rule and retained manifest
  hashes only as provenance.

## 2026-08-25 - Fiberlet crop principal-direction visualization

- Added deterministic non-orthogonal two-direction fitting over accepted
  traces' local axial steps and 75% arc-length dominant/mixed classification.
- Preserved the complete line OBJ while adding direction-1, direction-2, and
  mixed line subsets plus matching actual seed-anchor point OBJs.

## 2026-08-23 — Explicit shared inference Z-band lifecycle

- Replaced the cross-band multi-device event state machine with a lazy explicit
  Z-band barrier and exact shared-slot ownership for both Fiber and Lasagna.
- Made accumulator queue submission nonblocking and flush batches finalize and
  release their circular generations immediately after full acknowledgement.
- Replaced the flush timeout/process-limit guess with stage-aware quiescence
  diagnostics, while retaining bounded per-process Zarr and native/Blosc
  thread limits.
- Restored atomic read submission for ordinary local inputs after live-fetch
  integration had routed even its disabled path through the new staged state.
  Live materialization now remains a bounded upstream ledger feeding the same
  queue-based inference stages, with checked per-worker task and slot ownership.
- Fixed a proven canonical-frontier capacity inversion: one input/result pair
  is now reserved for the first non-skipped accumulation-frontier tile, so
  later out-of-order GPU results cannot occupy every result slot and deadlock
  canonical accumulation. Queue delivery remains fire-and-forget.

## 2026-08-26 - H/V-only fiber-piece labeling diagnostic

- Added an opt-in reduced HiGHS model that solves active/broken and H/V labels
  without parity variables, constraints, or winding costs.
- Preserved the existing CSV and five OBJ interface by reporting fixed-even
  parity, leaving the two odd visualization layers empty.
- Added an explicit diagnostic switch to retain all finite winding-distance
  constraints without changing the default 1.5-winding cutoff.
- Added an exact-perpendicular mixed H/V model with binary broken decisions,
  continuous piece orientation, true absolute-difference edge losses, and no
  triangle surrogate.
- Added deterministic iterative H/V consensus growth over original crop
  fibers, with spatial/count priority, incremental broken decisions, final
  H/V/broken OBJ layers, and scheduled growth snapshots.
- Restricted the primary consensus seed to long crop-spanning candidates and
  ranked equal-straightness candidates by true distance to the crop center.
- Added final and milestone broken-fiber OBJ layers alongside consensus H/V.
- Added opt-in deterministic mutual top-K constraint-strength pruning shared by
  constraint visualization, HiGHS labeling, and iterative consensus, with
  source-fiber degree and connectivity diagnostics plus minimum-link recovery
  of the original positive-strength component partition.
- Exposed the existing crop-direction dominance fraction to trace and stored
  visualization commands while preserving the 0.75 default.
- Replaced binary nearest-axis votes in crop direction grouping with calibrated
  per-segment angular support, so off-axis and bend length gradually moves a
  fiber into the mixed visualization group without changing the fitted axes.
- Added `vc_fiber_trace_chunk direction-diagnostic`, which removes mixed
  direction groups before canonical constraint extraction and discrete
  H/V-only MILP solving, then reports component-gauge-aligned orientation and
  broken-label errors against the initial direction assignment.
- Added `direction-ablation`, which cumulatively admits Mixed fibers by gradual
  direction confidence, independently rebuilds and solves coarse checkpoints,
  and reports H/V versus mixed-defect errors for both the discrete MILP and its
  thresholded LP relaxation under a trusted-only component gauge.
- Added a ranked-prefix admission limit for fast, repeatable broken-cost
  ablation sweeps.
- Added an opt-in perpendicular-only labeling filter for constraint and
  direction-ablation experiments, retaining existing same-trace continuity
  evidence and reporting excluded measured links.
- Added an opt-in no-split post-solve perpendicular consensus diagnostic with
  confidence-weighted synchronous updates and short fixed-value OBJ layers.

## 2026-08-27 - Binary BP consistency diagnostics

- Added a HiGHS-free final-cohort BP-only direction-ablation path using the
  shared perpendicular constraint selector.
- Added per-fiber hard, unresolved, strength-weighted, soft same-label, neighbor
  support-balance, and neighbor-certainty diagnostics with CSV output, grouped
  quantiles, and Mixed-vs-trusted AUROC.
