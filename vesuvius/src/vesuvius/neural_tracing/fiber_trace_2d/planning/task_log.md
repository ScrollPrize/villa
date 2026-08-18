# Task Log: staged fiberlet anchor acceleration

## Baseline

- Commit: `73fe64e09` (`Use float peak scoring for fiber anchors`).
- Command:
  `volume-cartographer/build/bin/vc_fiberlets fiberlet-replay /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fibers/david/Paris4_fibers/dj_20260805T025256484_000003.json /tmp/fiberlet-replay --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --threads 32 --length 5000 --maximum-iterations 2`.
- Latest measurement: 16.37 seconds total wall, 9.76 seconds anchor wall,
  277.38 seconds anchor CPU, 5.97 seconds fiberlet wall, and 159.68 seconds
  fiberlet CPU.
- Work/quality: 2520 anchors, 48,972 searched / 24,518 accepted fiberlets,
  170,500 sampled voxels, 47,924,048 interpolated/DP node entries, 2 greedy
  failures, and 1 fiberlet failure.

## Workflow Notes

- Implement one checkpoint at a time and stop after reporting its benchmark.
- Small numeric differences are allowed; accepted populations and replay
  quality remain explicit review inputs.
- Independent subagent plan review was not run because this session has no
  user authorization to delegate work. The plan was checked directly against
  `AGENTS.md`, the current specs, and the previous measured task record.

## Checkpoint 1: Tile-Owned Compact Observations

- Production extraction now creates one compact float32 observation per tile
  voxel, normalizes each sampled direction once, and reuses those records via
  canonical-order 32-bit cell indices. A parallel byte vector preserves the
  previous cell-halo gradient-validity rule. The expanded public vector API and
  indexed production path instantiate the same fitter.
- The memory budget includes compact tile records and maximum cell index/
  validity scratch. Temporary decoded samples and gradient caches are released
  before cell fitting.
- Three canonical runs produced identical work/quality populations: 2520
  anchors, 48,972 searched / 24,518 accepted fiberlets, 170,500 sampled voxels,
  47,924,054 DP node entries, 2 greedy failures, and 1 fiberlet failure.

  | metric | minimum | median | maximum | baseline |
  |---|---:|---:|---:|---:|
  | total wall | 14.54 s | 14.73 s | 15.18 s | 16.37 s |
  | total CPU | 385.20 s | 387.73 s | 401.72 s | 439.28 s |
  | anchor wall | 8.03 s | 8.16 s | 8.52 s | 9.76 s |
  | anchor CPU | 224.27 s | 224.72 s | 230.33 s | 277.38 s |
  | fiberlet wall | 5.88 s | 5.93 s | 6.02 s | 5.97 s |
  | fiberlet CPU | 158.76 s | 160.83 s | 162.01 s | 159.68 s |

- Relative to the single recorded baseline run, median total wall improved
  10.0%, median anchor wall improved 16.4%, and median anchor CPU improved
  19.0%. Observation-construction worker time fell from 41.89 to a 17.08-17.20
  second range; final-evaluation and setup time also fell because directions
  are normalized once per tile.
- Float normalization changed seed work slightly (95,344 to 95,450 seeds) and
  a few downstream DP counts, but did not change retained populations or replay
  failures. The accepted checkpoint was committed as `9ef0876f7`.

## Checkpoint 2: Reuse Robust State

- Tested fusing the fixed-axis baseline spatial objective into retained
  centroid accumulation. Baseline Gaussian values were reused for centroid
  weights whenever projected and clamped positions matched, and subsequent
  backtracking evaluated only the moved candidate.
- The implementation preserved all measured work/quality populations across
  three runs: 2520 anchors, 48,972 searched / 24,518 accepted fiberlets,
  170,500 sampled voxels, 47,924,054 DP node entries, 2 greedy failures, and 1
  fiberlet failure.

  | metric | minimum | median | maximum | checkpoint 1 median |
  |---|---:|---:|---:|---:|
  | total wall | 14.59 s | 14.75 s | 15.04 s | 14.73 s |
  | total CPU | 384.39 s | 389.04 s | 398.17 s | 387.73 s |
  | anchor wall | 8.18 s | 8.24 s | 8.43 s | 8.16 s |
  | anchor CPU | 225.57 s | 227.69 s | 234.21 s | 224.72 s |
  | fiberlet wall | 5.79 s | 5.89 s | 5.98 s | 5.93 s |
  | fiberlet CPU | 156.65 s | 159.19 s | 161.80 s | 160.83 s |

- Median total wall regressed 0.1%, anchor wall regressed 1.0%, and anchor CPU
  regressed 1.3%. Local-refinement worker time rose to a 109.98-112.73 second
  range. The existing paired baseline/first-candidate evaluator already shares
  one observation traversal; moving the baseline into centroid construction
  instead turned the formerly sparse retained-evidence centroid calculation
  into an all-site pass and worsened locality.
- Final membership and final support were not fused: direction-conditioned peak
  refinement changes component positions between those phases, so Gaussian
  support state is not reusable without changing the fitting semantics or
  retaining another large per-observation stream.
- Focused GCC and Clang builds and tests passed on the experiment. Because it
  did not improve performance, the code and temporary profile-schema change
  were removed. Production remains at accepted checkpoint 1 pending review of
  checkpoint 3.

## Checkpoint 3: Batched Peak Responses

- Tested three locality strategies against checkpoint 1 without changing the
  candidate domain, exact circular cutoff, acceptance checks, or retained
  replay populations.
- Batched-neighborhood evaluation computed each hill-climb neighborhood in one
  observation traversal. It reduced 261,943 logical grid responses to 56,599
  physical observation scans, but peak-search worker time increased to 39.18
  seconds due to extra branch and accumulator pressure.
- A two-dimensional contiguous counting-sort/CSR broad phase reduced physical
  candidate visits from 2.37 billion logical visits to 0.98 billion. Peak
  search still increased to 39.20 seconds because many short bin ranges
  disrupted sequential observation access.
- A one-dimensional contiguous counting sort retained longer ranges and
  reduced physical candidate visits to 1.74 billion. It performed worse:
  peak-search worker time was 42.13 seconds, anchor wall was 8.38 seconds,
  anchor CPU was 231.34 seconds, and total wall was 14.91 seconds. Checkpoint 1
  medians were 8.16 seconds, 224.72 seconds, and 14.73 seconds respectively.
- Every variant retained 2520 anchors, 48,972 searched / 24,518 accepted
  fiberlets, 170,500 sampled voxels, 47,924,054 DP nodes, 2 greedy failures,
  and 1 fiberlet failure.
- The experiments show that the compact sequential scan is cheaper than these
  reductions in arithmetic count. All checkpoint-3 production and profile
  changes were removed; production remains at accepted checkpoint 1.

## Checkpoint 4: One Robust Pass

- Tested the existing one-pass configuration with
  `--maximum-iterations 1`; no production code change was required. A stale
  profile-version-7 executable was detected after the first attempt, rebuilt,
  and that invalid timing was discarded.
- Three clean profile-version-6 runs produced:

  | metric | minimum | median | maximum | checkpoint 1 median |
  |---|---:|---:|---:|---:|
  | total wall | 13.68 s | 13.69 s | 13.77 s | 14.73 s |
  | total CPU | 356.34 s | 359.31 s | 360.00 s | 387.73 s |
  | anchor wall | 6.69 s | 6.72 s | 6.79 s | 8.16 s |
  | anchor CPU | 183.92 s | 184.80 s | 185.81 s | 224.72 s |
  | fiberlet wall | 6.25 s | 6.33 s | 6.34 s | 5.93 s |
  | fiberlet CPU | 170.17 s | 171.99 s | 172.31 s | 160.83 s |

- Median total wall improved 7.1%, total CPU improved 7.3%, anchor wall
  improved 17.7%, and anchor CPU improved 17.8%. Fiberlet work became slower
  because the one-pass result retained more anchors and generated more
  candidates.
- Deterministic work/quality changed from 2520 to 2603 anchors, 48,972 to
  51,782 searched fiberlets, 24,518 to 26,494 accepted fiberlets, 170,500 to
  170,813 sampled voxels, and 47,924,054 to 50,822,225 DP nodes. Replay
  remained at 2 greedy failures and 1 fiberlet failure.
- Matched visualizations were generated in
  `/tmp/fiberlet-replay-two-pass-vis` and
  `/tmp/fiberlet-replay-one-pass-vis`. Cell-local pairing in the two
  identical greedy-failure windows found median displacement of 0.66-0.73 base
  voxels and median axis changes of 1.66-2.01 degrees. The p95 displacement was
  20.0-25.1 base voxels and p95 axis change was 12.8-16.0 degrees, with
  24-37 unmatched anchors from either result per window.
- The one-pass result was accepted as the new default after visual review.
  `--maximum-iterations` remains available and is documented prominently as
  a quality/speed knob because additional passes materially change difficult
  overlapping-fiber fits.

## Checkpoint 5: Shared Tile-Halo Sampling

- Exact coordinate-compressed union accounting found 39,701,808 total tile
  coordinates but only 6,162,456 globally unique tile voxels: 84.5% of the
  original submissions were duplicate halo coordinates.
- Retained a conservative reuse design that pairs tiles deterministically by
  maximum overlap. Each pair is one bounded worker job. The second tile copies
  overlapping raw prediction samples and submits only missing coordinates;
  gradients, compact observations, and cell processing remain unchanged.
- The canonical workload forms 98 groups from 192 tiles, keeps 32 workers,
  reuses 12,960,096 samples, and reduces actual sampler submissions by 32.6%
  to 26,741,712.

  | metric | minimum | median | maximum | checkpoint 4 median |
  |---|---:|---:|---:|---:|
  | total wall | 13.54 s | 13.54 s | 13.55 s | 13.69 s |
  | total CPU | 343.82 s | 345.87 s | 349.19 s | 359.31 s |
  | anchor wall | 6.64 s | 6.65 s | 6.70 s | 6.72 s |
  | anchor CPU | 173.22 s | 174.07 s | 175.78 s | 184.80 s |
  | prediction sampling work | 22.26 s | 23.08 s | 23.15 s | 42.44 s |

- Median total wall improved 1.1%, total CPU improved 3.7%, and anchor CPU
  improved 5.8%. Coordinate construction increased by about 0.8 summed-worker
  seconds due to overlap copy/scatter bookkeeping.
- All three runs retained exactly 2603 anchors, 51,782 searched / 26,494
  accepted fiberlets, 170,813 sampled fiberlet voxels, 50,822,225 DP nodes,
  2 greedy failures, and 1 fiberlet failure.

## Checkpoint 6: Direct Packed-Key DP Index

- Replaced each candidate's `unordered_map<uint32_t, size_t>` node index with
  a direct `uint32_t` table indexed by the existing packed local-lattice key.
  Missing entries use an explicit sentinel; node/transition order and all DP
  cost calculations are unchanged.
- The canonical workload stored 50,718,661 nodes in 109,469,154 table slots,
  for 46.3% occupancy. Aggregate direct-table payload is about 438 MB versus
  the prior 1.22 GB hash-storage estimate; both figures sum per-candidate
  transient storage rather than concurrent peak RSS.

  | metric | minimum | median | maximum | checkpoint 5 median |
  |---|---:|---:|---:|---:|
  | total wall | 13.21 s | 13.26 s | 13.33 s | 13.54 s |
  | total CPU | 335.40 s | 337.69 s | 340.78 s | 345.87 s |
  | anchor wall | 6.62 s | 6.63 s | 6.72 s | 6.65 s |
  | anchor CPU | 172.44 s | 173.35 s | 175.77 s | 174.07 s |
  | fiberlet wall | 5.94 s | 5.97 s | 6.01 s | not recorded |
  | fiberlet CPU | 160.77 s | 162.17 s | 162.83 s | not recorded |
  | direct-index work | 0.390 s | 0.398 s | 0.402 s | not recorded |
  | peak RSS | 2.49 GiB | 2.50 GiB | 2.55 GiB | not recorded |

- Median total wall improved 2.1% and total CPU improved 2.4%. Anchor timing
  stayed within run noise; the gain is isolated to fiberlet search.
- All runs retained exactly 2603 anchors, 51,782 searched / 26,494 accepted
  fiberlets, 170,813 sampled voxels, 50,822,225 DP nodes, 282,121,134
  transition lookups, 2 greedy failures, and 1 fiberlet failure. Their
  `fiber_replay.json` files have identical SHA-256
  `eab2edfb769986649c900ebf5b5ab72fb2d615c07ba06e88cd3a1868cecaa98b`.

## Checkpoint 7: Paged Scoring Lookup

- A temporary canonical measurement compared dense index pages around the
  170,813 unique sampled scoring voxels:

  | page | pages | slots | occupancy | stencil page probes |
  |---:|---:|---:|---:|---:|
  | `4^3` | 4,131 | 264,384 | 64.6% | 98,846,028 |
  | `8^3` | 825 | 422,400 | 40.4% | 72,225,851 |
  | `16^3` | 199 | 815,104 | 21.0% | 61,144,696 |

- Selected `16^3` pages. Their dense `uint32_t` indices require about 3.1 MiB
  and reduce sparse-directory probes by 6.6x from the 406,380,154 original
  per-corner voxel hash lookups. The temporary measurement code was removed.
- Each interpolation owns an eight-entry page cache. The sparse page directory
  resolves a page at most once per stencil; dense local offsets then resolve
  its corners. Interpolation arithmetic and corner order are unchanged.

  | metric | minimum | median | maximum | checkpoint 6 median |
  |---|---:|---:|---:|---:|
  | total wall | 13.14 s | 13.22 s | 13.24 s | 13.26 s |
  | total CPU | 333.02 s | 335.86 s | 336.96 s | 337.69 s |
  | interpolation materialization wall | 2.28 s | 2.30 s | 2.31 s | 2.40 s |
  | interpolation materialization CPU | 72.21 s | 72.96 s | 73.22 s | 75.99 s |
  | peak RSS | 2.50 GiB | 2.50 GiB | 2.52 GiB | 2.50 GiB |

- Median materialization wall improved 4.2%, but total wall improved only 0.3%
  and total CPU 0.5%. The remaining materialization cost is predominantly the
  interpolation math rather than sparse lookup.
- All populations, DP counters, and failures match checkpoint 6. All three new
  runs and the checkpoint-6 reference produced identical `fiber_replay.json`
  SHA-256 `eab2edfb769986649c900ebf5b5ab72fb2d615c07ba06e88cd3a1868cecaa98b`.

## Checkpoint 8: Prepared Scoring Tensors

- A bounded one-in-4096-per-worker profile sampled 12,423 of 50,822,225
  interpolations and 99,344 corners. Its measured phase shares were 15.7%
  sparse lookup, 23.1% prediction corner work, 21.1% normal corner work,
  21.2% prediction principal-axis resolution, and 18.9% normal principal-axis
  resolution. Neither prediction nor normal used the identical-axis shortcut.
- Each of 170,813 unique scoring voxels now validates and normalizes its
  prediction and normal once. The prepared representation stores float32 axes,
  presence, validity, and six symmetric outer-product components. Interpolation
  still accumulates weighted components in double precision and uses the same
  principal-axis solver.

  | metric | minimum | median | maximum | checkpoint 7 median |
  |---|---:|---:|---:|---:|
  | total wall | 12.48 s | 12.51 s | 12.69 s | 13.22 s |
  | total CPU | 315.48 s | 315.81 s | 317.49 s | 335.86 s |
  | interpolation materialization wall | 1.69 s | 1.70 s | 1.72 s | 2.30 s |
  | interpolation materialization CPU | 53.82 s | 53.89 s | 54.45 s | 72.96 s |
  | peak RSS | 2.50 GiB | 2.51 GiB | 2.53 GiB | 2.50 GiB |

- Median total wall improved 5.4%, total CPU 6.0%, materialization wall 26.1%,
  and materialization CPU 26.1%. Scoring preparation itself costs under 0.01
  seconds wall.
- All three optimized runs produced the same replay and the same fiberlet OBJ
  hash as checkpoint 7. Anchor/fiberlet populations and replay failures were
  unchanged. The output JSON differs only in loss values around `1e-6`; DP
  transition lookups changed by 9, reached-state visits by 1, and relaxations
  by 7 out of hundreds of millions.

## Checkpoint 9: Closed-Form Principal Axis

- Fiberlet interpolation now uses analytic eigenvalues for each symmetric 3x3
  tensor and reconstructs the dominant eigenvector from the largest stable
  cross-product of rows of `A - lambda I`. A scale-aware residual check gates
  the existing Jacobi fallback. Ambiguous top eigenvalues remain invalid and
  do not invoke a fallback.
- Each canonical run performed 50,822,225 prediction and 50,822,225 normal
  closed-form resolutions. All three recorded zero iterative fallbacks.

  | metric | minimum | median | maximum | checkpoint 8 median |
  |---|---:|---:|---:|---:|
  | total wall | 12.36 s | 12.42 s | 12.42 s | 12.51 s |
  | total CPU | 310.05 s | 310.98 s | 311.10 s | 315.81 s |
  | interpolation materialization wall | 1.50 s | 1.55 s | 1.55 s | 1.70 s |
  | interpolation materialization CPU | 47.38 s | 48.81 s | 48.81 s | 53.89 s |
  | peak RSS | 2.50 GiB | 2.50 GiB | 2.50 GiB | 2.51 GiB |

- Median materialization wall improved 8.8%, materialization CPU 9.4%, total
  wall 0.7%, and total CPU 1.5%. Anchor/fiberlet populations, DP work, replay
  failures, and complete replay artifacts were unchanged. Checkpoint 8 and all
  checkpoint-9 runs have SHA-256
  `83bfadf690ac5d3badcd6a07822d95c2fa2d44fcb06e28dd8d821e308d4c7197`.

## Checkpoint 10: Prepared DP Nodes And Edges

- Checkpoint 9 is the baseline: median total wall 12.42 seconds, total CPU
  310.98 seconds, fiberlet DP about 1.85 seconds wall / 58 CPU-seconds,
  50,718,661 retained nodes, 282,121,125 transition lookups, 31,364,056
  reached-state visits, and 62,970,698 relaxations.
- Planned independent variants are solve-local expanded node data, outgoing
  edge reuse per reached node, a full candidate edge table, and compact DP
  states with key-derived predecessors/incoming geometry. A prepared normalized
  scoring path is conditional on the resulting profile.
- Numeric identity is not required, but each retained variant must preserve
  acceptable geometry, replay outcomes, and bounded memory.
- Independent review confirmed the predecessor invariant with corrections:
  states 0--8 derive `(layer-1,u-du,v-dv)` and retain the predecessor's state;
  state 9 is source-only; source/sink handling stays outside interior edge
  tables; the direct index survives through reconstruction; the initial
  variants preserve the double strict prediction-deviation gate.
- Checkpoint-10 measurements use the checkpoint-9 one-pass workload, not the
  obsolete two-pass command still recorded for the original checkpoint.
- Each variant reports exact solve-local allocation estimates and reached/edge
  reuse counters in addition to phase timings and RSS.

### Variant 10A: Solve-Local Prepared Nodes

- Cached decoded prediction axes, normal axes, presence, and validity once for
  every retained node. Transition order, DP state, feasibility, and scoring
  arithmetic remained unchanged.
- One canonical run measured 12.41 seconds total wall and 303.89 seconds total
  CPU. Search measured 1.624 seconds wall / 50.63 CPU-seconds, comprising 0.682
  node-index, 2.810 node-preparation, and 47.63 DP worker-seconds. Checkpoint 9
  search was about 58 CPU-seconds and total CPU median was 310.98 seconds.
- The largest candidate used 125,312 prepared-node bytes, 13,872 direct-index
  bytes, and 1,879,680 state bytes. Peak RSS was 2,624,136 KiB.
- Populations, transition/state/relaxation counts, 2 greedy / 1 fiberlet replay
  failures, and `fiber_replay.json` SHA-256
  `83bfadf690ac5d3badcd6a07822d95c2fa2d44fcb06e28dd8d821e308d4c7197`
  matched checkpoint 9 exactly. Retain this variant.

### Variant 10B: Reached-Node Outgoing-Edge Reuse

- Constructed each reached node's at most nine neighbor lookups, physical edge
  geometry, and strict prediction-deviation result once, then reused the
  descriptors while retaining the canonical incoming-state/outgoing-edge loop
  order for strict ties.
- One canonical run measured 12.32 seconds total wall and 295.40 seconds total
  CPU. Search fell from variant 10A's 1.624 seconds wall / 50.63 CPU-seconds to
  1.435 seconds / 43.39 CPU-seconds; DP work fell from 47.63 to 40.91 seconds.
- Only 8,670,563 of 50,718,661 retained nodes were reached. They generated
  77,960,094 physical neighbor descriptors, of which 30,566,294 were valid;
  91,494,705 repeated valid-edge constructions were avoided across additional
  incoming states. Logical transition/state/relaxation counts were unchanged.
- Peak RSS was 2,616,320 KiB. Populations, failures, and replay SHA-256 remained
  identical to checkpoint 9. Retain this variant.

### Variant 10C: Candidate-Wide Edge Table

- Pre-generated every interior node's edge table after node/index preparation.
  This generated 453,751,032 physical descriptors and 114,243,122 valid edges,
  versus 77,960,094 / 30,566,294 for reached-node generation.
- Table-backed DP improved from 40.91 to 36.20 CPU-seconds, but edge preparation
  cost 27.95 CPU-seconds. Combined search regressed from variant 10B's 43.39 to
  66.60 CPU-seconds and from 1.435 to 2.152 seconds wall. Total wall was 13.05
  seconds and total CPU 320.49 seconds.
- The largest edge table occupied 720,544 bytes. Peak RSS was 2,607,264 KiB;
  output and failures remained identical.
- Reject and remove this variant. Reachability is sparse enough that eagerly
  processing every retained node overwhelms the modest table-backed DP gain.

### Variant 10D: Compact Key-Derived State

- Reduced `DpState` from 96 to 48 bytes by retaining cumulative cost,
  reachability, and the predecessor's state index. States 0--8 derive their
  predecessor node and incoming geometry from `(layer-1,u-du,v-dv)`; state 9
  terminates reconstruction at the source. Missing/invalid derived predecessors
  fail loudly.
- One canonical run measured 12.11 seconds total wall and 289.55 seconds total
  CPU. Search improved from variant 10B's 1.435 seconds / 43.39 CPU-seconds to
  1.192 seconds / 37.23 CPU-seconds; DP work was 35.00 seconds.
- Largest-candidate state storage halved from 1,879,680 to 939,840 bytes. Peak
  RSS remained dominated by earlier phases at 2,625,512 KiB, while process
  minor faults fell from 1,205,966 to 437,594.
- Populations, all logical DP counters, failures, and the replay hash remained
  identical. Retain this variant.

### Variant 10E: Prepared Normalized Metric Inputs

- Extracted one shared prepared-input entry point from `FiberLocalScoring`.
  Existing callers retain the validating/normalizing wrapper; fiberlet DP
  prepares node axes and each physical incoming/outgoing step once, then uses
  the same alignment and smoothness equations without repeated normalization
  or metric-config construction.
- One canonical run measured 11.90 seconds total wall and 284.89 seconds total
  CPU. Search improved from variant 10D's 1.192 seconds / 37.23 CPU-seconds to
  1.073 seconds / 33.21 CPU-seconds. Node preparation increased from 2.72 to
  5.45 CPU-seconds; DP fell from 35.00 to 28.44 CPU-seconds.
- Largest prepared-node storage increased from 125,312 to 187,968 bytes. Peak
  RSS fell to 2,577,104 KiB.
- Fiberlet geometry/OBJ, populations, and replay failures were unchanged. One
  relaxation changed out of 62,970,698 and serialized smoothness costs changed
  by small float-scale amounts, so the JSON hash changed as permitted. Retain
  this variant.

### Variant 10F: Slim Prepared Nodes

- Removed redundant double normal and presence fields. Interior scoring uses
  prepared float data; the strict prediction gate retains its cached double
  axis, while source/sink normals use the authoritative compact node.
- One canonical run measured 11.77 seconds total wall and 282.82 seconds total
  CPU. Search was 1.045 seconds / 32.81 CPU-seconds, with node preparation down
  from 5.45 to 4.85 CPU-seconds and DP at 28.19 seconds.
- Largest prepared-node storage returned from 187,968 to 125,312 bytes.
  Geometry, counters, and failures matched variant 10E. Retain this variant.

### Variant 10G: Rolling Layer-Local DP Costs

- Retained one byte of predecessor-state identity for every global node/state,
  but kept cumulative double costs only for the current and next DAG layers.
  Canonical node/state/edge relaxation order and reconstruction are unchanged.
- One canonical run measured 11.81 seconds total wall and 281.25 seconds total
  CPU. Search improved from variant 10F's 1.045 seconds / 32.81 CPU-seconds to
  1.000 seconds / 31.64 CPU-seconds; DP work fell from 28.19 to 26.71 seconds.
- Largest-candidate rolling state/backpointer storage was 214,680 bytes versus
  939,840 bytes for the all-node compact state.
- The prior all-node loop also generated dead outgoing lookups from the final
  interior layer before separately finalizing it to the sink. The rolling loop
  stops at that layer, reducing reached-node/state visits and logical lookups
  while preserving all 62,970,699 relaxations and producing byte-identical
  output to variant 10F. Retain this variant.

### Variant 10H: Float Cumulative Costs

- Stored the five cumulative DP cost components as float32, matching the
  precision of every local metric contribution, while retaining public/output
  costs as doubles at the boundary.
- One canonical run measured search at 0.983 seconds / 31.03 CPU-seconds versus
  variant 10G's 1.000 seconds / 31.64 CPU-seconds. DP work fell from 26.71 to
  26.11 CPU-seconds and largest rolling state storage from 214,680 to 114,630
  bytes. Total run time was noisy at 12.04 seconds wall / 283.89 CPU-seconds.
- Ten relaxation decisions changed out of 62,970,699. Selected fiberlet
  geometry/OBJ, populations, and 2 greedy / 1 fiberlet failures were unchanged;
  serialized costs changed at expected float accumulation scale. Retain this
  variant under the task's explicit float/numeric-relaxation permission.

### Final Checkpoint 10 Composition

- Removed an instrumentation-only serial scan over all 50,718,661 retained
  nodes. Maximum adjacent-layer population is now accumulated while each
  candidate's nodes are already generated in layer order. Profile residual
  wall time returned from about 0.18 seconds to 0.006--0.008 seconds.
- Three canonical final runs:

  | metric | minimum | median | maximum | checkpoint 9 median |
  |---|---:|---:|---:|---:|
  | total wall | 11.81 s | 11.91 s | 12.08 s | 12.42 s |
  | total CPU | 283.63 s | 283.73 s | 289.31 s | 310.98 s |
  | fiberlet wall | 4.51 s | 4.52 s | 4.56 s | 4.60 s |
  | fiberlet CPU | 107.31 s | 108.20 s | 109.40 s | 119.03 s |
  | search wall | 0.989 s | 0.996 s | 1.005 s | about 1.85 s |
  | search CPU | 31.02 s | 31.39 s | 31.78 s | about 58 s |
  | peak RSS | 2.46 GiB | 2.48 GiB | 2.49 GiB | 2.50 GiB |

- Every run prepared 50,718,661 nodes. Search reached 6,400,256 nodes,
  generated 57,574,182 outgoing descriptors, retained 30,566,294 valid
  descriptors, and avoided 91,494,705 repeated constructions. It performed
  189,586,809 logical lookups, 21,071,661 reached-state visits, and 62,970,689
  relaxations.
- Largest-candidate solve-local storage was 125,312 prepared-node bytes,
  13,872 direct-index bytes, and 114,630 rolling-state/backpointer bytes.
- All three final JSON artifacts have SHA-256
  `41fa73c76bc3a20528d064e2baed78552a20bed41542f9ed4e2ddcfb5e739215`.
  Selected geometry, OBJ content, populations, and 2 greedy / 1 fiberlet
  failures match variant 10G/checkpoint 9; expected float cumulative-cost
  serialization differences remain.
- GCC and Clang builds passed for `test_fiber_anchors`,
  `test_fiberlet_paths`, and `test_fiber_replay`; both focused CTest runs passed
  all three tests. `test_fiberlet_paths` now directly checks equality between
  validating and prepared local-scoring components. `git diff --check` passed.
