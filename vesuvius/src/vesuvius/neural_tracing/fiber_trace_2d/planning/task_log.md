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

## Checkpoint 11: Lazy Node Scoring Materialization

- Kept global native-corner collection, prediction/normal sampling, prepared
  scoring voxels, and the immutable paged scoring index. Exact endpoints remain
  eager; each candidate lazily interpolates interior nodes into its own direct
  cache. A shared compact conversion helper preserves the prior encode/decode,
  presence rounding, validity, strict gate, and scoring boundaries.
- The independent plan review required cache indices instead of unstable
  references, candidate-local mutable lookup state, explicit shared/local
  memory accounting, stable candidate/key-hash profiling, and separate
  endpoint/request/miss/hit counters. All corrections were incorporated.
- A first run exposed three changed relaxations because source/final transitions
  consumed a prepared float normal. Retaining the compact normal bytes for
  those transitions restored exact checkpoint-10 arithmetic and artifact hash.
- Reached current and final nodes use a checked existing-cache lookup, avoiding
  8,670,563 redundant miss-capable requests without changing scoring.

  | metric | minimum | median | maximum | checkpoint 10 median |
  |---|---:|---:|---:|---:|
  | total wall | 10.65 s | 10.76 s | 10.85 s | 11.91 s |
  | total CPU | 244.49 s | 248.84 s | 251.92 s | 283.73 s |
  | fiberlet wall | 3.25 s | 3.30 s | 3.34 s | 4.52 s |
  | fiberlet CPU | 68.14 s | 69.38 s | 70.28 s | 108.20 s |
  | endpoint materialization wall | 0.0147 s | 0.0152 s | 0.0165 s | 1.52 s all-node |
  | search wall | 1.268 s | 1.293 s | 1.311 s | 0.996 s |
  | search CPU | 39.99 s | 40.66 s | 41.10 s | 31.39 s |
  | peak RSS | 2.11 GiB | 2.11 GiB | 2.12 GiB | 2.46 GiB |

- Every run retained 50,718,661 geometry nodes, interpolated 103,564 endpoints
  and 14,478,750 unique lazy nodes, issued 59,294,549 lazy requests, and served
  44,815,799 cache hits. The shared prepared-scoring/page-index payload was
  16,930,232 bytes; largest candidate-local node map/cache/state payloads were
  7,832 / 131,072 / 114,630 bytes.
- DP populations and all counters matched checkpoint 10, including 62,970,689
  relaxations. All runs retained 2 greedy / 1 fiberlet replay failures and
  produced SHA-256
  `41fa73c76bc3a20528d064e2baed78552a20bed41542f9ed4e2ddcfb5e739215`.
- GCC and Clang builds and focused `test_fiber_anchors`,
  `test_fiberlet_paths`, and `test_fiber_replay` CTest runs passed. The path
  suite has 44 cases and now covers endpoint-only zero-lazy work, pruned
  materialization, request/miss/hit accounting, serial/parallel counter parity,
  and the existing invalid/ambiguous interpolation behavior. `git diff
  --check` passed.

## Checkpoint 12: Canonical Anchor Support Stencil

- Reused the canonical command and warmed local inputs recorded in the
  baseline section with the regular GCC `QuickBuild` tree, 32 threads, and
  three measured repetitions. The Clang validation tree also uses `QuickBuild`.
- Complete cells with a full volume halo now expand one immutable ordered
  `(z, y, xBegin, xEnd)` owned-or-radius stencil through each tile's actual
  strides. Partial cells and cells clipped by a volume boundary retain the
  prior scalar scan. Crop and tile boundaries do not affect eligibility.
- The independent plan review required 3D rather than flattened canonical
  offsets, exact full-halo predicates, a scalar ordered-index oracle, odd/even
  cell and gradient cases, multiple cells with differing tile/cell origins,
  explicit fallback coverage, logical counter invariants, and profile version
  15. The implementation incorporates those corrections.
- The canonical workload used the stencil for all 13,027 work cells and kept
  exact checkpoint-11 populations: 2,603 anchors, 51,782 searched / 26,494
  accepted fiberlets, 170,813 sampled voxels, and 2 greedy / 1 fiberlet replay
  failures.

  | metric | minimum | median | maximum | checkpoint 11 median |
  |---|---:|---:|---:|---:|
  | total wall | 10.37 s | 10.37 s | 10.40 s | 10.76 s |
  | total CPU | 238.72 s | 238.78 s | 238.88 s | 248.84 s |
  | anchor wall | 6.397 s | 6.416 s | 6.463 s | 6.817 s |
  | anchor CPU | 167.10 s | 167.49 s | 167.96 s | 177.28 s |
  | observation construction worker time | 11.73 s | 11.84 s | 11.98 s | 18.99 s |
  | fiberlet wall | 3.272 s | 3.294 s | 3.298 s | 3.299 s |
  | peak RSS | 2.08 GiB | 2.08 GiB | 2.10 GiB | 2.11 GiB |

- Median observation construction improved 37.6%, anchor CPU 5.5%, total CPU
  4.0%, and total wall 3.6%. All three `fiber_replay.json` files retained exact
  SHA-256
  `41fa73c76bc3a20528d064e2baed78552a20bed41542f9ed4e2ddcfb5e739215`.
- GCC and Clang builds and focused `test_fiber_anchors`,
  `test_fiberlet_paths`, and `test_fiber_replay` CTest runs passed. Tests cover
  scalar-order/stride equivalence, odd/even cells, gradient/no-gradient halos,
  serial/parallel full-stencil extraction, exact full-halo eligibility, and a
  partial-cell fallback.

## Checkpoint 13: Inline Robust Membership (Rejected)

- Replaced the materialized retained/not-retained byte vector experimentally
  with component assignments, residual bins, two cutoff bins, and one shared
  inline predicate. Final membership was move-owned by refinement state and
  consumed by const reference through centroid, objective, peak, and final
  support loops.
- Independent review added explicit move ownership, a shared internal test seam,
  component-compaction lifetime requirements, cutoff/bin boundary coverage,
  and the version-15 logical-visit accounting invariant.
- Clang exposed an all-components-removed path where final evaluation entered
  with empty membership. An equivalent zero-component early return fixed it;
  focused GCC and Clang anchor/path/replay suites then passed all three tests.
- The canonical command, inputs, `QuickBuild` tree, warmed cache, 32 threads,
  and 5,000-base-voxel interval were unchanged from checkpoint 12. Three runs
  produced:

  | metric | minimum | median | maximum | checkpoint 12 median |
  |---|---:|---:|---:|---:|
  | command wall | 10.92 s | 11.01 s | 11.58 s | 10.37 s |
  | anchor wall | 6.652 s | 6.653 s | 6.797 s | 6.416 s |
  | anchor CPU | 167.38 s | 169.09 s | 169.19 s | 167.49 s |
  | tensor-proposal worker time | 35.04 s | 35.65 s | 35.66 s | about 35.25 s |
  | fiberlet wall | 3.523 s | 3.568 s | 3.947 s | 3.294 s |
  | peak RSS | 2.00 GiB | 2.03 GiB | 2.04 GiB | 2.08 GiB |

- Every run performed 809,364,400 physical proposal visits and avoided the
  same number of materialization visits, reconstructing the prior
  1,618,728,800 logical visits exactly. Populations and DP work remained exact:
  2,603 anchors, 51,782 searched / 26,494 accepted fiberlets, 170,813 sampled
  voxels, and 62,970,689 DP relaxations. Failures remained 2 greedy / 1
  fiberlet.
- All three `fiber_replay.json` files retained SHA-256
  `41fa73c76bc3a20528d064e2baed78552a20bed41542f9ed4e2ddcfb5e739215`.
- The removed pass was cheaper than repeatedly evaluating residual-bin/cutoff
  membership in later billion-visit loops. The implementation, profile version
  16 experiment, helper, and tests were removed; checkpoint 12 remains the
  production baseline and no specification or user-documentation change was
  retained.

## Checkpoint 14: Direct Owned-Cell Initialization Range

- Approved for implementation. The production extractor already knows dense
  tile strides and exact clipped cell bounds; this checkpoint will expose that
  owned cube directly to initialization instead of rediscovering it by scanning
  the larger support range. Robust refinement remains on the unchanged support
  range, and the public vector API retains coordinate validation.
- Independent review clarified that the public API must retain stable input
  order and its historical count-only coverage check, including off-lattice and
  duplicate-plus-missing inputs. It also requires O(1) direct-range structural
  validation, explicit public/direct/avoided-support visit counters, invalid-direction
  denominator coverage, and comparison against fit setup rather than support
  observation-construction time.
- Implemented a validated dense-tile owned-cell layout and row visitor. Public
  fitting still performs the original stable coordinate filter and count-only
  coverage check; production initialization directly traverses the owned cube,
  while all robust refinement remains on the unchanged support range.
- Focused GCC and Clang anchor/path/replay suites passed. Added layout
  containment/cardinality tests, canonical row-order coverage, public shuffled/
  off-lattice/duplicate-plus-missing compatibility, invalid/NaN/unusable
  direction initialization parity, direct visit accounting, partial cells, and
  serial/parallel extraction coverage.

  | metric | baseline min | baseline median | baseline max | direct min | direct median | direct max |
  |---|---:|---:|---:|---:|---:|---:|
  | command wall | 10.70 s | 10.76 s | 11.15 s | 10.37 s | 10.65 s | 10.68 s |
  | total CPU | 237.95 s | 243.52 s | 247.12 s | 233.53 s | 237.40 s | 239.89 s |
  | anchor wall | 6.547 s | 6.553 s | 6.733 s | 6.061 s | 6.304 s | 6.339 s |
  | anchor CPU | 165.37 s | 169.45 s | 172.57 s | 160.03 s | 162.51 s | 164.47 s |
  | fitter setup before layout attribution | 10.336 s | 10.996 s | 11.082 s | 0.073 s | 0.074 s | 0.076 s |
  | fiberlet wall | 3.424 s | 3.470 s | 3.652 s | 3.564 s | 3.594 s | 3.599 s |

- Every direct run reported 833,728 physical owned-initialization visits and
  858,114,544 avoided support visits. All runs retained 2,603 anchors, 51,782
  searched / 26,494 accepted fiberlets, 170,813 sampled voxels, 62,970,689 DP
  relaxations, and 2 greedy / 1 fiberlet failures. All baseline and direct
  artifacts had SHA-256
  `41fa73c76bc3a20528d064e2baed78552a20bed41542f9ed4e2ddcfb5e739215`.
- The small fiberlet-wall increase was outside the changed anchor path and was
  outweighed by lower anchor and total cost. Checkpoint 14 is retained.
- Final review moved constant-time direct-layout validation into the fit-setup
  accounting. A post-change validation run reported 0.092 setup worker-seconds
  and retained the same artifact hash; its unrelated fiberlet stages ran under
  visible external load and were not added to the paired benchmark table.

## Checkpoint 15: Contiguous Peak-Grid Cache

- Approved for implementation. Replace the bounded peak hill-climb's ordered
  map and repeated grid geometry with direct row-major slots while preserving
  the response kernel, traversal and tie-breaking exactly.
- Independent review required exact artifacts/counters, preservation of the
  no-feasible-slot `(0,0)` fallback, separate double physical coordinates and
  float response coordinates, a shared tested internal cache seam, non-finite
  cache coverage, checked unsigned layout arithmetic, and peak-RSS comparison.
  The implementation plan incorporates these requirements.
- Implemented a checked internal row-major grid layout and response cache. Peak
  search precomputes physical points and feasibility once, while response
  coordinates retain their separate float index/step calculation. Computed
  state is independent of the response value, so NaN and infinities are cached.
- Added direct layout coverage at extents 0, 8, and 128; exact boundary and
  out-of-range behavior; non-finite response/cache-hit coverage; and an
  extent-zero production fit with one feasible slot and exact response counts.
- Final review restricted the internal helper to the same maximum extent 128
  enforced by fitter configuration, closing signed-offset overflow outside the
  production domain. Extent 129 is covered as an error.
- The planned no-feasible-slot and explicit tie tests were not added. A
  nonempty fitted cell's finite owned lattice positions enclose its center, so
  at least the center slot is feasible; the historical dormant center fallback
  remains structurally unchanged. Candidate comparison and tie logic were not
  edited and retain the existing behavioral tests. This is an explicit test
  simplification rather than new behavior.
- Focused GCC and Clang `test_fiber_anchors`, `test_fiberlet_paths`, and
  `test_fiber_replay` suites passed. The final extent-zero addition was rerun
  under both compilers.

  | metric | minimum | median | maximum | checkpoint 14 median |
  |---|---:|---:|---:|---:|
  | command wall | 10.37 s | 10.43 s | 10.47 s | 10.65 s |
  | total CPU | 231.46 s | 234.29 s | 235.60 s | 237.40 s |
  | anchor wall | 6.189 s | 6.221 s | 6.251 s | 6.304 s |
  | anchor CPU | 158.02 s | 160.30 s | 161.33 s | 162.51 s |
  | peak-search worker time | 42.757 s | 42.836 s | 43.664 s | about 43.9 s |
  | peak RSS | 2.01 GiB | 2.02 GiB | 2.03 GiB | 2.08 GiB at checkpoint 12 |

- Every run retained 929,769 peak response requests, 335,498 computed grid
  responses, 28,675 uncached acceptance responses, 2,603 anchors, 51,782
  searched / 26,494 accepted fiberlets, 170,813 sampled voxels, 62,970,689 DP
  relaxations, and 2 greedy / 1 fiberlet failures. Every `fiber_replay.json`
  retained SHA-256
  `41fa73c76bc3a20528d064e2baed78552a20bed41542f9ed4e2ddcfb5e739215`.
- Checkpoint 15 is retained.
- Checkpoint 14 did not record RSS separately. The nearest recorded baseline is
  checkpoint 12's 2.08 GiB median; checkpoint 15 did not regress against it.
## Checkpoint 16: split peak response/evidence streams

- User approved the experiment and explicitly reiterated that exact numeric
  identity and fixed accumulation order are not requirements.
- Baseline is committed checkpoint 15 (`f88aea31e`): median command wall
  10.43 s, total CPU 234.29 s, anchor wall 6.221 s, anchor CPU 160.30 s, and
  peak-search worker time 42.84 s across three canonical runs.
- Planned representation keeps one compact hot record per spatially relevant
  observation and moves alignment/gradient fields into a sparse stream indexed
  only by retained usable evidence. This avoids a second Gaussian evaluation
  while reducing the dominant scan's record width.
- Independent review required preserving invalid-gradient observations in the
  evidence stream, defining actual in-cutoff evidence visits separately from
  prepared evidence population, avoiding ABI-dependent legacy record-size
  claims, and applying axis/position distribution plus replay/visual quality
  gates if artifacts change. The implementation plan now includes each item.
- The first implementation used a 20-byte hot record containing its evidence
  index. Initial replay profiling found only 4.82% prepared evidence and 3.25%
  actual evidence visits. The experiment therefore moved indices to a parallel
  four-byte array, leaving a 16-byte hot kernel stream; the index array is only
  consulted after radial rejection.
- Three final 16-byte-stream canonical runs measured:

  | metric | minimum | median | maximum | checkpoint 15 median |
  |---|---:|---:|---:|---:|
  | command wall | 10.15 s | 10.25 s | 10.47 s | 10.43 s |
  | total CPU | 227.84 s | 229.90 s | 231.37 s | 234.29 s |
  | anchor wall | 5.922 s | 6.070 s | 6.222 s | 6.221 s |
  | anchor CPU | 153.83 s | 155.82 s | 157.00 s | 160.30 s |
  | peak-search worker time | 39.905 s | 39.940 s | 40.017 s | 42.836 s |
  | peak RSS | 2.03 GiB | 2.03 GiB | 2.05 GiB | 2.02 GiB |

- Every run prepared 199,261,642 hot records and 9,607,554 evidence records
  (4.82%). The 2,974,011,902 hot response visits loaded 96,698,222 evidence
  records after radial rejection (3.25%). Maximum per-component transient
  storage was 923,104 bytes on this target.
- All runs retained 2,603 anchors, 51,782 searched / 26,494 accepted fiberlets,
  170,813 sampled voxels, 62,970,689 DP relaxations, and 2 greedy / 1 fiberlet
  failures. All replay artifacts retained SHA-256
  `41fa73c76bc3a20528d064e2baed78552a20bed41542f9ed4e2ddcfb5e739215`.
- Checkpoint 16 is retained. Exact numeric identity was not used as a gate; it
  happened to be preserved.
- Focused GCC and Clang QuickBuild suites passed `test_fiber_anchors`,
  `test_fiberlet_paths`, and `test_fiber_replay`. The Clang tree reused the
  existing local libigl and PaStiX sources; no dependency download or install
  was performed. `git diff --check` passed.
- The review-proposed all-denominator/zero-evidence production fixture was not
  added: peak search is entered only for a retained component, so at least one
  retained evidence observation necessarily exists. The existing two-component
  profile fixture instead verifies mixed denominator-only/evidence records and
  confirms evidence with all gradients invalid remains indexed. No public test
  seam was introduced solely to construct an unreachable state.
- Logged a future peak-search option: first measure per-response radial survival,
  per-component unique use, and neighboring-candidate overlap. Use those results
  to choose conservative contiguous-block rejection or demanded-neighbor
  batching. The prior 2D CSR's 59% visit reduction but runtime regression is the
  explicit warning against optimizing visit counts without preserving locality.

## Checkpoint 17: reuse objective Gaussian values

- User approved eliminating duplicate `transverseGaussian()` calls in retained
  spatial objectives and final refined-state evaluation. Exact numeric identity
  is explicitly not required.
- Baseline is committed checkpoint 16 (`7bb2830fd`): median command wall
  10.25 s, total CPU 229.90 s, anchor wall 6.070 s, anchor CPU 155.82 s, and
  peak-search worker time 39.94 s across three canonical runs.
- Planned change retains the active-component Gaussian values already computed
  for each denominator and reuses the assigned value for its numerator. No
  persistent cache, extra traversal, or changed fitting equation is planned.
- Independent review confirmed `transverseGaussian()` is pure and local reuse
  preserves the equations and accumulation order. The plan was corrected to
  compute only active components, require axis/position and visual review if
  hashes change, and use existing phase timers instead of adding a hot-loop
  profile counter. The nearest profile measured roughly 23.44 local-state and
  13.76 final-evaluation worker-seconds.
- The implementation used fixed two-slot stack arrays but evaluated entries
  only for `activeComponents`. Existing GCC anchor/path/replay tests passed.
- Three canonical runs measured:

  | metric | minimum | median | maximum | checkpoint 16 median |
  |---|---:|---:|---:|---:|
  | command wall | 10.18 s | 10.23 s | 10.28 s | 10.25 s |
  | total CPU | 225.62 s | 229.14 s | 229.63 s | 229.90 s |
  | anchor wall | 6.033 s | 6.063 s | 6.069 s | 6.070 s |
  | anchor CPU | 152.41 s | 155.53 s | 155.67 s | 155.82 s |
  | local-state evaluation worker time | 24.242 s | 24.335 s | 24.636 s | about 23.44 s |
  | final-evaluation worker time | 14.035 s | 14.123 s | 14.303 s | about 13.76 s |
  | peak-search worker time | 39.258 s | 39.431 s | 40.044 s | 39.94 s |
  | peak RSS | 2.02 GiB | 2.03 GiB | 2.03 GiB | 2.03 GiB |

- Every run retained 2,603 anchors, 51,782 searched / 26,494 accepted
  fiberlets, 170,813 sampled voxels, 62,970,689 DP relaxations, and 2 greedy /
  1 fiberlet failures. Every replay artifact retained SHA-256
  `41fa73c76bc3a20528d064e2baed78552a20bed41542f9ed4e2ddcfb5e739215`.
- Checkpoint 17 is rejected because both directly targeted worker phases
  regressed despite effectively flat enclosing wall/CPU time. The likely cause
  is that the compiler already eliminates the pure repeated expression where
  profitable, while explicit arrays add register or spill pressure. The source
  change was removed.
- Clang validation, production specification/documentation updates, and a
  changelog entry were intentionally skipped after rejection because no
  production code or schema change remains.

## Checkpoint 18: radial demand and neighbor reuse

- User approved the measurement-led peak-search optimization. Baseline remains
  committed checkpoint 16 (`7bb2830fd`) at median 10.25 s command wall,
  229.90 s total CPU, 6.070 s anchor wall, 155.82 s anchor CPU, and 39.94
  peak-search worker-seconds.
- The first phase will temporarily count exact radial-cutoff passes and unique
  touched records per component. Low unique use selects conservative contiguous
  block rejection; high repeated neighboring use selects demanded-response
  batching. The temporary touch instrumentation will not remain in production.
- Independent review found those aggregate measurements insufficient by
  themselves. The measurement now also simulates conservative contiguous block
  rejection at 16/32/64 records, records actual simultaneously missing cohorts
  for hill-climb/separable/final neighborhoods, and separates cached-grid from
  uncached acceptance responses. Any retained block rule must use outward
  bounds and conservative axis separation; any batching path must explicitly
  collect missing slots, evaluate one cohort, and publish all values before
  preserving the existing comparison order and counters.
- The instrumented canonical replay preserved the exact baseline artifact and
  measured 2,974,011,902 full response-record visits. Grid responses had
  768,213,262 radial passes over 79,309,912 per-component unique records, a
  9.69x repeated-use factor; acceptance responses had 65,642,789 passes over
  55,962,374 unique records.
- Conservative contiguous blocks were not selective enough: 16/32/64-record
  blocks left 2.272B / 2.485B / 2.720B record visits, reductions of 23.6%,
  16.5%, and 8.5% from the full scan. Their maximum metadata was only
  12,384 / 6,192 / 3,096 bytes, but the arithmetic reduction is modest.
- Actual hill-climb missing-response cohorts had theoretical unbatched loads of
  2,739,837,736 records versus 536,179,860 loads when batched, an 80.4%
  reduction in hot-record loads. Separable and final neighborhoods had no
  missing responses on this workload because the hill-climb neighborhoods had
  already populated those cache slots. Demanded-neighbor batching is therefore
  selected; temporary block/touch/cohort profile fields will be removed.
- Full-cohort batching preserved the exact replay artifact but raised
  peak-search worker time to 46.406 s, anchor CPU to 161.71 s, and command wall
  to 10.54 s. Bounded width four reached 47.663 s peak work and 10.55 s wall;
  width two reached 49.156 s and 10.80 s. Baseline peak work is 39.94 s.
- The compact response stream evidently remains cache-resident between scalar
  scans. Interleaving independent candidates adds accumulator working state and
  register/spill pressure without removing the dominant exponential and
  compensated-sum arithmetic. Narrower batches increase repeated stream scans
  while retaining that accumulator overhead.
- All variants retained 2,603 anchors, 51,782 searched / 26,494 accepted
  fiberlets, 170,813 sampled voxels, 62,970,689 DP relaxations, 2 greedy / 1
  fiberlet failures, and replay SHA-256
  `41fa73c76bc3a20528d064e2baed78552a20bed41542f9ed4e2ddcfb5e739215`.
- Checkpoint 18 is rejected. All temporary measurement fields, cache APIs,
  focused tests, accumulator refactoring, and cohort code were removed. The
  16-record block path was not implemented because its best possible simulated
  visit reduction was only 23.6%, with additional metadata and branch cost.
  Three-run final benchmarking and Clang validation were skipped after the
  clear single-run regressions; no production documentation or changelog entry
  is retained.

## Checkpoint 19: plain float tensor accumulation

- User approved replacing compensated tensor accumulation and explicitly
  confirmed that numerical differences are acceptable when final extraction
  quality remains similar.
- The experiment is intentionally limited to the six-entry tensor histograms
  in `robustDirectionProposal()`. Earlier float peak scoring did not affect this
  fitter path. Residual/cutoff arithmetic, centroids, objectives, and final
  evaluation remain unchanged so the measured result identifies this cost.
- Baseline remains committed checkpoint 16 (`7bb2830fd`): median command wall
  10.25 seconds, total CPU 229.90 seconds, anchor wall 6.070 seconds, anchor CPU
  155.82 seconds, and approximately 35.5 local tensor-proposal worker-seconds.
- Independent review approved the narrow scope and corrected the final merge to
  ordinary double: float32 is used only for the repeated per-observation bin
  updates. It also added public-path low-mass/imbalanced/near-degenerate tests
  and concrete matched-anchor axis-angle and position-delta quality metrics.
- The float32 variant passed the existing public anchor/path/replay suites and
  retained 2,603 anchors, 51,782 searched / 26,494 accepted fiberlets, 170,813
  sampled voxels, 62,970,683 DP relaxations, and 2 greedy / 1 fiberlet failures.
  Three runs were deterministic with replay SHA-256
  `c2f251cf47e0b12008060f1ef6c84f0feabf7a86d5caff6b06896e0380d17c40`.
  Relative to compensated output, greedy routes were identical; fiberlet route
  point displacement was p50 0, p95 4.17e-7, maximum 1.37e-6 base voxels.
- Float32 medians were 11.02 seconds command wall, 235.35 seconds total CPU,
  6.449 seconds anchor wall, 158.19 seconds anchor CPU, and 38.26 tensor worker-
  seconds. Machine load varied: an intervening paired compensated run measured
  11.03, 236.88, 6.383, 160.27, and 38.71 seconds, so only a small CPU-side
  signal was present and the casts remained suspect.
- As a measured deviation from the initial float-only plan, ordinary double
  accumulation was tested to isolate compensation from conversion cost. Its
  three-run min/median/max results were command wall 10.21/10.35/10.41 seconds,
  total CPU 228.13/228.36/230.82 seconds, anchor wall 5.994/6.060/6.139 seconds,
  anchor CPU 153.62/154.60/156.02 seconds, and tensor proposal
  35.47/35.68/36.27 worker-seconds. All three artifacts exactly matched the
  compensated SHA-256
  `41fa73c76bc3a20528d064e2baed78552a20bed41542f9ed4e2ddcfb5e739215`.
- A final immediately paired compensated run was faster than the ordinary-
  double median at 10.12 seconds wall, 227.29 seconds total CPU, 5.924 seconds
  anchor wall, 153.51 seconds anchor CPU, and 35.16 tensor worker-seconds.
  Historical three-run baseline medians were likewise 10.25 seconds wall and
  approximately 35.5 tensor worker-seconds. Neither uncompensated variant has a
  repeatable benefit, so all production changes were removed.
- The planned extra near-degenerate public fixtures and Clang rerun were
  skipped after rejection. Existing robust trimming, angular-tail,
  one-/two-component, serial/parallel, path, and replay tests cover the restored
  production path. Production specs, user docs, changelog, and profile schema
  remain unchanged.
- User reported that the machine had competing work during every checkpoint-19
  run. All checkpoint-19 timing comparisons are therefore invalid, including
  the apparent float/double regressions and gains. The partial implementations
  remain removed, but checkpoint 19 is inconclusive rather than rejected.

## Checkpoint 20: scalar-specialized robust proposal

- User approved reimplementing the experiment without the avoidable
  float-to-double-to-float conversions. Production compact observations will
  stay float throughout the robust proposal hot loop; only fixed-size summaries
  cross to the existing double cutoff and eigensolver. The public observation
  API remains double through the same generic implementation.
- Canonical performance measurements are explicitly deferred until the user
  reports that the computer is free. Build and focused correctness validation
  may proceed meanwhile.
- Independent review confirmed the scope but required proposal-local scalar
  state, scalar residual binning, histogram-derived cutoff mass, actual compact
  extraction fixtures, explicit float-sensitive boundaries, and concrete
  aggregate quality metrics. The implementation plan incorporates these
  corrections.
- Implemented scalar-generic proposal helpers. Production compact observations
  now use float positions, directions, component/pivot copies, Gaussian and
  assignment arithmetic, residual histograms, and plain tensor bins throughout
  the per-observation proposal loop. Only the two fixed 256-bin histograms and
  six retained tensor entries are converted to double for the existing cutoff
  and eigensolver.
- The public double fitter instantiates the same shared proposal but retains its
  existing normalized directions, observation-order total-mass sum,
  compensated per-bin tensors, compensated final bin merge, double cutoff, and
  double eigensolver. Centroid, spatial objective, peak, final evaluation,
  persistent state, and output code are unchanged in both paths.
- Added extraction-level coverage that actually traverses the private compact
  storage. A stable two-direction fixture matches the double fitter within
  axis cosine `1e-5` and 0.05 prediction voxels; the observed position delta was
  0.019951 prediction voxels. A three-direction near-degenerate fixture at the
  exact presence floor verifies deterministic retention/removal and finite
  normalized retained axes.
- The GCC `QuickBuild` rebuilt `vc_fiberlets` and all three focused tests;
  `test_fiber_anchors`, `test_fiberlet_paths`, and `test_fiber_replay` passed in
  0.63 seconds. The production `FiberAnchors.cpp` compile command also passed
  under Clang; a separate linked Clang test tree is not currently configured
  and remains required before retention. `git diff --check` passes.
- After the user confirmed the machine was free, three optimized and three
  compensated-double runs were alternated on the canonical 5,000-base-voxel
  replay with 32 threads. Optimized medians were 9.58 seconds command wall,
  5.461 seconds anchor wall, 140.75 seconds anchor CPU, and 23.736 tensor-
  proposal worker-seconds. Baseline medians were 9.65, 5.504, 143.16, and
  25.632 seconds respectively. The float proposal therefore reduced its own
  worker time by 7.4%, anchor wall by 0.8%, anchor CPU by 1.7%, and command wall
  by 0.7%. Median peak RSS changed from 2,117,268 to 2,123,056 KiB (+0.3%).
- All three optimized artifacts had SHA-256
  `9ad06d494b886dc4e256e1adadc3cb12e70fee051c3292895a4593a475efa472`;
  all three baseline artifacts reproduced the historical SHA-256
  `41fa73c76bc3a20528d064e2baed78552a20bed41542f9ed4e2ddcfb5e739215`.
  Both variants retained 2,603 anchors, 2,560 graph nodes, 26,494 graph edges,
  and 2 greedy / 1 fiberlet failures. Their 352 emitted fiberlet route points
  differed by at most 1.3764e-6 base voxels (mean 5.63e-8). The optimized
  candidate was retained and committed as `397c1cbf3`.

## Checkpoint 21: compact-float spatial objectives

- User approved extending the successful compact float boundary to the
  fixed-direction spatial objective scans. The checkpoint intentionally leaves
  final evaluation and all persistent/acceptance state in double so its target
  is the approximately 22.3 local-state-evaluation worker-seconds measured in
  checkpoint 20.
- Baseline is committed checkpoint 20 (`397c1cbf3`): median command wall 9.58
  seconds, anchor wall 5.461 seconds, anchor CPU 140.75 seconds, and unchanged
  2,603 anchors / 2,560 graph nodes / 26,494 edges / 2 greedy and 1 fiberlet
  failures.
- Independent review approved the narrow scope but required explicit all-site
  denominator semantics, ordinary-float accumulator selection, one fused paired
  scan without checkpoint-17 Gaussian reuse, large-coordinate/cutoff fixtures,
  stronger branch-quality comparisons, and complete benchmark metadata. The
  implementation and validation plan incorporates these corrections. A linked
  Clang test tree is not currently configured; the touched translation unit
  will be compiled with Clang and macOS/arm64 CI remains the portability gate.
- Implemented scalar-specialized single and paired spatial objectives. The
  public expanded path retains its former double code and compensated sums.
  The compact path uses ordinary float numerators/denominators and existing
  compact scalar Gaussian/direction helpers, with every finite-position site
  entering each denominator before numerator eligibility is tested. The paired
  path remains one fused scan and final evaluation is unchanged.
- Added an extraction-level fixture near prediction coordinate 20,000 with
  invalid-direction, NaN-presence, and below-floor denominator-only samples.
  It checks deterministic compact extraction/backtracking and bounded geometry
  against the unchanged expanded double fitter. All 72 anchor cases and the
  focused anchor/path/replay CTest suites passed; the touched production unit
  compiled with Clang 22.1.8 and `git diff --check` passed.
- Three optimized and three checkpoint-20 baseline runs were alternated on the
  canonical replay. Optimized medians were 9.59 seconds command wall, 216.80
  seconds total CPU, 5.525 seconds anchor wall, 145.48 seconds anchor CPU,
  20.03 local-state-evaluation worker-seconds, and 31.32 tensor-proposal
  worker-seconds. Baseline medians were 9.44, 211.40, 5.385, 140.07, 22.48,
  and 23.47 seconds respectively.
- The intended objective kernel improved by 10.9%, but tensor proposal
  regressed by 33.4%, anchor CPU by 3.9%, total CPU by 2.6%, and command wall
  by 1.6%. All six replay artifacts exactly matched checkpoint 20 SHA-256
  `9ad06d494b886dc4e256e1adadc3cb12e70fee051c3292895a4593a475efa472`;
  populations and 2 greedy / 1 fiberlet failures were unchanged. The repeatable
  cross-phase regression is attributed to code generation/instruction locality,
  not numerical quality.
- Removed the production specialization, its temporary benchmark switch, and
  checkpoint-only fixture. Production source/tests are identical to committed
  checkpoint 20. Specifications, user documentation, changelog, and profile
  schema remain unchanged.

## Checkpoint 22: isolated compact-float spatial objectives

- User approved retesting checkpoint 21's faster compact objective kernel with
  code-generation isolation. Checkpoint 21 reduced local-state evaluation by
  10.9%, but placing the specialization in `FiberAnchors.cpp` increased
  unrelated tensor-proposal work by 33.4% and regressed total runtime.
- The planned private module owns the complete objective equation and provides
  expanded-double and indexed-compact-float entry points. Production storage is
  borrowed through spans, and the main fitter performs only small fixed-state
  conversion and dispatch. This avoids implementation copying and a hot-path
  benchmark branch.
- The current checkpoint-20 `libvc_fiber_tracer.so` will be preserved before
  rebuilding. Alternating runs will select that library or the rebuilt library
  through the dynamic loader, keeping both measured implementations branch-free.
- Independent review approved translation-unit isolation after requiring the
  exact logical-index/underlying-tile-index mapping, shared source-private type
  extraction, strict all-site denominator order, exact public-double behavior,
  direct private-module edge coverage, a rebuilt and loader-verified baseline,
  complete benchmark metadata, and linked Clang validation. The plan now
  includes each correction.
- Rebuilt checkpoint 20 before source edits and preserved
  `libvc_fiber_tracer.so` SHA-256
  `9dfecc2166c185634cf4a8ed693af9a5c1a5883e07a23608448b26817d40db26`.
  The isolated library SHA-256 was
  `87a47e9c5f73c15a8f39dc09a049a2f85fbbce6db165e4d20d0e92620550d8e4`.
  `ldd` verified the baseline loader path used the preserved library and the
  default path used the rebuilt workspace library.
- Implemented `FiberAnchorObjectives.cpp` plus a source-private shared header.
  The module owns the common objective equation and instantiates expanded
  compensated-double and indexed compact-float paths. Compact logical indices
  select assignment/membership entries independently from the underlying tile
  indices. Invalid cardinalities and underlying indices fail before evaluation.
- Added direct private-module coverage for nonconsecutive/repeated tile indices,
  exact expanded single/paired parity, compact single/paired parity, compact-
  versus-expanded tolerance, empty and zero-component inputs, invalid mappings,
  denominator-only invalid/NaN/below-floor evidence, realistic coordinates near
  20,000, and exact/adjacent cutoff positions.
- GCC and Clang 22.1.8 QuickBuild linked suites passed `test_fiber_anchors`,
  `test_fiberlet_paths`, and `test_fiber_replay`. The Clang tree reused local
  libigl and PaStiX sources; no install or download was performed. Its initial
  `/tmp` run failed only because the machine's temp quota prevented test output;
  rerunning with workspace `TMPDIR` passed all three suites. `git diff --check`
  passed.
- Ran alternating order isolated-1, baseline-1, isolated-2, baseline-2,
  isolated-3, baseline-3 with the canonical 32-thread, 5,000-base-voxel replay
  on the Paris4 `fiber_s1_002` prediction manifest, fiber
  `dj_20260805T025256484_000003.json`, and `las_008` normal manifest. Both
  variants used the same QuickBuild executable, warm local inputs, no explicit
  warmup, and an explicitly verified dynamic-library path.
- Controlled min/median/max results were:

  | metric | isolated | checkpoint 20 | median change |
  |---|---:|---:|---:|
  | command wall | 9.08 / 9.17 / 9.26 s | 9.54 / 9.56 / 9.58 s | -4.1% |
  | total CPU | 201.42 / 201.84 / 203.46 s | 211.74 / 212.93 / 213.27 s | -5.2% |
  | anchor wall | 4.998 / 5.050 / 5.179 s | 5.421 / 5.425 / 5.458 s | -6.9% |
  | anchor CPU | 130.40 / 130.74 / 131.34 s | 140.28 / 140.80 / 141.24 s | -7.1% |
  | local-state objective work | 13.77 / 13.86 / 13.91 s | 22.53 / 22.59 / 22.61 s | -38.6% |
  | tensor-proposal work | 22.66 / 22.78 / 22.89 s | 23.71 / 23.72 / 23.81 s | -4.0% |
  | peak RSS | 2,084,764 / 2,115,560 / 2,116,312 KiB | 2,109,644 / 2,112,200 / 2,122,024 KiB | +0.2% |

- Every run retained 2,603 anchors, 2,560 graph nodes, 26,494 graph edges,
  62,970,388 DP relaxations, accepted all 12,275 position candidates at depth
  zero, and produced 2 greedy / 1 fiberlet failures. All six artifacts retained
  SHA-256 `9ad06d494b886dc4e256e1adadc3cb12e70fee051c3292895a4593a475efa472`.
  Checkpoint 22 is retained; no profile-schema change was needed.

## Checkpoint 23: float final anchor evaluation

- User rejected treating compensated double arithmetic as intrinsically
  required for final support evaluation. The production compact path will stay
  float32 through observation access, Gaussian/direction math, and accumulators,
  widening only the fixed-size summary at the persistent output boundary.
- Numerical identity is explicitly not an acceptance requirement. The gates are
  deterministic repeatability, stable support/acceptance decisions, comparable
  anchor geometry and downstream replay quality, and measured enclosing speed.
- The expanded/public final-evaluation dispatch will exercise the same float32
  equation as production rather than retaining a compensated-double special
  case. Its source doubles narrow on access and only the fixed summary widens.
- The first three-pair layout improved median final-evaluation work from 13.68
  to 12.35 worker-seconds (-9.7%), but co-locating it with checkpoint 22's
  objective kernels regressed local objective work from 14.03 to 20.09 seconds
  (+43.2%). Median anchor CPU rose from 131.87 to 136.90 seconds and command
  wall from 9.14 to 9.40 seconds. All six artifacts remained byte-identical
  with unchanged populations, DP work, and failures.
- This layout is rejected. The shared low-level equation will be extracted to a
  private header and final evaluation moved to a separate translation unit so
  it cannot perturb objective code generation. The isolated layout will be
  rebuilt and measured as the actual checkpoint candidate.
- Independent review approved strict isolation and required fresh measurements,
  checked public narrowing, scale-safe expanded normalization, float-derived
  support/coherence/objective ratios, denominator/count edge coverage, and GCC
  plus linked Clang validation. The implementation and tests include each item.
- The attempted shared-helper extraction still left local objective work at
  20.89 worker-seconds. The final correction restored
  `FiberAnchorObjectives.cpp` byte-for-byte to checkpoint 22 and extracted the
  pre-existing final-reduction behavior into its own private implementation.
  A screening run restored objective/tensor work to 13.69/22.54 seconds.
- Three fresh pairs alternated candidate-1, baseline-1, candidate-2,
  baseline-2, candidate-3, baseline-3. Both used the same QuickBuild executable,
  warm local Paris4 inputs, 32 threads, a 5,000-base-voxel replay, and explicitly
  loader-verified libraries (candidate SHA-256
  `239b1ebdbae79f8e0df2a5dcdc6a05286d21de61f0f74d66257c31023e91bf90`; baseline
  SHA-256 `87a47e9c5f73c15a8f39dc09a049a2f85fbbce6db165e4d20d0e92620550d8e4`).
- Controlled min/median/max results were:

  | metric | float final | checkpoint 22 | median change |
  |---|---:|---:|---:|
  | command wall | 9.02 / 9.22 / 9.32 s | 9.11 / 9.26 / 9.43 s | -0.4% |
  | total CPU | 199.84 / 202.89 / 203.88 s | 202.81 / 204.33 / 206.03 s | -0.7% |
  | anchor wall | 4.888 / 5.043 / 5.134 s | 5.016 / 5.103 / 5.212 s | -1.2% |
  | anchor CPU | 128.26 / 130.55 / 131.52 s | 131.27 / 132.40 / 133.30 s | -1.4% |
  | final evaluation | 12.73 / 13.11 / 13.12 s | 13.56 / 13.79 / 13.84 s | -4.9% |
  | local objective | 13.71 / 14.10 / 14.14 s | 13.94 / 14.05 / 14.20 s | +0.4% |
  | tensor proposal | 22.45 / 23.08 / 23.09 s | 22.79 / 23.09 / 23.41 s | -0.0% |
  | peak RSS | 2,121,456 / 2,121,656 / 2,158,252 KiB | 2,107,536 / 2,116,432 / 2,131,040 KiB | +0.2% |

- Every run retained 2,603 anchors, 2,560 graph nodes, 26,494 edges,
  62,970,388 DP relaxations, and 2 greedy / 1 fiberlet failures. All artifacts
  retained SHA-256
  `9ad06d494b886dc4e256e1adadc3cb12e70fee051c3292895a4593a475efa472`.
  Direct tests cover compact/public support and coherence agreement, threshold
  equality and adjacent decisions, checked nonrepresentable inputs, and exact
  assigned-count/denominator semantics. Checkpoint 23 is retained.

## Checkpoint 24: end-to-end float anchor and fiberlet state

- Committed checkpoint 23 as `07176ccd6` before beginning this experiment.
- Converted anchor observations, solver configuration, robust/refinement and
  peak state, retained anchors, diagnostics, artifact fields, fiberlet geometry,
  sampled scoring state, path/graph costs, DP state, candidates, and graph data
  to float32. Integer lattice, index, count, and flag state remains unchanged.
- Kept only timing/process accounting, cold external scale parsing, and
  reference/replay calculations double. Shared prediction and normal samplers
  remain external double-valued APIs, but their results are checked and narrowed
  once; anchor tile reuse and all extraction state retain only float32 values.
- Generalized the shared tensor implementation while preserving the existing
  named double API. Explicit `canonicalFiberAxisF`, `fiberAxisTensorF`,
  `principalFiberAxisF`, and `principalFiberAxisClosedFormF` entry points avoid
  making existing brace-initialized double calls ambiguous.
- Replaced ineffective inherited double-scale tolerances with deliberate float
  convergence, matrix, geometry, and response-comparison tolerances. Graph
  construction now rejects prediction-scale underflow and base-coordinate
  overflow. Version-1 and version-2 JSON remain readable with checked float
  representability and no schema change.
- GCC and Clang QuickBuild production/focused targets compile. On both builds,
  `test_fiber_anchors`, `test_fiberlet_paths`, and `test_fiber_replay` pass.
  Focused tests cover float field types, extreme values, float principal-axis
  uniqueness/ambiguity, graph underflow/overflow, artifact round trips, legacy
  version loading, and out-of-range rejection. `git diff --check` passes.
- The canonical 32-thread, 5,000-base-voxel replay is intentionally deferred
  because the user reports the CPUs busy. No performance, population, geometry,
  DP-work, replay-quality, or retention conclusion has been recorded yet.
- At user request, one canonical replay was subsequently run for quality only;
  its timing is invalid because the CPUs remained busy. Against checkpoint 23's
  retained replay SHA-256
  `9ad06d494b886dc4e256e1adadc3cb12e70fee051c3292895a4593a475efa472`,
  the float run retained 2,603 anchors and the same 2 greedy / 1 fiberlet
  failures. Greedy replay output was exactly identical. Graph nodes changed
  2,560 to 2,562, accepted edges 26,494 to 26,445, and DP relaxations
  62,970,388 to 62,873,000. The first fiberlet segment retained the same 238
  route points and failed only 0.00645 base voxels later in reference arc; its
  pointwise displacement had median 0.0044, p95 5.43, and maximum 14.54 base
  voxels before reconverging to an evaluator point about 0.0066 base voxels
  from baseline. The final segment had median 0.0039, p95 0.0169, and maximum
  0.0316 base-voxel displacement and reached the identical reference end.
  This is comparable replay behavior, not numeric or topology identity. Three
  controlled runs are still required before any performance/retention decision.
- Post-replay review restored concrete public principal-axis result structs,
  rejected prediction-grid extents above `2^24`, and added checked float32
  scaling for serialized and OBJ base coordinates. GCC and Clang builds pass
  `test_fiber_anchors`, `test_fiberlet_paths`, and `test_fiber_replay` with
  focused overflow and oversized-grid coverage. These are cold validation and
  output-path guards and do not alter the canonical valid-grid replay result.
- After the user confirmed CPU availability, three measured checkpoint-24
  replays used the same QuickBuild executable, warm local Paris4 inputs,
  32 threads, and a 5,000-base-voxel reference interval. Results against the
  controlled checkpoint-23 measurements were:

  | metric | checkpoint 24 min / median / max | checkpoint 23 min / median / max | median change |
  |---|---:|---:|---:|
  | command wall | 8.86 / 8.96 / 8.98 s | 9.02 / 9.22 / 9.32 s | -2.8% |
  | total CPU | 197.05 / 199.47 / 200.68 s | 199.84 / 202.89 / 203.88 s | -1.7% |
  | anchor wall | 4.963 / 4.984 / 5.032 s | 4.888 / 5.043 / 5.134 s | -1.2% |
  | anchor CPU | 128.80 / 129.92 / 130.80 s | 128.26 / 130.55 / 131.52 s | -0.5% |
  | fiberlet wall | 3.236 / 3.298 / 3.313 s | not recorded | n/a |
  | fiberlet CPU | 66.01 / 67.33 / 67.68 s | not recorded | n/a |
  | final evaluation work | 11.99 / 12.25 / 12.34 s | 12.73 / 13.11 / 13.12 s | -6.5% |
  | peak RSS | 1,994,740 / 2,007,020 / 2,018,200 KiB | 2,107,536 / 2,121,656 / 2,158,252 KiB | -5.4% |

- Every checkpoint-24 run retained 2,603 anchors, 2,562 graph nodes, 26,445
  edges, 62,873,000 DP relaxations, and 2 greedy / 1 fiberlet failures. All
  three artifacts and the earlier quality-only run had SHA-256
  `f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`.
  Together with the recorded localized route comparison, this satisfies the
  deterministic performance and comparable-quality gates. Checkpoint 24 is
  retained.
- Recorded the post-checkpoint-24 queue as checkpoints 25-30: parallel corner
  finalization, sparse paged corner deduplication, ready-cell anchor scheduling,
  bounded peak Gaussian acceleration, one-pass membership reuse, and remaining
  DP scheduling/vectorization. The plan explicitly excludes unchanged retries
  of previously rejected response batching, inline membership, and eager-edge
  variants.
- Checkpoint 25 moved worker-local corner-set conversion and sorting onto the
  configured bounded workers while preserving the existing deterministic merge
  tree. Exact vector capacities are reserved on the calling thread before the
  parallel phase, and peak-memory accounting now includes simultaneously live
  merge inputs and outputs. Focused empty, overlap, duplicate-heavy, and uneven
  set tests compare the production finalizer with a serial reference. GCC and
  Clang each pass all 49 `test_fiberlet_paths` cases.
- One cold repository-local replay was excluded because it incurred 1,670 major
  faults and 773,656 filesystem-input blocks. Three subsequent warm runs had
  command wall 8.08 / 8.23 / 8.35 seconds, total CPU 201.66 / 202.91 / 205.02
  seconds, corner-finalization wall 0.265 / 0.268 / 0.268 seconds, and peak RSS
  2,060,168 / 2,060,332 / 2,077,188 KiB. Checkpoint 24 medians were 8.96 wall,
  199.47 CPU, about 1.05 corner-finalization wall, and 2,007,020 KiB RSS.
- Every checkpoint-25 warm run retained 170,778 sampled voxels, 2,603 anchors,
  2,562 graph nodes, 26,445 edges, 62,873,000 DP relaxations, 2 greedy / 1
  fiberlet failures, and exact artifact SHA-256
  `f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`.
  Checkpoint 25 is retained for its 8.1% median wall-time gain; its 1.7% CPU and
  2.7% RSS increases are recorded tradeoffs.

## Checkpoint 26: sparse paged corner bitmap

- Replaced the 32 worker-local interpolation-corner `unordered_set`s with
  sparse `16^3` pages containing 4,096 occupancy bits each. The page edge
  reuses the accepted scoring-index geometry, which previously measured only
  199 occupied pages for the canonical 170,778 sampled voxels.
- Added an immediate-page fast path and eight-entry page-pointer cache per
  worker. Across the three retained runs, about 361.61 million insertions hit
  the immediate page, 44.04 million hit the small cache, and only 75 thousand
  reached the page directory.
- Worker-local page populations varied slightly with scheduling: about 6,017
  pages and 4.645 million worker-local unique voxels. Bitwise OR reduced these
  to 199 merged pages and exactly 170,778 global voxels. Set-bit enumeration is
  followed by the established stored-coordinate sort because page-major order
  alone is not global Z/Y/X order.
- GCC passed 49 `test_fiberlet_paths`, 78 `test_fiber_anchors`, and 6
  `test_fiber_replay` cases. The repository-local Clang QuickBuild passed all
  49 focused path cases. The serial-reference finalization fixture covers
  empty, overlapping, duplicate-heavy, uneven, page-boundary, and signed
  synthetic coordinates.
- The canonical command used `build/bin/vc_fiberlets fiberlet-replay`, Paris4
  `fiber_s1_002.lasagna.json`, fiber
  `dj_20260805T025256484_000003.json`, `las_008` normals, 32 threads, and length
  5000. Outputs are under `volume-cartographer/build/benchmarks/checkpoint26`.
  Host process load was checked before and after every retained timing run and
  stayed below the two-core exclusion threshold.
- Three warm runs produced:

  | metric | checkpoint 26 min / median / max | checkpoint 25 min / median / max | median change |
  |---|---:|---:|---:|
  | command wall | 7.65 / 7.77 / 7.84 s | 8.08 / 8.23 / 8.35 s | -5.6% |
  | total CPU | 195.97 / 196.18 / 197.57 s | 201.66 / 202.91 / 205.02 s | -3.3% |
  | anchor wall | 4.942 / 5.067 / 5.102 s | 5.011 / 5.124 / 5.236 s | -1.1% |
  | anchor CPU | 129.51 / 129.78 / 130.80 s | 130.65 / 131.49 / 133.02 s | -1.3% |
  | fiberlet wall | 2.151 / 2.165 / 2.181 s | 2.488 / 2.519 / 2.534 s | -14.0% |
  | fiberlet CPU | 64.09 / 64.59 / 64.68 s | 68.26 / 69.31 / 69.89 s | -6.8% |
  | corner finalization wall | 0.0190 / 0.0196 / 0.0202 s | 0.265 / 0.268 / 0.268 s | -92.7% |
  | peak RSS | 1,692,124 / 1,697,516 / 1,715,028 KiB | 2,060,168 / 2,060,332 / 2,077,188 KiB | -17.6% |

- Bitmap collection itself costs about 8.23 worker-seconds and remains a
  preparation hot path; the retained gain comes from eliminating worker-vector
  conversion/merge storage and reducing finalization to about 20 milliseconds.
- Every run retained 2,603 anchors, 2,562 graph nodes, 26,445 edges,
  62,873,000 DP relaxations, 2 greedy / 1 fiberlet failures, exact sampled-voxel
  order, and artifact SHA-256
  `f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`.
  Checkpoint 26 is retained.

## Checkpoint 27: cooperative ready-cell anchor scheduling

- Added schema-19 p50/p95/maximum timings for complete group jobs, tile
  preparation, and individual cell processing. The unchanged scheduler's
  measurement run showed a 4.027-second maximum group job against a
  28.78-millisecond maximum cell, confirming group-level load imbalance.
- Kept sampling groups as deterministic reuse and memory-ownership units.
  Prepared tile cells enter a shared cooperative queue; owners retain immutable
  tile observations and help drain work until every dependent cell completes.
  No additional sampled group or tile is retained.
- Rejected over-budget tile pairing, closing the reviewed case where each tile
  fit independently but the paired staging peak exceeded
  `maximumConcurrentSampleBytes`.
- Three warm canonical runs under
  `volume-cartographer/build/benchmarks/checkpoint27` measured command wall at
  6.97 / 6.97 / 6.99 seconds, total CPU at 193.91 / 194.12 / 194.37 seconds,
  anchor wall at 4.251 / 4.262 / 4.264 seconds, anchor CPU at
  126.97 / 126.98 / 127.42 seconds, and peak RSS at
  1,684,328 / 1,687,504 / 1,709,368 KiB. Host process checks before and after
  each run found no competing load above the two-core exclusion threshold.
- Against checkpoint 26 medians, total wall improved 10.3%, anchor wall 15.9%,
  total CPU 1.1%, and anchor CPU 2.2%. Median maximum group-job duration fell
  from the measured 4.027 seconds to 1.647 seconds.
- Every retained run produced exact replay SHA-256
  `f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`
  and exact fiberlet-route SHA-256
  `1ec7df7b8d2417ddc762652be3bf0057eef8b93a329a24d36f02a8837465014b`,
  with 2,603 anchors, 2,562 graph nodes, 26,445 edges, 62,873,000 DP
  relaxations, and 2 greedy / 1 fiberlet failures.
- Independent plan review identified ownership, deadlock, and memory-accounting
  risks in retaining several fully prepared sampling groups. The retained
  implementation deliberately deviates from that detail: it publishes cells
  only for the current tile and makes the owner wait while participating in the
  same work-conserving queue. This keeps the previous tile/group memory
  lifetime and needs no second worker pool or bounded producer queue.
- The review requested additional active-worker occupancy and queue-depth
  instrumentation. That was not retained: complete group-job, tile-preparation,
  and cell-processing tails were sufficient to identify the imbalance, and the
  measured wall-time reduction plus exact artifacts validate the scheduler.
  No queue-depth or occupancy field is silently claimed by schema 19.
- Validation passed GCC `test_fiber_anchors` (78), `test_fiberlet_paths` (49),
  and `test_fiber_replay` (6), plus repository-local Clang
  `test_fiber_anchors` (78). `git diff --check` passed.

## Checkpoint 28: extraction-wide raw prediction reuse

- Checkpoint-27 profiling leaves 26,741,712 sampler submissions for
  39,701,808 tile occurrences, although the exact tile-box union contains only
  6,162,456 prediction voxels. Pair-local overlap reuse therefore removes less
  than half of the avoidable repeated sampling.
- The planned representation is a sorted sparse set of `(z,y)` rows with
  merged X intervals and one contiguous float32 sample array. It stores only
  exact union voxels and supports contiguous tile copies without a hot
  per-voxel hash lookup.
- The first experiment deliberately separates shared sampling from tile
  preparation/fitting. This may lose useful pipeline overlap, so reduced
  submitted samples alone is not a retention result; end-to-end wall, CPU, and
  RSS remain the decision gates.
- Independent review rejected a whole-extraction residency requirement because
  the existing implementation streams workloads larger than the memory cap.
  The implementation will instead use one bounded partitioning algorithm,
  with separate sampling- and fitting-phase admission, deterministic batch
  failures, and tile-owned cooperative fitting. The review also required a
  profile-version bump, explicit ownership/termination, structured checked
  keys, and broader budget/boundary/failure coverage; all are incorporated in
  the revised plan.
- Implemented schema-20 bounded exact-union partitions. Each partition merges
  tile X ranges by structured `(z,y)` row, samples the contiguous union in
  deterministic bounded batches, joins the sampling phase, and then lets
  tile owners copy complete rows before publishing cells to the cooperative
  fitting queue. Sampling and fitting worker counts are admitted separately.
- Focused tests now cover exact three-tile deduplication, low-budget multi-
  partition streaming with identical serialized results, wrong sampler result
  sizes, and earliest-batch error selection under reversed completion. GCC
  passes 82 anchor, 49 path, and 6 replay cases; the repository-local Clang
  build passes all 82 anchor cases.
- One warm screening replay produced the exact checkpoint-27 artifact SHA-256
  `f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`.
  It submitted exactly 6,162,456 union voxels instead of 26,741,712, measured
  4.063 anchor wall / 111.16 anchor CPU seconds and 6.82 command wall seconds,
  with 1,673,116 KiB peak RSS. This is screening evidence only; three retained
  runs remain required.
- Hardened the cooperative queue after review: its complete task capacity is
  reserved before workers start, completion is published before fallible timing
  storage, and a timing-allocation error cannot overwrite an earlier cell error.
  Sampling failures are attached to every affected partition cell, while the
  final canonical cell scan preserves failure precedence across partitions.
- Three warm canonical QuickBuild runs used
  `/usr/bin/time -v volume-cartographer/build/bin/vc_fiberlets fiberlet-replay
  /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json
  /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fibers/david/Paris4_fibers/dj_20260805T025256484_000003.json
  volume-cartographer/build/benchmarks/checkpoint28/runN --normal-manifest
  /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json
  --threads 32 --length 5000`.
  Host checks before and after each run found no unrelated process above the
  two-core exclusion threshold.
- Checkpoint-28 min / median / max results were command wall
  6.80 / 6.82 / 6.84 seconds, total CPU 178.10 / 178.82 / 179.57 seconds,
  anchor wall 4.067 / 4.069 / 4.084 seconds, anchor CPU
  111.20 / 111.61 / 112.04 seconds, and peak RSS
  1,664,980 / 1,675,944 / 1,683,836 KiB. Shared sampling wall was
  0.225 / 0.228 / 0.232 seconds, shared-sampling CPU
  6.11 / 6.25 / 6.41 seconds, prediction-sampling worker time
  6.05 / 6.23 / 6.31 seconds, and tile-copy worker time
  2.45 / 2.47 / 2.53 seconds.
- Against checkpoint-27 medians, command wall improved 2.2%, total CPU 7.9%,
  anchor wall 4.5%, and anchor CPU 12.1%; median peak RSS improved 0.7%.
  Every run used one partition, submitted the exact 6,162,456-voxel union
  instead of 26,741,712 voxels (-77.0%), reused 33,539,352 tile occurrences,
  and reported 150,686,048 maximum shared bytes and 902,563,752 maximum
  accounted live bytes.
- Every retained run produced exact replay SHA-256
  `f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`,
  2,603 anchors, 2,562 graph nodes, 26,445 edges, 62,873,000 DP relaxations,
  and 2 greedy / 1 fiberlet failures. Checkpoint 28 is retained.
- Final validation passed GCC `test_fiber_anchors` (83),
  `test_fiberlet_paths` (49), and `test_fiber_replay` (6), plus repository-local
  Clang `test_fiber_anchors` (83). `git diff --check` passed.

## Checkpoint 29: peak Gaussian acceleration

- Inspected `findDirectionConditionedLocalPeak()`. The axial Gaussian is already
  computed once during response-record preparation. The remaining target is
  solely the transverse `expf` inside every in-cutoff `responseAt()` visit;
  checkpoint-24 profiling attributes roughly 33.75 worker-seconds and 2.97
  billion logical visits to the enclosing peak scan.
- Planned a 512-interval, 2 KiB process-wide float table over normalized
  exponents `[0,8]`, with linear interpolation and the library calculation as
  fallback outside the bounded ordinary domain. The default cutoff's maximum
  exponent is 4.5. Cutoff handling remains in the caller, and no response
  traversal, accumulator, cache, hill-climb, or tie behavior changes.
- Independent review corrected the target count: the 2.974 billion profile
  value counts all response-record visits before radial rejection. Checkpoint-18
  instrumentation measured 768.2 million in-cutoff grid passes and 65.6 million
  in-cutoff acceptance passes, or about 833.9 million actual transverse
  exponentials. The 33.75 worker-seconds is the complete peak-search phase, not
  isolated exponential time.
- The review also required concrete error/geometry gates, external comparison
  instead of a second production runtime path, alternating baseline/candidate
  QuickBuild runs, fixed-seed boundary tests, and an inline lookup with the
  513-entry immutable table acquired outside the hot loop. The plan now uses
  serialized discrete/separable/joint diagnostics for direct peak comparison
  and records changed discrete peaks separately.
- Implemented and directly tested the 513-entry lookup. Across 16,777,217
  evenly spaced exponents in `[0,8]`, maximum/mean absolute error was
  `3.0279e-5`/`2.5418e-6` and maximum/mean relative error was
  `3.0651e-5`/`2.0344e-5`. GCC and Clang each passed 85 anchor cases.
- The lookup screening run measured 6.85 seconds command wall, 4.095 seconds
  anchor wall, 111.76 seconds anchor CPU, and 31.43 peak-search worker-seconds.
  A subsequent direct alternating pair measured:

  | metric | checkpoint 28 | lookup |
  |---|---:|---:|
  | command wall | 6.84 s | 6.84 s |
  | anchor wall | 4.087 s | 4.087 s |
  | anchor CPU | 111.45 s | 111.33 s |
  | peak-search work | 31.11 s | 31.43 s |
  | peak RSS | 1,684,644 KiB | 1,663,816 KiB |

  Both lookup runs were deterministic with replay SHA-256
  `e5df94aa0280b7f5401819e929bbf4df22e2434d79722a26bef2e2f4a9f7d4eb`,
  unchanged populations and 2 greedy / 1 fiberlet failures. Greedy routes were
  exact; fiberlet route displacement was p50/p95 zero and maximum 0.0081 base
  voxels. With no enclosing gain and slightly worse target work, the lookup was
  removed without running three pairs.
- Tested an arithmetic-only degree-six polynomial after `ln(2)` range reduction.
  Across the same dense exponent set, maximum/mean absolute error was
  `1.4067e-5`/`3.0679e-7` and maximum/mean relative error was
  `2.8133e-5`/`3.1395e-6`. It reproduced the exact checkpoint-28 replay SHA-256
  `f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`
  and exact greedy/fiberlet routes, but regressed to 35.54 peak-search worker-
  seconds, 4.204 anchor wall / 114.62 anchor CPU seconds, and 6.97 command wall
  seconds. It was removed after this clear screening rejection.
- Final production source matches checkpoint 28. GCC passes 83 anchor, 49 path,
  and 6 replay cases; repository-local Clang passes all 83 anchor cases.
  Production specifications, user documentation, and changelog are unchanged.

## Checkpoint 30: terminal membership reuse

- Reviewed the terminal robust-membership flow independently before editing.
  Current production computes `M(S_n)`, derives accepted geometry `S_(n+1)`,
  then computes a membership-only `M(S_(n+1))` for peak evidence attribution
  and final support. The experiment removed only that terminal refresh; later
  outer iterations still began with a fresh proposal.
- The review corrected two planning errors: the reused membership belongs to
  the geometry at the start of the terminal iteration rather than the accepted
  final geometry, and it controls peak evidence attribution rather than
  geometric peak ownership. It also established that the refresh makes two
  complete observation scans and contributes to aggregate robust mass/outlier
  counters.
- Implemented the isolated removal without a runtime selector. Focused tests
  asserted two observation scans per attempted outer iteration, forced a
  two-iteration off-center fit, and checked deterministic repeated output. GCC
  passed 84 anchor, 49 path, and 6 replay cases; repository-local Clang passed
  all 84 anchor cases.
- The canonical QuickBuild screening command used 32 threads, length 5000, the
  Paris4 `fiber_s1_002.lasagna.json` prediction manifest, reviewed fiber
  `dj_20260805T025256484_000003.json`, and Lasagna normals
  `las008_s1_full/las_008.lasagna.json`. Host load was below the agreed two-core
  exclusion threshold.
- Candidate performance was command wall `6.45` seconds, total CPU `166.90`
  seconds, anchor wall `3.734` seconds, anchor CPU `100.48` seconds, and peak RSS
  `1,655,756` KiB. Relative to checkpoint-28 medians, command wall improved
  `5.4%`, anchor wall `8.2%`, and anchor CPU `10.0%`.
- Quality regressed: retained anchors changed `2603 -> 2568`, graph nodes
  `2562 -> 2528`, graph edges `26445 -> 26082`, and DP relaxations
  `62,873,000 -> 62,214,882`. Greedy output remained exact across 631 compared
  points, but the fiberlet route changed from 352 to 351 points and fiberlet
  replay failures increased `1 -> 2`. The candidate artifact SHA-256 was
  `ff497ed5079b105a3371caf3735c96c86a694b6a08bda8fe092adfc77bb10616`,
  versus checkpoint 28
  `f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`.
- Rejected and removed the experiment after the decisive failure regression;
  three paired runs and visualization review were skipped as planned for an
  early quality rejection. Production behavior, user documentation, and
  changelog remain unchanged. Corrected the pre-existing `specs.md` default
  from two outer iterations to the implemented and documented value one.
- Final restored validation passes 83 GCC anchor, 49 GCC path, 6 GCC replay,
  and 83 repository-local Clang anchor cases. `git diff --check` passes and
  there is no remaining production-source or test diff from checkpoint 28.

## Checkpoint 31: largest-candidate-first DP scheduling

- Started an isolated scheduling experiment from committed checkpoint 28.
  Prepared retained-node count will be used as a stable, zero-extra-traversal
  work estimate. Workers will consume a separate descending-cost permutation;
  all candidate data, profiles, errors, and results remain in original order.
- This checkpoint changes only inter-candidate execution order. It does not
  change any candidate's node, transition, state, scoring, or accumulation
  order. A canonical screening replay will decide retention before transition
  arithmetic work begins.
- Independent review accepted the isolated trial but corrected the estimate's
  description: retained nodes are a deterministic heuristic, while direct-index
  initialization depends on key-layout size and dominant DP work depends on
  reached states and valid transitions. The implementation will record complete
  candidate and worker durations, test canonical profile/error indexing, and
  include peak RSS in the retention decision.
- Implemented a stable descending retained-node-count permutation and temporary
  complete candidate/worker timing. GCC and repository-local Clang each passed
  all 50 fiberlet-path cases, including exact serial/parallel JSON and OBJ
  equivalence.
- The canonical 32-thread screening replay measured `1.2263` seconds search
  wall and `38.36` seconds search CPU. Worker busy times ranged only
  `1.2140-1.2191` seconds, maximum candidate solve time was `0.0173` seconds,
  and retained-node-count/solve-duration Pearson correlation was `0.4762`.
  Command wall was `6.92` seconds, total CPU `179.54` seconds, anchor wall
  `4.135` seconds, and peak RSS `1,671,000` KiB.
- The artifact SHA-256 remained exactly
  `f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`,
  with 2,603 anchors, 2,562 graph nodes, 26,445 edges, 62,873,000 relaxations,
  and 2 greedy / 1 fiberlet failures. Because the existing dynamic queue was
  already balanced and no search or command gain appeared, the experiment and
  profile-schema additions were removed.

## Checkpoint 32: DP transition-cost profiling

- Started a measurement-only split of the remaining DP work into coarse
  per-candidate phases. Timing will stay outside individual transition calls so
  the profiler does not manufacture the bottleneck it is intended to measure.
- Independent review found that outgoing construction and transition scoring
  are interleaved per reached node, so coarse timers cannot split them directly.
  The revised two-tier profile uses four full candidate boundaries, then times
  the interleaved sections only on a deterministic candidate sample. It also
  distinguishes generated-edge lookups, scored transitions
  (`valid + reused`), and successful relaxations.
- GCC and repository-local Clang each passed all 49 fiberlet-path cases. The
  canonical instrumented replay measured `1.2300` seconds search wall and
  `38.56` seconds search CPU, versus `1.2263`/`38.36` for the immediately prior
  uninstrumented scheduling run. The temporary profile therefore adds only
  about 0.3% search-wall overhead.
- Of `38.9321` aggregate DP worker-seconds, initialization/source seeding used
  `3.9327`, propagation `33.5895`, sink evaluation `1.3890`, traceback
  `0.0173`, and residual `0.0036`. The deterministic sample covered 808
  candidates, 103,097 reached nodes, 504,620 valid edges, and 2,062,773 scored
  transitions. Sampled propagation split into `0.2372` seconds outgoing
  construction/lazy validation, `0.3145` transition scoring/relaxation, and
  `0.0192` residual.
- Command wall was `6.89` seconds, peak RSS `1,664,732` KiB, and artifact SHA
  remained exactly
  `f2b8e679c23470d1221f7930a21b0c37fa0906845de0bc2cbf3e8ab7329f78ee`,
  with unchanged populations and 2 greedy / 1 fiberlet failures.

## Checkpoint 33: prepared DP direction reuse

- Started an isolated removal of second normalization for directions already
  constructed from a checked positive finite length. The checkpoint-32 profile
  remains temporarily active to show whether outgoing construction improves.
- Independent review found that the current internal edge checks do not
  explicitly reject infinite lengths; the existing second normalization only
  sanitizes the resulting direction. The experiment will add explicit finite
  guards, remain limited to geometry-created directions, and retain all decoded
  prediction/normal normalization. Removing the second normalization permits
  small cost changes near DP ties, so route quality rather than exact artifact
  identity is the acceptance gate.
- Implemented explicit finite edge-length checks, collapsed duplicate incoming
  direction/length fields, and passed checked `delta / length` directions
  directly to prepared metric scoring. Prediction and normal normalization are
  unchanged. GCC and repository-local Clang each passed all 49 path cases.
- Three canonical instrumented runs measured search wall
  `1.2051/1.2156/1.2158` seconds and DP worker time
  `38.1351/38.4658/38.4838` seconds (min/median/max). The identically
  instrumented checkpoint-28 profile measured `1.2300`/`38.9321`; median DP
  worker time improved 1.2%. Median command wall was `6.90` seconds, but that
  remained dominated by anchor variation and is not claimed as an enclosing
  gain.
- All three candidate artifacts had SHA-256
  `833bd8cbf8699fb2a3f4558402ee2112fa705adbe2d9036a97db34707f6d4882`.
  Accumulated costs and eight relaxation decisions changed slightly, but both
  fiberlet route-point arrays and greedy output were exact against checkpoint
  28. Populations remained 2,603 anchors, 2,562 graph nodes, 26,445 edges, and
  failures remained 2 greedy / 1 fiberlet. Temporary DP profiling was then
  removed for the required production-build measurement.
- The required final uninstrumented build disproved that retention decision:
  search wall was `1.2266` seconds and DP worker time `38.8303` seconds, versus
  checkpoint-28 `1.2263` and roughly `38.7-38.9`. The apparent instrumented
  gain was a code-generation interaction with the temporary profile. Because
  production had no measurable gain while costs and eight relaxations changed,
  the direction reuse and its proposed docs/spec/changelog updates were removed.
- After restoring the checkpoint-28 source exactly, final validation passed all
  49 GCC fiberlet-path cases, 6 GCC replay cases, and 49 repository-local Clang
  fiberlet-path cases. No checkpoint-31 through checkpoint-33 source,
  documentation, specification, or changelog changes remain.

## Checkpoint 34: inline prepared DP metric implementation

- Optimized-binary inspection found that every hot prepared DP transition
  crosses interposable calls to `fiberLocalMetricCostPrepared()`,
  `fiberLocalAlignmentLoss()`, and `fiberLocalSmoothnessCost()`. Checkpoint 32
  measured transition scoring/relaxation as 55.1% of sampled propagation, so a
  shared private inline implementation is the next isolated target.
- Independent review required three private primitives in one source-private
  header, call-site-specific disassembly, exported-symbol verification, wider
  exact branch parity, and alternating rebuilt baseline/candidate timing. These
  constraints were incorporated before implementation.
- The first helper version used external-linkage `inline` functions. Binary
  inspection caught that GCC still emitted interposable helper calls. Changing
  the source-private primitives to standard `static inline` removed every
  scoring-helper call from the propagation loop while leaving the exported
  wrappers and all five public symbols intact.
- Exact generic/prepared parity now covers valid normal-aware scoring, null and
  invalid current predictions, candidate-axis sign flips, invalid candidates,
  isotropic fallback, degenerate directions, nonpositive lengths, and
  non-finite presence. All 49 GCC path, 6 GCC replay, and 49 repository-local
  Clang path cases pass.
- An initial paired set was discarded because the detached baseline had
  `VC_TESTING=OFF` while the established QuickBuild had `VC_TESTING=ON`. The
  baseline was reconfigured to match the relevant main build options before
  final timing; no result from the mismatched set is used below.
- Three alternating uninstrumented pairs measured baseline/candidate medians:
  command wall `7.87/7.75` seconds, search wall `1.1652/1.0509`, search CPU
  `36.4426/32.9779`, DP worker time `36.8775/33.2330`, fiberlet wall
  `2.1290/2.0077`, fiberlet CPU `62.5471/59.1116`, and peak RSS
  `1,611,608/1,601,256` KiB. The search, DP, and fiberlet improvements were
  repeatable in every pair; total command wall improved despite monotonically
  rising anchor time across the alternating sequence.
- All six runs retained 2,519 anchors, 48,852 searched candidates, 24,475
  accepted candidates, 47,790,462 evaluated nodes, 58,058,924 relaxations, and
  2 greedy / 1 fiberlet failures. Every `fiber_replay.json` was byte-identical
  with SHA-256
  `904c39d08e39c6b7b65ac95fd47d28d50e254a33609201c92aef71c6cc131308`.

## Checkpoint 35: lazy isotropic smoothness evaluation

- Started from committed checkpoint 34 (`0d104426e`). The shared smoothness
  primitive currently evaluates one isotropic `acos` and its weighted penalty
  before normal validation, even though the common normal-aware path discards
  both whenever its two projected tangents are valid.
- The isolated experiment will defer that calculation to invalid-normal or
  degenerate-projected-tangent fallback paths. Returned arithmetic and output
  must remain exact, and alternating uninstrumented measurements against a
  matching checkpoint-34 build will decide retention.
- Independent review found that the existing generic/prepared parity test is
  not an independent oracle because both paths share this helper. The corrected
  plan adds a test-local legacy equation, all one-sided/two-sided tangent and
  invalid-normal branches, baseline/candidate code-generation inspection, and
  load checks before counterbalanced `B/C, C/B, B/C` benchmark runs. It also
  clarifies that this is shared local scoring even though fiberlet DP is the
  measured hotspot.
- Implemented the shared-helper rewrite and an independent test-local copy of
  the old equation. Exact branch parity covers valid projected tangents,
  `normalValid=false`, a valid zero normal, previous-only tangent degeneracy,
  candidate-only degeneracy, and two-sided degeneracy. GCC passed 50 path and
  6 replay cases; repository-local Clang passed all 50 path cases.
- GCC baseline inspection found two angle call sites in the solve body: the
  eager isotropic angle and the projected-tangent angle. The candidate has
  branch-specific invalid-normal, valid-tangent, and degenerate-tangent call
  sites, so the common valid-normal/valid-tangent path executes one `acos`
  instead of two. Candidate Clang also keeps fallback calls behind branches.
- Canonical command, with either baseline or candidate binary and a distinct
  output directory:

  ```text
  /usr/bin/time -v <vc_fiberlets> fiberlet-replay \
    /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json \
    /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fibers/david/Paris4_fibers/dj_20260805T025256484_000003.json \
    <output> \
    --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json \
    --threads 32 --length 5000 --maximum-iterations 2
  ```

  Both builds used matching repository `QuickBuild` configuration with
  `VC_TESTING=ON`. Baseline was commit `0d104426e`; run order was
  `B/C, C/B, B/C`, with host-load checks before every run. Logs and artifacts
  are under `volume-cartographer/build/benchmarks/checkpoint35/`.

  | metric | baseline min/median/max | candidate min/median/max | median change |
  |---|---:|---:|---:|
  | command wall | `7.59/7.61/7.66 s` | `7.53/7.55/7.65 s` | `-0.8%` |
  | total CPU | `201.70/204.17/204.48 s` | `201.80/202.23/203.50 s` | `-1.0%` |
  | anchor wall | `5.1082/5.1200/5.1327 s` | `5.0888/5.1154/5.1825 s` | `-0.1%` |
  | fiberlet wall | `1.9432/1.9605/1.9944 s` | `1.9017/1.9091/1.9291 s` | `-2.6%` |
  | fiberlet CPU | `57.485/57.974/58.520 s` | `56.221/56.431/56.739 s` | `-2.7%` |
  | search wall | `1.0242/1.0368/1.0399 s` | `0.9696/0.9723/0.9870 s` | `-6.2%` |
  | search CPU | `32.219/32.571/32.601 s` | `30.547/30.556/30.850 s` | `-6.2%` |
  | DP worker | `32.362/32.772/32.864 s` | `30.618/30.713/31.173 s` | `-6.3%` |
  | peak RSS | `1,604,484/1,608,012/1,613,112 KiB` | `1,605,436/1,605,860/1,612,480 KiB` | `-0.1%` |

- All six measured runs retained 2,519 anchors, 48,852 searched / 24,475
  accepted candidates, 170,521 sampled voxels, 47,790,462 evaluated DP nodes,
  58,058,924 relaxations, and 2 greedy / 1 fiberlet failures. Every measured
  `fiber_replay.json` and the screening artifact were byte-identical with
  SHA-256
  `904c39d08e39c6b7b65ac95fd47d28d50e254a33609201c92aef71c6cc131308`.
- Retained the checkpoint because the targeted search and DP gains repeated in
  all three pairs, enclosing fiberlet time improved, command time did not
  regress, and replay output remained exact.

## Checkpoint 36: prepared outgoing-edge smoothness

- Started from committed checkpoint 35 (`08b4ea9cb`). The canonical profile
  records 28,279,855 valid outgoing edges and 83,916,118 reused edge
  evaluations, so candidate-side normal projection, tangent normalization, and
  inverse-sine are currently repeated about four times per valid edge.
- The isolated experiment will move those candidate-only intermediates into a
  shared private prepared descriptor stored on each stack-local `DpEdge`.
  Existing public and non-DP callers will prepare on demand through the same
  scorer. Exact costs, transitions, populations, failures, and replay bytes are
  required; alternating measurements against `08b4ea9cb` decide retention.
- Independent review required the descriptor to own its normal, preservation
  of the invalid-prediction early return, a direct private-path comparison with
  an independent full-metric oracle, non-finite/degenerate branch coverage,
  descriptor-size and reuse accounting, and explicit benchmark reproducibility
  details. The plan now includes each requirement. It also removes the proposed
  redundant candidate normal component; only its computed angle is retained.
- Implemented the source-private prepared descriptor and one shared
  candidate-prepared smoothness/metric path. The unchanged public prepared
  scorer preserves invalid-prediction early return and otherwise prepares on
  demand. Each valid `DpEdge` now prepares once after prediction-deviation
  admission and reuses the descriptor for every reached incoming state.
- Added a direct private-path test against an independent test-local copy of
  the complete legacy metric equation. Coverage includes valid normal-aware
  scoring, invalid/zero/NaN/Inf normals, each and both tangent degeneracies,
  zero/NaN/Inf directions, invalid prediction, and nonpositive lengths. GCC
  passed 51 path and 6 replay cases; repository-local Clang passed all 51 path
  cases. All five exported local-scoring symbols retain their existing ABI.
- The prepared descriptor is 32 bytes and `DpEdge` grows from 24 to 56 bytes,
  increasing the fixed nine-edge worker stack by 288 bytes. Disassembly maps
  the candidate `asin` to outgoing-edge construction at `FiberPaths.cpp:2014`;
  the reused transition loop retains only the previous-side `asin`. The
  baseline had both inverse-sine calls together in transition scoring.
- Canonical command, substituting only the baseline/candidate executable and a
  distinct output directory:

  ```text
  /usr/bin/time -v <vc_fiberlets> fiberlet-replay \
    /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json \
    /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fibers/david/Paris4_fibers/dj_20260805T025256484_000003.json \
    <output> \
    --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json \
    --threads 32 --length 5000 --maximum-iterations 2
  ```

  Both builds used matching QuickBuild configuration with `VC_TESTING=ON`.
  Baseline was commit `08b4ea9cb`; order was `B/C, C/B, B/C`, with host-load
  checks before every run. Logs and artifacts are under
  `volume-cartographer/build/benchmarks/checkpoint36/`.

  | metric | baseline min/median/max | candidate min/median/max | median change |
  |---|---:|---:|---:|
  | command wall | `7.50/7.53/7.56 s` | `7.52/7.52/7.53 s` | `-0.1%` |
  | total CPU | `199.42/200.06/200.20 s` | `199.16/200.31/201.12 s` | `+0.1%` |
  | anchor wall | `5.0809/5.0846/5.1172 s` | `5.0880/5.1034/5.1043 s` | `+0.4%` |
  | anchor CPU | `141.753/142.159/142.395 s` | `142.139/142.969/143.247 s` | `+0.6%` |
  | fiberlet wall | `1.8906/1.8996/1.9083 s` | `1.8745/1.8814/1.9008 s` | `-1.0%` |
  | fiberlet CPU | `55.605/55.725/55.828 s` | `54.945/55.268/55.813 s` | `-0.8%` |
  | search wall | `0.9632/0.9701/0.9710 s` | `0.9471/0.9505/0.9554 s` | `-2.0%` |
  | search CPU | `30.133/30.251/30.397 s` | `29.713/29.821/29.978 s` | `-1.4%` |
  | DP worker | `30.418/30.640/30.696 s` | `29.935/29.988/30.192 s` | `-2.1%` |
  | peak RSS | `1,614,972/1,616,896/1,619,728 KiB` | `1,609,032/1,609,532/1,619,236 KiB` | `-0.5%` |

- All six runs retained 2,519 anchors, 48,852 searched / 24,475 accepted
  candidates, 28,279,855 valid outgoing edges, 83,916,118 reused edge
  evaluations, 58,058,924 relaxations, and 2 greedy / 1 fiberlet failures.
  Every artifact was byte-identical with SHA-256
  `904c39d08e39c6b7b65ac95fd47d28d50e254a33609201c92aef71c6cc131308`.
- Retained the checkpoint because search and DP gains repeated in all three
  pairs, enclosing fiberlet time improved, RSS did not regress, and replay
  output remained exact. Future performance checkpoints will use one invariant
  benchmark launcher command and change baseline/candidate/output selection
  only through local launcher configuration.

## Checkpoint 37: prepared two-sided alignment inputs

- Started from committed checkpoint 36 (`d9cebed3f`). The canonical run has
  about 19.4 million reached incoming states, 28.3 million valid outgoing
  edges, and 112.2 million scored transitions. Current-side orientation and one
  clamped factor are therefore reused about 5.8 times per incoming state;
  candidate-side orientation, presence clamping, and one clamped factor are
  reused about four times per valid edge.
- The experiment will move only side-local work into shared private prepared
  descriptors. Four pair-dependent dot products and the original seven-factor
  multiplication order remain in the transition scorer. Public callers prepare
  on demand through the same implementation. Exact costs, populations,
  failures, and replay bytes are the correctness gate; an invariant launcher
  and load-checked alternating runs decide retention.
- Independent review required preserving the standalone alignment API's raw
  caller-oriented semantics, one owned authoritative candidate direction,
  explicit exclusion of endpoint paths, exact factor-order wording, broader
  nonfinite/randomized/DP-reuse coverage, and post-run load checks. The plan
  now includes all constraints before implementation.
- Implemented one owned 64-byte candidate metric descriptor containing its
  authoritative direction, oriented prediction axis, individual clamped
  candidate factors, and existing smoothness state. It replaces the former
  direction-plus-smoothness edge fields, so `DpEdge` grows from 56 to 76 bytes
  rather than carrying duplicated directions. A transient 28-byte incoming
  descriptor is prepared once per reached interior state.
- The standalone alignment API and all endpoint paths remain unchanged. The
  shared private fully prepared scorer retains all four pair-dependent dots and
  the exact original factor order. GCC and repository-local Clang each pass 52
  focused path cases; GCC passes all 6 replay cases. Tests include 1,024
  deterministic randomized bitwise comparisons against the independent legacy
  equation, nonfinite/negative-zero branches, poisonous invalid candidates,
  and explicit DP edge-reuse assertions. All five public scoring symbols remain
  exported.
- Candidate-side preparation maps to outgoing-edge construction and incoming-
  side preparation maps before the transition scan in optimized GCC output.
  The matching QuickBuild `VC_TESTING=ON` baseline at `d9cebed3f` is rebuilt.
  A fixed no-argument launcher under the checkpoint build directory records
  selection, output, and pre/post host load. Timed replay is waiting because an
  unrelated `vc_fiberlets` process currently occupies about 25 cores.
- The first baseline screening attempt is excluded. The unrelated
  `vc_fiberlets` process remained present despite a low instantaneous load
  average; the screening run received only 892% CPU, incurred 1,806 major
  faults, and took 28.68 seconds rather than the established roughly 7.5
  seconds. No timing from that run is used. Further runs wait until the other
  process disappears and reuse the exact same launcher command.
- Corrected the launcher scope after review: the checkpoint-specific command
  was removed and replaced by the generic permanent command
  `volume-cartographer/build/benchmarks/fiberlet_replay/run`. Checkpoint,
  executable, commit, output root, and label are selection-file data, so future
  checkpoints reuse the command unchanged.
- After the competing process cleared, the permanent command was:

  ```text
  volume-cartographer/build/benchmarks/fiberlet_replay/run
  ```

  Its selection file expanded to the canonical `vc_fiberlets fiberlet-replay`
  invocation on `fiber_s1_002.lasagna.json` and
  `dj_20260805T025256484_000003.json`, with the Lasagna normal manifest, 32
  threads, length 5,000, and two maximum iterations. Baseline and candidate
  were matching QuickBuild `VC_TESTING=ON` builds; baseline was commit
  `d9cebed3f`. Three warm pairs ran in order `B/C, C/B, B/C`. Logs and outputs
  are under `volume-cartographer/build/benchmarks/checkpoint37/runs/`.

  | metric | baseline min/median/max | candidate min/median/max | median change |
  |---|---:|---:|---:|
  | command wall | `7.51/7.52/7.53 s` | `7.53/7.58/7.62 s` | `+0.8%` |
  | total CPU | `200.42/200.63/201.29 s` | `200.04/200.48/200.91 s` | `-0.1%` |
  | anchor wall | `5.0801/5.0885/5.0988 s` | `5.1366/5.1479/5.1821 s` | `+1.2%` |
  | anchor CPU | `143.271/143.317/143.726 s` | `143.295/143.519/143.875 s` | `+0.1%` |
  | fiberlet wall | `1.8723/1.8768/1.8815 s` | `1.8612/1.8693/1.8785 s` | `-0.4%` |
  | fiberlet CPU | `55.031/55.284/55.483 s` | `54.681/54.887/54.955 s` | `-0.7%` |
  | search wall | `0.9488/0.9495/0.9549 s` | `0.9216/0.9279/0.9331 s` | `-2.3%` |
  | search CPU | `29.825/29.907/30.037 s` | `29.002/29.060/29.197 s` | `-2.8%` |
  | DP worker | `29.978/29.984/30.158 s` | `29.096/29.290/29.441 s` | `-2.3%` |
  | peak RSS | `1,600,788/1,603,572/1,610,764 KiB` | `1,599,980/1,609,776/1,624,868 KiB` | `+0.4%` |

- All six measured runs retained 2,519 anchors, 48,852 searched / 24,475
  accepted candidates, 28,279,855 valid outgoing edges, 83,916,118 reused edge
  evaluations, 58,058,924 relaxations, and 2 greedy / 1 fiberlet failures.
  Every artifact was byte-identical with SHA-256
  `904c39d08e39c6b7b65ac95fd47d28d50e254a33609201c92aef71c6cc131308`.
- Retained the checkpoint because the targeted search and DP gains repeated in
  all three pairs, enclosing fiberlet wall and CPU improved, total CPU did not
  regress, and replay output remained exact. The command-wall increase tracks
  the untouched anchor phase and is recorded rather than attributed to this
  local DP change.

## Checkpoint 38: portable batched transition alignment

- Started from committed checkpoint 37 (`836303cea`). The remaining interior
  loop evaluates four pair-dependent alignment dots for about 112.2 million
  scored state/edge pairs. Smoothness contains branch-dependent inverse
  trigonometry and remains scalar; relaxation order also remains scalar.
- The experiment will expose only the independent fixed-nine alignment work as
  compact SoA arrays. A shared private primitive owns the arithmetic, and the
  existing scalar scorer remains the correctness oracle. Standard C++ plus
  GCC/Clang optimized-code inspection decides whether the batch is genuinely
  vectorized or merely adds packing overhead.
- Independent review identified QuickBuild `-O1`, invalid-lane reads, dot/FMA
  ordering, duplicated cost assembly, packing traffic, and scalar-vs-batch DP
  comparison as explicit risks. The revised plan compares finite fixed lanes
  with compact valid lanes, never reads invalid source descriptors, retains
  shared cost assembly, validates all 512 masks bitwise under matching
  optimized flags, records layout/code-generation evidence, and uses identical
  optimized baseline/candidate builds behind the invariant launcher.
- Implemented a compact-valid-lane SoA batch in the shared source-private local
  scorer. Outgoing construction appends valid edges in ascending transition-
  slot order. Each reached incoming state computes the four pair-dependent
  alignment dots for that compact batch; the existing scalar loop then performs
  shared smoothness/cost assembly, accumulated-state addition, strict-less
  comparisons, and backpointer writes in the same order. Source, sink, and
  direct paths remain unchanged.
- The chosen batch is 312 bytes and its output array is 36 bytes. `DpEdge`
  remains 76 bytes. GCC stack-usage reports show `solveCandidate()` growing
  from 2,064 to 2,480 bytes (`+416` bounded dynamic bytes). The canonical run
  has 28,354,560 valid outgoing edges over 5,963,689 reached nodes, or 4.75
  compact lanes per batch, and reuses them for 84,166,532 incoming-state/edge
  evaluations.
- GCC 16 `-O3` reports the batch loop vectorized with 32-byte vectors and an
  eight-lane unroll plus a 16-byte epilogue. Clang `-O3` reports vectorization
  width eight and interleave count one. Both implementations remain portable
  scalar C++ with no intrinsics, explicit alignment, or fast-math.
- Added an all-512-mask scalar/batch bitwise oracle with poisoned invalid source
  slots and edge cases, plus a multi-incoming-state relaxation fixture that
  compares complete cost components, totals, relaxation counts, and predecessor
  choices. GCC and Clang QuickBuild each pass 54 path cases and 6 replay cases.
  The matching GCC `-O3` replay suite passes. Its full path suite retains 295
  failures in the older independent legacy metric bitwise oracle at line 393;
  those optimizer-dependent failures existed before this batch, while the new
  batch checks emit no failures. No sanitizer-configured local build existed,
  so sanitizer execution was not added during this checkpoint.
- Matching GCC 16 `-O3`, no-LTO, `VC_TESTING=ON` baseline and candidate builds
  used the invariant command
  `volume-cartographer/build/benchmarks/fiberlet_replay/run`. The launcher used
  `fiber_s1_002.lasagna.json`,
  `dj_20260805T025256484_000003.json`, the regular Lasagna normal manifest, 32
  threads, length 5,000, and two maximum iterations. Baseline was checkpoint-37
  commit `836303cea`; three warm pairs ran `B/C, C/B, B/C`. Logs and artifacts
  are under `volume-cartographer/build/benchmarks/checkpoint38/runs/`.

  | metric | baseline min/median/max | candidate min/median/max | median change |
  |---|---:|---:|---:|
  | command wall | `5.47/5.54/5.58 s` | `5.41/5.50/5.64 s` | `-0.7%` |
  | total CPU | `139.76/141.22/141.75 s` | `139.09/140.01/141.86 s` | `-0.9%` |
  | anchor wall | `3.4458/3.4476/3.4915 s` | `3.4354/3.5253/3.5722 s` | `+2.3%` |
  | anchor CPU | `90.887/91.652/91.825 s` | `90.795/91.529/92.175 s` | `-0.1%` |
  | fiberlet wall | `1.6307/1.6779/1.6995 s` | `1.5789/1.5820/1.6594 s` | `-5.7%` |
  | fiberlet CPU | `46.913/47.675/48.012 s` | `46.403/46.593/47.783 s` | `-2.3%` |
  | search wall | `0.7796/0.7899/0.8545 s` | `0.7429/0.7433/0.7737 s` | `-5.9%` |
  | search CPU | `23.256/23.868/24.042 s` | `23.183/23.292/23.583 s` | `-2.4%` |
  | DP worker | `24.678/24.986/27.088 s` | `23.525/23.528/24.512 s` | `-5.8%` |
  | peak RSS | `1,347,112/1,347,144/1,359,368 KiB` | `1,347,348/1,349,252/1,356,008 KiB` | `+0.2%` |

- Every measured run retained 2,521 anchors, 48,944 searched / 24,526 accepted
  candidates, 28,354,560 valid edges, 84,166,532 reused edge evaluations,
  58,211,093 relaxations, and 2 greedy / 1 fiberlet failures. Every artifact
  was byte-identical with SHA-256
  `79e9163de700ed1f93e3ae2c15073cf1fb196d5678f296d8126b5c6dbcc291aa`.
- Retained the compact layout because the targeted search/DP and enclosing
  fiberlet gains repeated while replay output remained exact. The planned
  initialized fixed-nine neutral-lane alternative was intentionally not
  implemented after compact lanes both vectorized and avoided 4.25 invalid
  lanes per average batch; it would add masked invalid-lane work without
  addressing a measured deficiency. This is the checkpoint's only functional
  experiment-plan simplification.

## Checkpoint 39: reusable robust-proposal eligibility

- Selected anchor robust proposal after checkpoint 38 because the canonical
  candidate medians leave anchor fitting at about 91.5 CPU-seconds and 3.53
  seconds wall. Robust tensor proposal alone costs about 29.0 worker-seconds
  over 2.43 billion reported logical visits, versus 23.3 CPU-seconds for the
  now-optimized fiberlet search.
- Current compact proposal repeatedly checks the same observation validity,
  finite normalized direction, and presence-floor predicate. The fitter already
  performs one complete logical-observation scan to derive bounds before any
  proposal. The planned candidate reuses that scan to build ascending eligible
  logical indices, retaining complete-cardinality output arrays and original
  accumulation order among observations that can contribute.
- The first stage is measurement-only: split axis-producing and membership-only
  calls and report logical/eligible visits and time. No reusable index will be
  added unless the measured eligible fraction and repeated-call count justify
  its allocation and indirection.
- Added profile schema version 21 without changing proposal traversal or
  decisions. The regular build passed 83 anchor, 54 path, and 6 replay tests.
- One canonical run reported 24,550 axis proposals over 809,364,400 logical /
  525,650,776 eligible observations and 12,275 membership proposals over
  404,682,200 logical / 262,825,388 eligible observations. Eligibility was
  64.9%, so repeated proposals performed 425,570,436 predicate checks on
  observations that could never contribute. Axis proposals cost 21.56 worker-
  seconds and membership refreshes 10.61 worker-seconds. This justifies a
  production-path candidate.
- The candidate is restricted to compact observations. It will gather ascending
  `uint32_t` logical indices during the existing bounds scan, then use them for
  proposal accumulation and cutoff application. Expanded/public observations
  retain their current per-call direction normalization and traversal.
- Independent review required version-21 schema documentation, unambiguous
  logical/eligible/indexed/cutoff counters, checked compact indices, memory-
  budgeted reusable per-worker scratch, explicit optimized-path semantics, and
  matching checkpoint-38 benchmark conditions. The plan now includes these
  corrections. A private proposal test hook is intentionally not added: public
  zero-eligible input exits before refinement and zero iterations is rejected,
  while existing compact extraction parity and profile counters cover the
  reachable production path without duplicating private equations.
- Implemented one checked ascending `uint32_t` eligibility index per compact
  cell during the fitter's existing bounds scan. Every robust proposal reuses
  it, while assignments and retained-inlier state remain indexed to the full
  logical support. The vector is reusable worker scratch and its worst-case
  capacity is included in anchor live-memory admission. Expanded/public fitting
  retains its previous traversal.
- The regular GCC build passes 83 anchor, 54 path, and 6 replay tests. The
  matching GCC `-O3` build passes anchor and replay coverage; its path suite has
  the same 295 failures in the old optimizer-dependent bitwise oracle at line
  393 as checkpoint 38, while the new tests pass. The regular candidate screen
  preserved exact output and reduced proposal work by 21.9%.
- The first checkpoint-39 baseline was mistakenly configured at QuickBuild
  optimization level one while the candidate used level three. Results under
  `checkpoint39/o3-runs` are excluded. Both corrected builds use GCC 16,
  QuickBuild optimization level three, no LTO, `VC_TESTING=ON`, and the
  invariant launcher. Corrected logs are under
  `volume-cartographer/build/benchmarks/checkpoint39/o3-corrected-runs/`.

  | metric | baseline min/median/max | candidate min/median/max | median change |
  |---|---:|---:|---:|
  | command wall | `5.41/5.41/5.44 s` | `5.26/5.36/5.38 s` | `-0.9%` |
  | total CPU | `139.57/139.70/140.67 s` | `135.13/137.95/137.96 s` | `-1.3%` |
  | anchor wall | `3.4303/3.4382/3.4609 s` | `3.3026/3.3788/3.4056 s` | `-1.7%` |
  | anchor CPU | `91.339/91.402/92.109 s` | `87.326/89.404/89.568 s` | `-2.2%` |
  | robust proposal work | `27.6979/27.7541/27.9421 s` | `21.6085/22.0790/22.3082 s` | `-20.4%` |
  | fiberlet wall | `1.5764/1.5780/1.5827 s` | `1.5594/1.5759/1.5843 s` | `-0.1%` |
  | peak RSS | `1,354,772/1,356,116/1,362,628 KiB` | `1,350,988/1,353,100/1,354,280 KiB` | `-0.2%` |

- Every corrected run retained 2,521 anchors, 48,944 searched / 24,526
  accepted candidates, identical DP populations and failures, and byte-
  identical SHA-256
  `79e9163de700ed1f93e3ae2c15073cf1fb196d5678f296d8126b5c6dbcc291aa`.
  Retained the checkpoint because the targeted and enclosing gains repeat with
  no quality or memory regression.

## Checkpoint 40: reusable robust-proposal result storage

- Selected full-support proposal allocation, initialization, and copying as the
  next behavior-preserving target. Each canonical nonempty cell normally runs
  two axis proposals plus one final membership proposal. Each call constructs
  two complete-cardinality byte vectors. Every accepted iteration additionally
  copies both vectors into evaluation state, but no consumer reads that state
  before the unconditional final membership refresh replaces it.
- The candidate will initialize one pair of full-support vectors per cell,
  overwrite all traversed entries on every proposal, retain sentinel values for
  immutable compact-path ineligible entries, remove only the dead intermediate
  copies, and move final membership into the returned evaluation. Arithmetic,
  traversal, membership refresh, and empty-component behavior remain unchanged.
- Measurement-only profile schema 22 confirms 36,825 proposal-buffer
  initializations, zero reuses, 2,428,093,200 initialized bytes, and another
  2,428,093,200 bytes copied from proposals into evaluation state on the
  canonical run. This first copy count excludes the additional final-support
  copy identified during review and added to the measurement before the
  candidate.
- Independent review required fit-local ownership, explicit two-byte-per-
  observation memory admission, complete fixed-summary and traversed-entry
  reset, formula-based rather than workload-fixed profile assertions, and
  coverage for component removal and empty output. It also identified a second
  full membership copy when final support evaluation replaces the state after
  peak search. The candidate now updates final scalar summaries in place and
  does not attempt cross-cell buffer reuse.
- The combined reuse candidate reduced initialized bytes from 2,428,093,200 to
  809,364,400 and evaluation-copy bytes from 3,237,457,600 to zero. Despite a
  roughly 10% local-control improvement, three matching GCC `-O3` pairs left
  total CPU flat and changed median command wall from `5.23` to `5.24` seconds.
  Caller-owned proposal reuse was therefore removed rather than retained for
  its apparent cleanliness.
- The narrowed candidate restores the original return-value robust-proposal
  kernel. It removes the dead per-iteration copies, moves final membership into
  the result, updates final support scalars in place after peak search, and
  includes its two membership bytes per observation in worker memory admission.
  Proposal initialization remains unchanged; evaluation-copy bytes are zero.
- One narrowed-candidate run stopped making progress after 12,492 / 13,027
  cells and was interrupted. Four other narrowed-candidate runs and all prior
  checkpoint-40 runs completed. The stalled run is excluded from timing and
  recorded as an isolated concurrency anomaly; it did not recur in two
  immediate confirmation runs.
- Three completed narrowed-candidate runs are compared with the same three
  fresh matching optimized baselines under
  `volume-cartographer/build/benchmarks/checkpoint40/`:

  | metric | baseline min/median/max | candidate min/median/max | median change |
  |---|---:|---:|---:|
  | command wall | `5.19/5.23/5.23 s` | `5.12/5.17/5.17 s` | `-1.1%` |
  | total CPU | `133.59/133.83/133.85 s` | `131.59/132.11/132.38 s` | `-1.3%` |
  | anchor wall | `3.2613/3.2804/3.2927 s` | `3.2112/3.2391/3.2583 s` | `-1.3%` |
  | anchor CPU | `86.219/86.344/86.446 s` | `84.882/85.284/85.375 s` | `-1.2%` |
  | fitting work | `67.0548/67.2324/67.5103 s` | `65.9451/66.1249/66.9410 s` | `-1.6%` |
  | local control work | `4.0345/4.0604/4.0914 s` | `3.8848/3.8947/3.9233 s` | `-4.1%` |
  | fiberlet wall | `1.5414/1.5442/1.5539 s` | `1.5204/1.5253/1.5360 s` | `-1.2%` |
  | peak RSS | `1,350,692/1,352,552/1,356,436 KiB` | `1,345,272/1,358,412/1,363,088 KiB` | `+0.4%` |

- Every completed run retained identical populations, failures, DP work, and
  byte-identical SHA-256
  `79e9163de700ed1f93e3ae2c15073cf1fb196d5678f296d8126b5c6dbcc291aa`.
  The narrowed candidate is retained; reusable proposal storage is rejected.

## Checkpoint 41: float peak-response accumulation

- Selected the remaining peak response loop because retained checkpoint 40
  leaves roughly 21 worker-seconds over about 2.37 billion hot response-record
  visits. Each response maintains six compensated float sums. The experiment
  changes only those accumulators to same-order ordinary float addition; all
  record traversal, Gaussian/evidence arithmetic, response caching, candidate
  order, and decisions remain structurally unchanged.
- Independent review found the premise stale: `FloatSum::add()` is already
  exactly `sum += value`, and checkpoint 24 applied it to these peak-response
  accumulators. The proposed raw-float spelling would not remove any arithmetic
  or memory traffic. The experiment was rejected before source changes or
  timing; an in-progress duplicate baseline build was stopped. No production,
  test, specification, user-documentation, or changelog changes result.

## Checkpoint 42: sparse peak signal storage

- Retained checkpoint 40 prepares about 199.3 million response records and
  only 10.1 million evidence records across the canonical extraction. The hot
  loop performs about 2.37 billion response-record visits but only about 80.2
  million evidence visits. Signal is nonzero only for evidence-bearing records,
  so storing it in every 16-byte hot record wastes one field load plus a
  multiply/add on the dominant no-evidence population.
- The candidate moves signal into sparse evidence, reducing the hot record to
  12 bytes and growing the evidence record from 16 to 20 bytes. It removes only
  exact-zero numerator additions; every nonzero numerator contribution retains
  its original response-record order.
- Independent review corrected the invariant to one-way implication: nonzero
  signal requires evidence, but positive-alignment evidence can have zero
  signal when presence is zero. The implementation must branch on the evidence
  index and still add a zero signal for such evidence. Review also required
  compile-time 12/20-byte layout assertions and explicit boundary coverage. It
  confirmed this extends checkpoint 16 rather than repeating its rejected
  evidence-index-in-hot-record prototype.
- Implemented a 12-byte response record containing only transverse coordinates
  and axial Gaussian. Signal now lives in the 20-byte sparse evidence record.
  Response evaluation accumulates every denominator term, then loads evidence
  and adds its signal and gradient terms. Positive-alignment, zero-presence
  evidence remains represented and contributes an explicit zero numerator.
- Regular GCC passed 83 anchor, 54 path, and 6 replay tests. The optimized
  candidate also passed the anchor suite. Three counterbalanced GCC `-O3`,
  no-LTO pairs are under
  `volume-cartographer/build/benchmarks/checkpoint42/runs/`:

  | metric | baseline min/median/max | candidate min/median/max | median change |
  |---|---:|---:|---:|
  | command wall | `5.11/5.13/5.14 s` | `5.10/5.10/5.12 s` | `-0.6%` |
  | total CPU | `132.37/133.18/133.66 s` | `131.66/131.71/132.12 s` | `-1.1%` |
  | anchor wall | `3.2198/3.2216/3.2356 s` | `3.1935/3.2005/3.2133 s` | `-0.7%` |
  | anchor CPU | `85.697/86.310/86.589 s` | `84.931/85.030/85.366 s` | `-1.5%` |
  | fitting work | `66.1087/66.4133/66.6739 s` | `65.3739/65.4847/65.4928 s` | `-1.4%` |
  | peak-search work | `19.9842/20.0239/20.0694 s` | `19.0222/19.0278/19.0543 s` | `-5.0%` |
  | fiberlet wall | `1.5057/1.5151/1.5261 s` | `1.5092/1.5136/1.5150 s` | `-0.1%` |
  | peak RSS | `1,347,508/1,347,528/1,350,648 KiB` | `1,350,176/1,355,920/1,357,304 KiB` | `+0.6%` |

- Every run retained identical populations, failures, DP work, and byte-
  identical SHA-256
  `79e9163de700ed1f93e3ae2c15073cf1fb196d5678f296d8126b5c6dbcc291aa`.
  The candidate is retained because its targeted 5.0% gain and enclosing CPU
  gain repeat without a quality regression.

## Checkpoint 43: packed peak evidence presence

- Selected the dense 32-bit evidence-index stream as the next peak bandwidth
  target. Evidence records are appended in response order and only about 5% of
  response records carry evidence, so random indices encode more information
  than response evaluation needs. A packed bit identifies evidence-bearing
  responses; a sequential cursor consumes sparse evidence in the same order.
- The experiment keeps the response record and evidence record from checkpoint
  42, denominator traversal, nonzero contribution order, peak equations, and
  all decisions. Retention requires focused word-boundary coverage, complete
  memory accounting, GCC/Clang validation, exact canonical replay parity, and
  a repeatable enclosing gain.
- Independent review found one critical ordering requirement: the sparse cursor
  must advance for a set bit even when that response record is radially
  rejected, although the evidence itself need not be loaded. It also required
  explicit profile-schema replacement, separate allocation/admission
  accounting, unsigned overflow-safe packing, and adversarial repeated-scan and
  63/64/65-boundary tests. The plan now includes all corrections. The bitmap
  remains a valid experiment, but the universal bit test may outweigh the
  substantial memory reduction; measurement decides retention.
- Implemented the reviewed candidate with a reusable packed-map helper,
  version-24 bitmap allocation counters, worst-case worker admission, and
  direct 0/1/63/64/65/129-record boundary coverage. GCC passed 84 anchor, 54
  path, and 6 replay tests; the optimized anchor suite also passed.
- The canonical optimized screen preserved exact artifact SHA-256
  `79e9163de700ed1f93e3ae2c15073cf1fb196d5678f296d8126b5c6dbcc291aa`
  and reduced maximum peak-observation storage from roughly 0.92 MiB to
  `729,424` bytes. However, the bitmap had to be tested and its sparse cursor
  advanced before radial rejection for all 2.37 billion response visits.
  Peak-search work regressed to `21.9527` worker-seconds versus checkpoint 42's
  `19.02-19.05`, anchor CPU to `87.9517` seconds, and command wall to `5.22`
  seconds. The candidate was removed without further pairs or Clang testing.
- Checkpoint 43 is rejected. Production source is restored exactly to the
  checkpoint-42 baseline; only this experiment record remains.

## Checkpoint 44: isolated robust proposal workspace

- Selected a controlled revisit of checkpoint 40's rejected caller-owned
  proposal storage. That candidate removed about 1.6 GiB of initialization and
  3.2 GiB of copies and improved local control around 10%, but neighboring hot
  code in the same translation unit regressed enough to erase total benefit.
- First extract the single shared robust implementation behind compact and
  expanded entry points and verify a neutral exact-output baseline. Only then
  add fit-local reusable workspace in that isolated module. This directly
  addresses the prior code-generation failure rather than repeating the
  rejected implementation unchanged.
- Independent review approved the two-gate experiment but required explicit
  compact storage/logical/eligible index spans, complete workspace reset and
  transfer semantics, one owner for histogram/Gaussian/cutoff policy, and a
  profile-version bump when initialization semantics change. It also stressed
  that TU isolation changes the kernel's own ABI/inlining and is therefore only
  an experiment, not a guaranteed fix. The plan now includes those corrections
  and limits extraction to one private proposal module; refinement remains in
  `FiberAnchors.cpp`.
- Checkpoint 44a extracted the shared robust proposal and cutoff implementation
  into a private translation unit. GCC anchor/path/replay tests passed and the
  canonical replay retained SHA-256
  `79e9163de700ed1f93e3ae2c15073cf1fb196d5678f296d8126b5c6dbcc291aa`.
  An initial comparison was invalid because the isolated baseline was
  accidentally configured as `RelWithDebInfo -O2`; it produced a different
  artifact and is excluded. After rebuilding the exact snapshot as
  `QuickBuild -O3`, paired baseline/candidate command wall was `5.20/5.21 s`,
  anchor wall `3.2566/3.2802 s`, and anchor CPU `85.922/86.592 s`. Robust axis
  and membership work rose from `14.4616/7.1267` to `14.6748/7.2303` worker-
  seconds. The extraction is wall-neutral enough to run checkpoint 44b, but it
  is not independently a performance win and will survive only if workspace
  reuse produces an enclosing gain.
- Checkpoint 44b initialized one logical-cardinality workspace per nonempty fit,
  reused it for the second axis proposal and final membership proposal, reset
  only the compact eligible entries, and moved final membership into the fit
  result. The canonical screen remained byte-identical. Initializations fell
  from `36,825` / `2,428,093,200` bytes to `12,275` / `809,364,400` bytes, with
  `24,550` reuses resetting `525,650,776` indexed entries. Despite that traffic
  reduction, robust axis/membership work regressed further to
  `14.9992/7.4334` worker-seconds. Command wall was unchanged at `5.20 s`,
  anchor wall was `3.2760 s`, and anchor CPU was `86.640 s`; there is no
  enclosing gain over either the `5.20 s` checkpoint-42 baseline or 44a.
  Checkpoints 44a and 44b were therefore both removed, including the private
  module and version-24 profile additions. Production returns exactly to
  checkpoint 42.

## Checkpoint 45: final accepted robust membership reuse

- Selected the remaining membership-only robust refresh, which costs about
  `7.13` worker-seconds over `404,682,200` logical and `262,825,388` eligible
  observation visits in the canonical run. The experiment will move the last
  accepted axis proposal's membership into final evaluation instead of
  recomputing it after the final spatial update.
- Review identified the semantic boundary: component-removal retries cannot
  publish their old component labels; zero-component exits keep their explicit
  empty state; only a proposal associated with an accepted component state may
  be retained. The final assignments may differ because the membership was
  evaluated one accepted spatial update earlier. Exact output is therefore not
  a gate; geometry/support distributions and replay quality are.
- The optimized canonical screen removed all `12,275` final membership calls,
  `404,682,200` logical visits, and `262,825,388` eligible/cutoff visits. Anchor
  CPU fell from `85.92` to `77.90` worker-seconds and command wall from `5.20`
  to `5.13` seconds. However, retained anchors fell `2,519 -> 2,490`, searched
  fiberlets `48,944 -> 48,078`, accepted fiberlets `24,526 -> 24,095`, and
  graph nodes `2,473 -> 2,438`. The artifact changed to SHA-256
  `fa744ce59c3f197910624252c75313513a1b653064bfe0cb733941c1e8eeb3f4`.
  Greedy/fiberlet failure counts stayed `2/1`, but losing 1.1-1.8% extraction
  coverage for about 1.3% wall improvement is not acceptable. The candidate,
  profile field, tests, and schema bump were removed without further pairs or
  Clang testing.

## Checkpoint 46: shared-union compact observations

- Current exact-union sampling stores 6,162,456 raw samples, then copies them
  into 39,539,352 additional tile occurrences before independently computing
  35,843,136 gradients and building tile compact observations. Canonical work
  is about `2.94` worker-seconds for tile sample copies, `4.9` for gradients,
  and `11.9` for observation construction including cell indexing.
- The candidate will construct gradients and compact observations once in
  shared-union order and use dense tile-local uint32 maps. Review requires
  preserving tile-local Z/Y/X traversal and original tile-interior gradient
  validity even when the larger union happens to provide neighbors outside a
  tile. Exact row/interval lookup avoids both a full-volume array and a copied
  private paged-index implementation.
- Implemented partition-owned compact observations and gradients. Tile-local
  support and owned traversal map through dense uint32 indices; explicit tile-
  interior checks preserve the old gradient-validity boundary. Raw samples are
  released before fitting. Admission includes raw-plus-compact coexistence,
  preparation control, persistent compact storage, tile maps, ready queues,
  timing storage, and per-cell scratch.
- Profile version 24 reports shared-observation voxels/construction and tile-
  index-map work separately. On the canonical workload, gradient construction
  fell from `35,843,136` repeated tile computations to `7,138,880` partition-
  union computations. Tile map construction costs about `0.05-0.07` worker-
  seconds, shared observation construction about `0.48`, and gradient work
  about `0.95`, replacing roughly `19.5` worker-seconds of tile copies,
  repeated gradients, and compact-observation construction.
- Three clean checkpoint-42 baselines and three matching candidate runs are in
  `volume-cartographer/build/benchmarks/checkpoint46/runs/`. The first attempted
  third baseline (`pair3-baseline`) overlapped an unrelated compile/LTO job and
  is excluded; `pair4-baseline` is its clean replacement.

  | metric | baseline min/median/max | candidate min/median/max | median change |
  |---|---:|---:|---:|
  | command wall | `5.15/5.22/5.26 s` | `4.87/4.87/4.89 s` | `-6.7%` |
  | total CPU | `132.53/132.98/133.88 s` | `123.36/123.82/123.94 s` | `-6.9%` |
  | anchor wall | `3.2245/3.2766/3.2843 s` | `2.8866/2.8916/2.9017 s` | `-11.7%` |
  | anchor CPU | `85.636/85.724/85.953 s` | `75.898/75.933/76.163 s` | `-11.4%` |
  | peak RSS | `1,350,328/1,351,736/1,361,404 KiB` | `1,240,440/1,255,576/1,255,728 KiB` | `-7.1%` |

- The finalized schema/accounting replay remained at `4.86 s` wall,
  `2.8847 s` anchor wall, and `75.282 s` anchor CPU. GCC and Clang passed 83
  anchor, 54 path, and 6 replay cases. Every canonical run retained populations
  (`2521` anchors, `48944` searched, `24526` accepted, `2473` graph nodes),
  failures (`2/1`), and byte-identical SHA-256
  `79e9163de700ed1f93e3ae2c15073cf1fb196d5678f296d8126b5c6dbcc291aa`.
  Checkpoint 46 is retained.

## Checkpoint 47: packed robust membership

- Selected the two-stream robust membership state after checkpoint 46. The
  canonical fit performs about `788.5M` physically indexed robust proposal/
  cutoff visits, `1.61B` centroid visits, `1.62B` refined-state visits,
  `804.4M` peak-preparation visits, and `404.7M` final-evaluation visits.
  Most downstream scans currently inspect separate assignment and retained
  arrays.
- The experiment packs transient component and 8-bit residual bin into one
  uint16 stream, then rewrites it after cutoff to component or unassigned.
  This preserves two bytes of bounded storage per logical observation while
  reducing stream count, initialization passes, and downstream loads. It does
  not reuse stale membership or alter fitting semantics.
- Independent review required disjoint tagged transient/normalized encodings,
  one shared unsigned decoder across all translation units, explicit loss of
  otherwise-unobserved trimmed assignments, non-identity compact-index tests,
  zero/component-removal paths, and 16-bit/256-bin static assertions. The plan
  now includes each correction. Payload remains two bytes per observation;
  the experiment targets one stream and one downstream load, not smaller
  bounded storage.
- Implemented the reviewed uint16 representation and shared decoder, including
  cutoff-boundary, component, invalid, and non-identity indexed coverage. GCC
  and optimized anchor suites passed, and the canonical replay retained the
  exact artifact SHA-256
  `79e9163de700ed1f93e3ae2c15073cf1fb196d5678f296d8126b5c6dbcc291aa`.
- Three clean counterbalanced checkpoint-46/candidate pairs measured median
  command wall `4.89/4.88 s`, anchor wall `2.9049/2.9062 s`, and anchor CPU
  `76.112/75.838 s`. The differences are within run noise: wall improved only
  `0.2%`, anchor wall regressed `0.05%`, and anchor CPU improved `0.36%`.
  Peak RSS was also effectively unchanged because one uint16 stream has the
  same payload as two uint8 streams.
- Checkpoint 47 is rejected as performance-neutral. The private membership
  helper and all production/test edits were removed, restoring checkpoint 46;
  only this experiment record remains.

## Checkpoint 48: contiguous robust-proposal evidence

- Selected the repeated compact proposal dereference as the next measured
  target. Checkpoint 46 uses partition-shared observations, so every eligible
  proposal visit follows an eligible logical index through the cell index into
  shared storage. The canonical run performs `525,650,776` axis and
  `262,825,388` final-membership indexed visits.
- The experiment pays one canonical-order materialization per compact cell for
  a proposal-only record containing position, already-normalized direction,
  presence, and logical destination. Both axis proposals and final membership
  reuse it. All downstream full-support consumers remain unchanged.
- Retention requires bounded memory accounting, focused indexed/parity tests,
  unchanged fit decisions and replay quality, and a repeatable enclosing gain.
- Review confirmed that eligibility does not imply a finite position: invalid
  positions must remain in the prepared stream and naturally receive zero
  Gaussian mass. It also requires original logical destinations, increasing
  logical-order traversal, immutable reuse across component-removal retries,
  an explicit empty-evidence path, and replacing rather than adding the old
  eligible-index bytes in bounded admission.
- Implemented a 32-byte record and reused worker-local capacity across cells.
  Version 25 reports `262,825,388` prepared records in the canonical run and
  `3.97` worker-seconds of one-time preparation. The two axis passes fell from
  a median `13.727` to `12.732` worker-seconds and final membership from
  `6.828` to `6.330`.
- Three clean counterbalanced pairs measured median command wall
  `4.82 -> 4.81 s`, anchor wall `2.8644 -> 2.8503 s`, anchor CPU
  `75.504 -> 74.530 s`, and local refinement `39.905 -> 38.910 s`. Accounted
  worst-case live fitting memory increases `220,209,496 -> 307,422,552` bytes,
  while measured median RSS did not increase (`1,253,484 -> 1,242,400 KiB`).
- GCC and Clang anchor/path/replay suites pass. All measured candidates retain
  `2521` anchors, `48944` searched and `24526` accepted fiberlets, `2473` graph
  nodes, failures `2/1`, and exact SHA-256
  `79e9163de700ed1f93e3ae2c15073cf1fb196d5678f296d8126b5c6dbcc291aa`.
  Checkpoint 48 is retained for its repeatable `1.3%` anchor-CPU and `2.5%`
  local-refinement gains.

## Checkpoint 49: prepared compact spatial objectives

- Selected the `~14.2` worker-second refined-state phase. Its compact kernel
  repeatedly follows cell-to-shared indices for every denominator site and
  again for retained numerator evidence.
- The experiment adds one logical-order float position stream and consumes the
  retained checkpoint-48 evidence for numerators. Denominator and numerator
  accumulators are independent; splitting their traversals preserves each
  contribution order. Expanded/public fitting stays on the original path.
- Review requires preserving invalid positions in the position stream, using
  original logical membership indices, checked cardinalities, component and
  zero-active behavior, worker-local capacity reuse, and complete additional
  memory admission.
- Implemented the compact-only split traversal and profile schema 26. The
  optimized anchor test passed and the canonical replay preserved populations,
  failures, and exact SHA-256
  `79e9163de700ed1f93e3ae2c15073cf1fb196d5678f296d8126b5c6dbcc291aa`.
- The screen decisively regressed: command wall was `4.92 s`, anchor wall
  `2.9271 s`, and anchor CPU `76.874 s`, versus checkpoint-48 medians of
  `4.81 s`, `2.8503 s`, and `74.530 s`. State evaluation rose to `15.155`
  worker-seconds, preparation rose to `4.938` worker-seconds, and accounted
  live bytes increased to `344,799,576`.
- Checkpoint 49 is rejected. The position stream, split objective APIs, profile
  fields, tests, and schema bump were removed. Production returns to retained
  checkpoint 48; Clang testing and repeated pairs were skipped because the
  targeted and enclosing metrics both regressed.

## Checkpoint 50: trusted compact evaluation paths

- Selected repeated compact input validation. Every retained spatial-objective
  and final-evaluation call scans the full logical index span before scanning
  it again for arithmetic, although production indices and membership vectors
  are generated together from bounded internal storage.
- The experiment keeps checked compact APIs and their failure tests unchanged.
  New explicitly trusted private calls share the same kernels and are used only
  by production `IndexedObservationRange` fitting. Expanded/public fitting and
  all equations, traversal, accumulation, and decisions remain unchanged.
- The combined spatial/final screen produced the exact artifact at `74.614 s`
  anchor CPU. Isolating trusted spatial objectives produced `73.845` and
  `74.648 s`, with state evaluation near `14.0-14.1` worker-seconds.
- A matched checkpoint-48 runtime rebuilt from the same source/build context
  measured `73.375 s` anchor CPU, `14.090 s` state evaluation, and `4.517 s`
  final evaluation. The candidates therefore have no repeatable enclosing
  gain. Checkpoint 50 is rejected and all trusted APIs/tests were removed.

## Checkpoint 51: prepared compact centroid evidence

- Selected the `~1.5` worker-second centroid phase. It reports `1.61B` logical
  visits because it loops the complete range once per active component, even
  though only robust-retained proposal-eligible records can reach an
  accumulator.
- Production compact fitting can reuse checkpoint-48 records without new
  storage. Their original logical indices preserve membership lookup and their
  increasing order preserves every contributing accumulation. Expanded/public
  fitting remains unchanged.
- The canonical screen preserved `2521` anchors, `48944/24526` searched/
  accepted fiberlets, failures `2/1`, and exact artifact SHA-256
  `79e9163de700ed1f93e3ae2c15073cf1fb196d5678f296d8126b5c6dbcc291aa`.
  Centroid work was `1.526` worker-seconds versus `1.511` in the matched
  checkpoint-48 runtime. Checkpoint 51 is rejected and removed.

## Checkpoint 52: one-pass robust refinement

- Selected the existing `maximumIterations` quality knob. The canonical
  checkpoint-48 run executes `24,550` robust attempts for `12,275` nonempty
  cells, exactly two per fit, and spends about `38` worker-seconds in local
  refinement.
- The experiment changes only benchmark configuration from two passes to one.
  Production code and defaults remain unchanged until quality and speed are
  measured. Retention requires explicit quality-knob documentation rather than
  presenting the result as behavior-preserving.
- One pass measured `4.50 s` command wall, `2.435 s` anchor wall, and `61.588 s`
  anchor CPU, versus the matched two-pass runtime's `4.77 s`, `2.819 s`, and
  `73.375 s`. Local refinement fell `38.224 -> 24.140` worker-seconds.
- Failures remained `2/1`, while anchors rose `2521 -> 2604`, searched/accepted
  fiberlets `48944/24526 -> 51780/26494`, graph nodes `2473 -> 2563`, and DP
  relaxations `58.21M -> 63.01M`. The fiberlet failure arc moved from
  `8135.66` to `8138.96` base voxels. This is a real quality/population change.
- Retained only as guidance for the existing knob. User documentation already
  states that one pass is the speed default and two passes are the first
  quality-oriented setting for nearby/crossing fibers, so no production or
  documentation edit is needed.

## Checkpoint 53: portable SIMD peak-response Gaussian

- Selected the peak search, currently about `18.7` worker-seconds in the
  matched two-pass replay. The hot response loop evaluates roughly `2.36B`
  transverse Gaussian contributions; prior scalar LUT and polynomial
  approximations did not improve it.
- The experiment changes the 12-byte response payload from AoS to three float
  streams and uses OpenCV universal intrinsics for distance and exponential
  evaluation. Accumulation and sparse evidence handling remain scalar and in
  original observation order. This uses the existing portable OpenCV dependency
  and avoids x86-only code.
- Numerical identity is not required for this task, but replay populations,
  failure behavior, and geometry must remain comparable. The candidate will be
  removed if the enclosing replay does not improve.
- Independent review required an OpenCV 4/non-SIMD scalar fallback, scalar
  cutoff decisions at the support boundary, padded SIMD tails, a bounded
  vector-exponential input range, direct numerical tests, and explicit compiled-
  ISA rather than runtime-dispatch portability language. The plan now includes
  each correction and uses checkpoint 48 as the matched baseline.
- The focused GCC anchor and replay tests passed. The optimized canonical
  replay preserved `2521` anchors, `48944/24526` searched/accepted fiberlets,
  `2473` graph nodes, failures `2/1`, and the same failure arc positions.
- Performance did not improve: command wall was `4.80 s`, anchor wall
  `2.832 s`, anchor CPU `74.774 s`, and peak search `20.095` worker-seconds,
  versus the matched checkpoint-48 run's `4.77 s`, `2.819 s`, `73.375 s`, and
  `18.695` worker-seconds. Checkpoint 53 is rejected and all production, test,
  and helper edits were removed. Clang/OpenCV-4 validation was skipped after
  the targeted phase decisively regressed.

## Checkpoint 54: trust bounded subpixel peak fits

- Selected the `28,348` uncached acceptance response scans in the canonical
  peak search. At least `24,400` are the separable candidates used as final
  anchor positions; the remainder validate diagnostic joint candidates.
- The experiment retains the discrete local maximum, feasible neighbors,
  negative curvature, half-step clamp, and owner/window bounds, but trusts the
  bounded parabolic candidate instead of rescanning the complete observation
  stream. This is a deliberate quality tradeoff, not a numeric-equivalent
  optimization.
- Independent review required splitting diagnostic joint and production
  separable guards. Checkpoint 54a now changes only diagnostic joint acceptance;
  separable anchor positions and their response guard remain exact. A later
  54b must explicitly cover cross-coupled lower-response fits, near-flat
  curvature, partial axes, domain rejection, and matched-anchor geometry.
- Checkpoint 54a removed `3,948` diagnostic joint response scans and preserved
  the exact replay artifact. Two alternating pairs nevertheless increased peak
  work from `18.701/18.880` to `19.442/19.436` worker-seconds; a minimal
  branch-local variant still measured `19.396`. Anchor CPU and command wall did
  not improve consistently. The joint guard is restored and 54a is rejected.
- A temporary guarded profile for 54b found `8,144/24,400` separable candidates
  rejected by the owner/response guard. Their proposed offset sum was `452.05`
  and maximum `0.3373` prediction voxels. The candidate therefore adds a
  scale-aware curvature floor and is evaluated as an explicit geometry tradeoff.
- Checkpoint 54b removed all `28,348` acceptance scans and added a scale-aware
  negative-curvature floor. The optimized replay changed anchors `2521 -> 2516`,
  accepted fiberlets `24526 -> 24496`, and graph nodes `2473 -> 2469`; failure
  counts and arc locations remained stable. Peak work was `18.701 s` versus
  `18.695 s` in the matched checkpoint-48 baseline, with command wall `4.76 s`
  versus `4.77 s`. With no measurable saving and changed geometry/populations,
  checkpoint 54b is rejected and the response guards are restored.

## Checkpoint 55: weighted-direction proposal records

- Replaced separate normalized direction and presence in the private compact
  proposal record with `sqrt(presence) * direction`, reducing each record from
  `32` to `28` bytes. Expanded/public fitting remained unchanged and focused
  anchor/replay tests passed.
- The canonical screen retained `2521` anchors and failures `2/1`, but accepted
  fiberlets changed `24526 -> 24523`. Robust proposal work regressed from the
  matched checkpoint-48 `18.768 s` to `19.585 s`; anchor CPU rose
  `73.375 -> 74.573 s`, and command wall was `4.79 s` versus `4.77 s`.
- The added square root, norm reconstruction, and alignment division cost more
  than the 12.5% record shrink saved. Checkpoint 55 is rejected and removed.

## Checkpoint 56: robust proposal subphase profile

- Temporary exclusive timing on the restored checkpoint-48 implementation
  measured `18.580 s` total proposal work: accumulation `17.412 s` (93.7%),
  cutoff materialization `0.966 s`, cutoff selection `0.088 s`, buffer setup
  `0.078 s`, and tensor reduction `0.025 s`.
- The diagnostic replay preserved exact checkpoint-48 populations and failure
  locations. Temporary profile schema 26 and all timers were removed. Further
  work should target the accumulation kernel rather than buffer initialization
  or histogram reduction.

## Checkpoint 57: compile-time robust proposal modes

- Instantiated separate axis-producing and membership-only private proposal
  kernels so the latter had no tensor storage or per-observation tensor branch.
  Focused tests passed and the canonical artifact remained exact.
- Membership proposal work improved `6.167 -> 6.115 s`, but axis proposal work
  regressed `12.413 -> 12.719 s`; total proposal work was `18.834 s` versus the
  instrumented restored kernel's `18.580 s`. Command wall remained `4.76 s`.
- The split is rejected because code-generation growth displaced more axis-path
  performance than it saved in membership. The single runtime kernel is
  restored.

## Checkpoint 58: fixed component-count proposal kernels

- Instantiated one- and two-component proposal kernels and dispatched once per
  call. Focused tests and canonical output remained exact.
- Proposal work regressed to `19.296 s` (`12.902 s` axis and `6.394 s`
  membership), preparation rose to `4.099 s`, and anchor CPU reached
  `75.034 s`; command wall was `4.79 s`. The restored instrumented kernel was
  `18.580 s` proposal work and `73.737 s` anchor CPU.
- Checkpoint 58 is rejected. Compiler specialization/code-size growth did not
  improve the already small component loops, so runtime component count is
  restored.

## Checkpoint 59: split proposal logical-index stream

- Split the exact 32-byte proposal payload into a 28-byte hot observation and
  parallel 4-byte logical-index stream. Accumulation consumed both in lockstep;
  cutoff materialization consumed only indices. Focused tests and canonical
  output remained exact, and bounded scratch bytes were unchanged.
- Proposal work was effectively flat at `18.558 s` versus `18.580 s`, while
  proposal preparation rose `3.754 -> 4.251 s`. Anchor CPU regressed to
  `75.795 s` and command wall to `4.82 s`.
- Checkpoint 59 is rejected. The extra stream write/access costs as much as the
  narrow cutoff scan saves, so the contiguous 32-byte record is restored.

## Checkpoint 60: hoisted compact proposal geometry

- Hoisted compact-path position loading and finite checks out of the component
  loop, computed each observation's pivot offset once, and precomputed the two
  component-to-pivot offsets once per robust proposal call. The generic
  expanded/public Gaussian helper remains unchanged.
- Focused tests passed. A matched baseline measured `18.791 s` robust-proposal
  work, `74.140 s` anchor CPU, `2.815 s` anchor wall, and `4.76 s` command wall.
  Two candidate runs measured `12.168/12.187 s`, `67.995/67.841 s`,
  `2.622/2.624 s`, and `4.56/4.57 s`, respectively.
- Anchors remained `2521`, searched/accepted fiberlets remained
  `48944/24526`, graph nodes remained `2473`, and failures remained `2/1` at
  identical arc locations. Of 349 emitted route points, 9 changed; p95
  displacement was zero, mean displacement `8.43e-5`, and maximum displacement
  `0.00436732` base voxels.
- Checkpoint 60 is retained. It removes repeated invariant compact geometry
  without changing the fitting model or accepted extraction populations.

## Checkpoint 61: pivot-relative compact proposal records

- Changed the private production record to store position relative to its
  cell's fixed pivot. Compact proposal passes consume this offset directly,
  removing one vector subtraction per eligible record per pass. Record size,
  order, logical destinations, and expanded/public fitting are unchanged.
- All 83 focused anchor tests passed. Two canonical runs produced the exact
  checkpoint-60 SHA-256
  `29f583fbb254e1b0f48d2783430e01a1d6f7294ba5cfd40358434a105b092780`
  with unchanged populations and failures.
- Robust-proposal work improved from `12.168/12.187` to `11.770/11.813`
  worker-seconds. Anchor CPU was `67.430/67.638` seconds and command wall
  `4.55/4.56` seconds versus checkpoint 60's `67.995/67.841` and
  `4.56/4.57`. Checkpoint 61 is retained.

## Checkpoint 62: scalar transverse-distance evaluation

- Tested `dot(offset, offset) - dot(offset, axis)^2` in the private compact
  Gaussian kernel to avoid constructing the projected transverse vector. The
  large-coordinate translation fixture failed with `1.125` prediction-voxel
  anchor displacement due to cancellation.
- Tested the more stable cross-product norm as a narrowed alternative. The same
  fixture still failed with `0.166667` prediction-voxel displacement.
- No replay benchmark was run because focused correctness coverage decisively
  rejected both forms. The original projected-vector calculation is restored.

## Checkpoint 63: precomputed compact Gaussian constants

- Precomputed the cutoff square and reciprocal Gaussian denominator once per
  proposal call, replacing hot-loop division with multiplication. All 83
  focused tests passed and replay output remained byte-identical.
- Robust-proposal work measured `11.890` worker-seconds and command wall
  `4.57` seconds, slightly worse than checkpoint 61's `11.770/11.813` and
  `4.55/4.56`. Checkpoint 63 is rejected and removed.

## Checkpoint 64: fused objective denominator/numerator traversal

- Fused denominator and retained numerator updates around one Gaussian value,
  without checkpoint 17's temporary Gaussian arrays. All 83 focused anchor
  tests passed.
- The isolated objective nevertheless regressed: local state evaluation rose
  from `14.334` to `17.223` worker-seconds, anchor CPU from `67.638` to
  `70.679` seconds, and command wall from `4.56` to `4.65` seconds.
- Checkpoint 64 is rejected and removed. The compiler favors the original
  branch-free denominator pass despite its source-level numerator call.

## Checkpoint 65: selected-cell-first ready-queue ordering

- Stable-partitioned each prepared tile's ready tasks so selected cells enter
  the cooperative queue before context-only cells. Results remain indexed by
  canonical cell index; fitting and output order are unchanged.
- All 83 focused tests passed. Two canonical runs retained exact SHA-256
  `29f583fbb254e1b0f48d2783430e01a1d6f7294ba5cfd40358434a105b092780`
  and unchanged populations/failures.
- Anchor wall measured `2.586/2.615` seconds (median `2.601`) versus checkpoint
  61's `2.607/2.622` (median `2.615`). CPU was flat and command wall
  `4.53/4.57` versus `4.55/4.56`. Checkpoint 65 is retained as a small,
  deterministic scheduling improvement.

## Checkpoint 66: finite prepared-proposal invariant

- Required finite positions when preparing private compact proposal records,
  then removed repeated finite checks from all compact robust-proposal passes.
  Invalid positions still remain unassigned; public/expanded fitting remains
  defensive.
- All 83 focused tests passed. Both canonical runs retained exact SHA-256
  `29f583fbb254e1b0f48d2783430e01a1d6f7294ba5cfd40358434a105b092780`,
  populations, counters, and failures. Unchanged prepared-record counts confirm
  production generated positions were already all finite.
- Proposal work improved from checkpoint 65's `11.780/11.839` to
  `10.771/10.818` worker-seconds. Anchor CPU fell to `66.445/66.340` seconds
  and command wall to `4.53/4.52`. Checkpoint 66 is retained.

## Checkpoint 67: finite-position production objective kernel

- Added checked and finite-position compact objective instantiations and routed
  only production extraction through the latter. Focused invalid-position and
  valid parity tests passed.
- The additional instantiation severely perturbed the isolated objective code:
  state evaluation regressed from `14.309` to `21.240` worker-seconds, anchor
  CPU from `66.340` to `72.396` seconds, and command wall from `4.52` to
  `4.67` seconds.
- Checkpoint 67 is rejected and fully removed. This repeats the earlier trusted-
  objective code-size warning; no analogous final-evaluation specialization is
  attempted without a different implementation strategy.

## Checkpoint 68: finite generated positions in existing peak kernels

- Marked only the existing internally generated indexed compact observation
  range as having finite positions. Peak-owner bounds and peak-record
  preparation compile out their repeated finite checks for this range; generic
  and public paths remain defensive. No additional template instantiation or
  compatibility API was added.
- All 83 focused anchor tests passed. Two canonical replays retained exact
  SHA-256 `29f583fbb254e1b0f48d2783430e01a1d6f7294ba5cfd40358434a105b092780`,
  with unchanged populations and failures.
- Peak-search work measured `17.533/17.565` worker-seconds versus approximately
  `18.94` before this checkpoint. Anchor CPU measured `65.027/65.175` seconds
  versus `66.340`, and command wall measured `4.50/4.51` seconds versus `4.52`.
  Checkpoint 68 is retained.

## Checkpoint 69: square-root-free peak gradient votes

- Rewrote the clamped peak gradient vote into its algebraically equivalent
  squared-dot form. Inward/outward attribution uses the unchanged radial-dot
  sign; response traversal, sparse evidence, accumulation order, and peak
  search remain unchanged. This removes one square root from each of roughly
  `80.2M` evidence visits.
- All 83 focused anchor tests passed. Both canonical runs retained `2521`
  anchors, `48944/24526` searched/accepted fiberlets, `2473` graph nodes, and
  failures `2/1` at the same locations. Ten internal DP nodes changed from
  expression rounding, but all 349 emitted route points were exactly unchanged.
- Peak-search work improved from checkpoint 68's `17.533/17.565` to
  `17.366/17.422` worker-seconds. Anchor CPU improved from `65.027/65.175` to
  `64.736/64.792` seconds, and command wall from `4.50/4.51` to `4.48/4.48`.
  Checkpoint 69 is retained.

## Checkpoint 70: corridor endpoint fast acceptance

- Added a direct fast acceptance for lattice offsets strictly inside the
  transverse corridor radius. The corresponding layer center is a corridor
  polyline vertex, so these points cannot fail the segment-distance test.
  Exact-boundary and outer-square points retain the original segment search.
- The initial non-strict endpoint-distance variant admitted 691 boundary nodes
  because its floating endpoint test did not reproduce the segment routine's
  boundary rounding. That form is discarded. The strict lattice-radius form
  retains exact SHA-256
  `29f583fbb254e1b0f48d2783430e01a1d6f7294ba5cfd40358434a105b092780`,
  populations, DP counters, routes, and failures in both canonical runs.
- Corridor segment tests fell from `222.879M` to `172.709M`. Node-enumeration
  work improved from approximately `13.03/13.21` to `12.40/12.61` worker-
  seconds, fiberlet CPU from `45.04/45.33` to `44.58/44.94` seconds, and command
  wall measured `4.45/4.49` seconds. Checkpoint 70 is retained.

## Checkpoint 71: monotonic enumerated node keys

- Replaced repeated checked packed-key construction in local-node enumeration
  with the key already implied by the validated monotonic lattice traversal.
  The checked packing helper remains for independent callers. Node order,
  coordinates, key values, and all downstream behavior are unchanged.
- Both canonical runs retained exact SHA-256
  `29f583fbb254e1b0f48d2783430e01a1d6f7294ba5cfd40358434a105b092780`
  and all populations, counters, routes, and failures.
- Node-enumeration work improved from checkpoint 70's `12.399/12.615` to
  `12.159/12.250` worker-seconds, preparation CPU from `20.527/20.646` to
  `20.189/20.216`, and fiberlet CPU from `44.580/44.940` to `44.014/44.110`.
  Command wall was `4.45/4.47` seconds. Checkpoint 71 is retained.

## Checkpoint 72: row-invariant lattice geometry

- Explicitly hoisted each layer and fixed-`u` row base out of the inner lattice
  loop while preserving arithmetic parenthesization and exact replay output.
- Node enumeration measured `12.238` worker-seconds versus checkpoint 71's
  `12.159/12.250`, while fiberlet CPU regressed to `44.419` seconds versus
  `44.014/44.110`. The compiler already performs the useful invariant motion;
  the source rewrite does not improve the enclosing phase.
- Checkpoint 72 is rejected and the explicit row-base code is removed.

## Checkpoint 73: validated finite-grid bounds in node enumeration

- Reused the existing per-node finite assertion and the corridor builder's
  validated nonzero grid dimensions. Node enumeration now precomputes the XYZ
  float upper bounds once per candidate and performs only the inclusive range
  comparisons; the general defensive grid helper remains unchanged elsewhere.
- Both canonical runs retained `2521` anchors, `48944/24526` searched/accepted
  fiberlets, `2473` graph nodes, and failures `2/1` at identical locations.
  Eight internal retained/DP nodes changed from code-generation rounding, but
  all 349 emitted route points were exactly unchanged.
- Node-enumeration work improved from checkpoint 71's `12.159/12.250` to
  `11.602/11.702` worker-seconds, preparation CPU from `20.189/20.216` to
  `19.695/19.771`, fiberlet CPU to `43.437/43.469`, and command wall to
  `4.42/4.44`. Checkpoint 73 is retained.

## Checkpoint 74: layer-corner finite validation

- Replaced per-lattice-point finite checks with four affine transverse-corner
  validations per layer. Replay behavior remained comparable.
- Node enumeration measured `11.734` worker-seconds versus checkpoint 73's
  `11.602/11.702`, while geometry setup also increased. The compiler already
  exploits the finite/range relationship well enough that the explicit layer
  proof provides no target gain.
- Checkpoint 74 is rejected and the per-point finite guard is restored.

## Checkpoint 75: prepared corridor segments

- Precomputed each corridor segment's start, delta, and squared length while
  preserving segment traversal and point-distance arithmetic.
- The larger prepared-segment stream regressed node enumeration to `11.935`
  worker-seconds versus checkpoint 73's `11.602/11.702`, and fiberlet CPU to
  `43.691` seconds versus `43.437/43.469`. Saved arithmetic did not offset the
  additional memory traffic.
- Checkpoint 75 is rejected and the compact reference-point representation is
  restored.

## Checkpoint 76: two-sided incident-segment fast path

- Tested both centerline segments incident to each curved-domain layer before
  the complete corridor fallback scan. Containment, populations, routes, and
  failures remained unchanged.
- Corridor segment tests fell only from checkpoint 73's `172.709M` to
  `172.379M` (`0.19%`). Node-enumeration work regressed to `11.780` worker-
  seconds versus `11.602/11.702`, and fiberlet CPU regressed to `43.541`
  seconds versus `43.437/43.469`.
- Checkpoint 76 is rejected and the single preferred incoming segment is
  restored.

## Checkpoint 77: reusable rolling DP layer buffers

- Reused two candidate-local vectors for alternating current and next DP
  layers instead of constructing the next-layer vector on every layer.
  Populations, routes, counters, and failures remained unchanged.
- Search CPU regressed to `22.280` seconds and DP work to `22.396` worker-
  seconds versus the retained baseline's approximately `21.82/21.91` seconds.
  The saved allocations do not offset the retained capacities and repeated
  assignment path.
- Checkpoint 77 is rejected and the original per-layer vector construction is
  restored.

## Checkpoint 78: local incident-segment corridor

- Replaced the production complete-polyline corridor scan with the two segment
  capsules incident to each curved-domain layer. The general complete-polyline
  helper remains available for diagnostics. This also prevents a distant bend
  from admitting shortcut nodes into an unrelated layer.
- Two canonical runs retained `2521` anchors, `48944/24526` searched/accepted
  fiberlets, `2473` graph nodes, identical routes and failures, and exact
  replay SHA-256
  `29f583fbb254e1b0f48d2783430e01a1d6f7294ba5cfd40358434a105b092780`.
  Only `2606` of `47.8M` internal retained nodes were removed.
- Segment tests fell from checkpoint 73's `172.709M` to `48.911M`. Node-
  enumeration work improved from `11.602/11.702` to `10.888/10.930` worker-
  seconds, preparation CPU from `19.695/19.771` to `18.933/18.952`, and
  fiberlet CPU from `43.437/43.469` to `42.702/43.010`. Command wall measured
  `4.41/4.43` seconds. Checkpoint 78 is retained.

## Checkpoint 79: layer-local prepared corridor segments

- Hoisted the two incident segments' delta and squared length into each layer
  and routed local corridor tests through a prepared scalar helper. Replay
  output and all populations remained unchanged.
- Node-enumeration work measured `10.935` worker-seconds versus checkpoint 78's
  `10.888/10.930`, and fiberlet CPU measured `42.885` seconds versus
  `42.702/43.010`. There is no measurable benefit.
- Checkpoint 79 is rejected and the original compact point/segment helper is
  restored.

## Checkpoint 80: single-page interpolation-cell corner insertion

- Extracted one authoritative interpolation-cell decomposition shared by
  weighted interpolation and corner collection. Cells with eight active
  corners in one `16^3` sparse bitmap page resolve that page once and set the
  eight local bits directly; integer, boundary, and page-crossing cells retain
  general insertion.
- Two canonical runs retained exactly `170512` globally sampled voxels in the
  same order, all path/graph populations and failures, and replay SHA-256
  `29f583fbb254e1b0f48d2783430e01a1d6f7294ba5cfd40358434a105b092780`.
- Corner-collection work improved from checkpoint 78's `8.054/8.068` to
  `2.827/2.854` worker-seconds, preparation CPU from `18.933/18.952` to
  `13.541/13.637`, fiberlet CPU to `37.004/37.541`, and command wall from
  `4.41/4.43` to `4.23/4.27` seconds. Checkpoint 80 is retained.

## Checkpoint 81: chordal interior-DP smoothness

- Replaced inverse-trigonometric angular squares only in reused interior DP
  transitions with squared chordal angular distance. Source, sink, direct,
  public, and diagnostic scoring remain on the angular implementation.
- Two canonical runs retained `2521` anchors, `48944/24526`
  searched/accepted fiberlets, `2473` graph nodes, failures `2/1`, and exact
  emitted route coordinates and lengths. Total and component costs changed as
  expected; DP relaxations changed by about ten thousand out of `58.2M`.
- Search CPU improved from checkpoint 80's approximately `21.94/22.02` to
  `19.334/19.478` seconds. Fiberlet CPU improved from `37.004/37.541` to about
  `35.0` seconds and command wall from `4.23/4.27` to `4.15/4.16` seconds.
  Checkpoint 81 is retained as an intentional numerical approximation with
  unchanged canonical geometry.

## Checkpoint 82: chordal-only outgoing-edge preparation

- Split the shared candidate-smoothness preparation at compile time so public
  angular callers still compute `normalAngle`, while interior chordal DP does
  not compute the unused per-edge `asin`.
- Both canonical runs were byte-identical to checkpoint 81, including costs,
  populations, routes, graph, and failures.
- Search CPU measured `19.156/19.205` seconds versus checkpoint 81's
  `19.334/19.478`, and wall measured `4.15/4.17` seconds. Checkpoint 82 is
  retained as a small behavior-neutral removal of dead work.

## Checkpoint 83: batched chordal smoothness

- Extended the outgoing alignment SoA with normal/tangent fields and evaluated
  chordal smoothness across its valid lanes before scalar DP relaxation.
- Canonical populations, routes, and failures remained unchanged, but search
  CPU measured `19.132` seconds versus checkpoint 82's `19.156/19.205`, which
  is performance-neutral.
- Checkpoint 83 is rejected. The widened batch, batched kernel, and tolerance-
  based oracle are removed; checkpoint 82's compact scalar path is restored.

## Checkpoint 84: projected-component chordal smoothness

- Replaced normalized tangent/normal-angle chordal interior smoothness with a
  decomposition into the scalar normal projection and unnormalized tangent-
  plane vector. At equal weights their squared differences sum exactly to the
  full direction chord. Zero-free-angle evaluation needs no square root,
  normalization, division, or trigonometry per transition.
- Two canonical runs kept `2521` anchors, `48944/24526` searched/accepted
  fiberlets, `2473` graph nodes, failures `2/1`, and exact emitted route points
  and lengths. Internal reached nodes and graph transitions changed slightly.
- Search CPU improved from checkpoint 82's `19.156/19.205` to
  `18.612/18.612` seconds; wall measured `4.14/4.15` seconds. Checkpoint 84 is
  retained and the superseded private normalized-chordal implementation is
  removed.

## Checkpoint 85: compact projected-chordal edge descriptor

- Replaced each stored outgoing edge's full angular candidate metric with a
  private projected descriptor containing only direction, normal, normal
  projection, and mode; its layout is asserted at no more than 32 bytes.
- Both canonical artifacts were byte-identical to checkpoint 84. Search CPU
  improved from `18.612/18.612` to `18.414/18.438` seconds and wall remained
  about `4.14` seconds. Checkpoint 85 is retained.

## Checkpoint 86: direct outgoing alignment append

- Wrote prediction orientation and alignment factors directly into the SoA
  batch instead of constructing a full temporary candidate metric.
- Search CPU measured `18.355` seconds versus checkpoint 85's
  `18.414/18.438`, but cost serialization changed from floating-point
  instruction-order differences. The small gain does not justify the duplicate
  implementation or additional numerical divergence.
- Checkpoint 86 is rejected and the shared candidate preparation plus append
  path is restored.

## Checkpoint 87: optional spatial-objective verification

- Made direct installation of the projected, clamped retained-evidence centroid
  the default local position update. Added `--verify-spatial-objective` to retain
  the previous full-support objective comparison and deterministic halving.
- Profile schema version 26 adds direct-centroid acceptances. Fast mode reports
  `24550` direct acceptances, zero backtracking evaluations, and zero refined-
  state evaluation visits; verification-focused tests retain the old path.
- Two canonical fast-mode runs were byte-identical. Compared with checkpoint
  85, anchors changed `2521 -> 2520`, accepted fiberlets `24526 -> 24493`, and
  graph nodes `2473 -> 2472`; failures remained `2/1`. Route p50/p95 movement
  was zero and maximum displacement was `0.005524` base voxels.
- Anchor CPU improved from `64.837` to `50.797/50.251` seconds and command wall
  from about `4.14` to `3.68/3.69` seconds. Checkpoint 87 is retained as an
  explicit speed/quality policy with the stricter verifier still available.

## Checkpoint 88: lazy separable grid-Gaussian streams

- Cached signed radial offsets and one-dimensional Gaussian factors for each
  first/second grid coordinate actually requested by peak search. Subpixel
  acceptance retained the original scalar response, and the 2D cutoff,
  evidence traversal, gradients, response order, and hill climb were unchanged.
- The canonical run materialized `169282` streams containing `1.382B` values.
  Peak work regressed `17.294 -> 23.871` worker-seconds, anchor CPU
  `50.251 -> 57.638` seconds, and command wall `3.69 -> 3.89` seconds.
- Checkpoint 88 is rejected. The stream cache and temporary version-27 profile
  fields are removed, restoring checkpoint 87.

## Checkpoint 89: post-cutoff compact tensor accumulation

- Replaced per-residual-bin compact tensor updates with one assigned-mass value
  per prepared observation and accumulated retained tensors during cutoff
  materialization. Public/expanded fitting remained unchanged.
- The canonical run kept command wall at `3.69` seconds, while robust-axis work
  regressed `7.130 -> 7.356` worker-seconds and local refinement
  `16.197 -> 16.642` seconds. Anchors changed `2520 -> 2519` and accepted
  fiberlets `24493 -> 24480`.
- Checkpoint 89 is rejected. The mass stream, changed accumulation order, and
  extra bounded-memory accounting are removed.

## Checkpoint 90: nearest-table peak Gaussian

- Replaced only the transverse peak-response `exp(-x)` after the exact radial
  cutoff with a nearest lookup in an immutable 2049-entry float table over
  `[0,8]`. Out-of-domain/nonfinite values retain `exp`; axial Gaussians and all
  search, evidence, gradient, and acceptance logic are unchanged.
- Added one shared private helper and direct coverage for exact endpoints,
  monotonicity, fallback, NaN propagation, and less than `0.21%` maximum
  relative error. All 84 focused anchor cases pass.
- Three canonical runs were deterministic with SHA-256
  `d4277b5f87a189aa4a1f96118733120ae25d7d80438b53459ad3dc9ce4f5ee2b`.
  Peak work improved `17.294 -> 15.465/15.496/15.328`, anchor CPU
  `50.251 -> 48.452/48.592/48.308`, and wall
  `3.69 -> 3.59/3.63/3.58` seconds.
- Anchors remained `2520`, accepted fiberlets changed by three, graph nodes
  remained `2472`, and failures remained `2/1`. Route geometry remained close;
  one route omitted one point, with symmetric nearest-path p95 zero and maximum
  `6.57` base voxels. Checkpoint 90 is retained.

## Checkpoint 91: compact robust-proposal Gaussian lookup

- Reused the checkpoint-90 lookup for the compact robust direction proposal
  while retaining exact cutoff predicates and the expanded/public path.
- Wall improved only `3.58 -> 3.56` seconds and anchor CPU
  `48.308 -> 47.845` seconds. In exchange, anchors changed `2520 -> 2513` and
  searched/accepted fiberlets changed `48897/24490 -> 48660/24388`.
- Checkpoint 91 is rejected. Robust membership is too sensitive to this
  approximation for its small enclosing gain, and exact `exp` is restored.

## Checkpoint 92: reuse final-evaluation Gaussian values

- Reused the Gaussian already computed for each component denominator when
  accumulating the retained assigned component's numerator and presence mass.
- The replay artifact remained byte-identical to checkpoint 90. Final-
  evaluation work improved `4.495 -> 4.312` worker-seconds; enclosing wall and
  CPU stayed within run noise.
- Checkpoint 92 is retained as a small exact simplification.

## Checkpoint 93: nearest-table peak axial Gaussian

- Extended checkpoint 90's tested lookup to axial peak preparation after the
  exact support cutoff, removing roughly 199 million remaining peak-search
  exponential evaluations in the canonical run.
- Peak work improved `15.380 -> 14.589` worker-seconds, anchor CPU
  `48.111 -> 47.480` seconds, and wall `3.61 -> 3.56` seconds.
- Anchors and failures were unchanged. Accepted fiberlets changed by three;
  greedy routes were identical and fiberlet p95/max displacement was
  `0.00437/0.00895` base voxels. Checkpoint 93 is retained.

## Checkpoint 94: quantify peak radial survivors

- Profile schema 27 records exact radial-cutoff survivors with one local hot-
  loop increment. The canonical run found `663893054/2367683884`, or `28.0%`.
- The replay artifact was byte-identical to checkpoint 93 and measured overhead
  was negligible. The diagnostic is retained.

## Checkpoint 95: peak-plane CSR broad phase

- Reordered peak response records into a bounded 2D CSR grid and visited only
  buckets intersecting each response's cutoff square.
- Indexed visits fell 45.5%, but peak work regressed `14.603 -> 15.118`
  worker-seconds and wall changed `3.61 -> 3.62` seconds. Route points were
  unchanged.
- Checkpoint 95 is rejected. CSR construction and fragmented traversal cost
  more than the skipped arithmetic; the index and its schema field are removed.

## Checkpoint 96: one-axis contiguous peak buckets

- Stable counting-reordered response records by one transverse coordinate so
  each response traversed one contiguous cutoff range.
- Examined records fell 30.4%, but peak work regressed
  `14.603 -> 15.353` worker-seconds. Emitted route points were unchanged.
- Checkpoint 96 is rejected. Even contiguous bucketing is not amortized by the
  small number of responses per component; all candidate code is removed.

## Checkpoint 97: sparse retained centroid traversal

- Built component-local retained logical-index lists during the existing
  compact cutoff pass and traversed them directly for centroid accumulation.
  Expanded/public fitting retains its full defensive scan.
- Actual centroid visits fell `1608838400 -> 19153533` and centroid work
  `1.492 -> 0.536` worker-seconds. Anchor CPU improved
  `47.615 -> 46.606` seconds; command wall measured `3.58` seconds.
- The replay artifact remained byte-identical. Checkpoint 97 is retained and
  worst-case index storage is covered by worker admission.

## Checkpoint 98: membership-first peak evidence preparation

- Moved direction validation and alignment behind the existing retained-
  assignment test during peak record preparation. Denominator geometry and all
  evidence order/equations remain unchanged.
- Peak work improved `14.603 -> 12.873` worker-seconds, anchor CPU
  `46.606 -> 45.004` seconds, anchor wall `1.975 -> 1.906` seconds, and command
  wall `3.58 -> 3.49` seconds.
- The replay artifact remained byte-identical. Checkpoint 98 is retained.

## Checkpoint 99: reuse refinement support bounds for peak ownership

- Carried the exact six support-bound floats already computed during refinement
  into peak ownership instead of rescanning the complete support.
- Peak work improved `12.873 -> 11.833` worker-seconds and anchor CPU
  `45.004 -> 44.014` seconds. Anchor wall stayed `1.906` seconds; command wall
  measured `3.51` seconds while the unrelated fiberlet phase was slower.
- The canonical artifact remained byte-identical. Checkpoint 99 is retained.

## Checkpoint 100: finite compact positions in robust preparation

- Reused the generated compact range's finite-position invariant while
  collecting observed bounds and proposal records, removing two repeated
  finite checks per compact observation. Expanded/public inputs stay defensive.
- Preparation improved `3.952 -> 3.527` worker-seconds, local refinement
  `14.990 -> 14.534`, anchor CPU `44.014 -> 43.231` seconds, and wall
  `3.51 -> 3.46` seconds.
- The canonical artifact remained byte-identical. Checkpoint 100 is retained.

## Checkpoint 101: carry known compact support bounds

- Derived full-halo observation bounds from the fixed support stencil once and
  carried them through the indexed compact range. Clipped/general ranges still
  derive bounds from their actual indices.
- Preparation improved `3.527 -> 3.105` worker-seconds, local refinement
  `14.534 -> 14.148`, anchor wall `1.882 -> 1.850`, and command wall
  `3.46 -> 3.45` seconds. Anchor CPU measured `43.036` seconds.
- The canonical artifact remained byte-identical. Checkpoint 101 is retained.

## Checkpoint 102: cache compact direction eligibility

- Computed the complete configured compact direction/presence eligibility
  predicate once per unique shared observation and stored it in existing record
  padding. Reused it across all compact fitting consumers; arbitrary expanded
  observations retain defensive validation.
- Preparation improved `3.105 -> 2.065` worker-seconds, local refinement
  `14.148 -> 12.997`, anchor CPU `43.036 -> 41.942` seconds, and command wall
  `3.45 -> 3.44` seconds.
- The canonical artifact remained byte-identical. Checkpoint 102 is retained.

## Checkpoint 103: split peak denominator and evidence traversal

- Removed the dense evidence-index stream from peak search. Dense 12-byte
  response records now serve denominator traversal, while sparse self-contained
  32-byte evidence records carry the geometry needed by numerator and gradient
  traversal in original observation order.
- Peak work improved `11.805 -> 11.586` worker-seconds, anchor wall
  `1.850 -> 1.804` seconds, command wall `3.44 -> 3.40` seconds, and anchor CPU
  `41.942 -> 41.811` seconds.
- The canonical artifact remained byte-identical. Checkpoint 103 is retained;
  profile schema 29 removes the obsolete evidence-index-size field.
