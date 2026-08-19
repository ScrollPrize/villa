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
