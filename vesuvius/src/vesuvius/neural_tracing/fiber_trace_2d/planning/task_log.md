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
