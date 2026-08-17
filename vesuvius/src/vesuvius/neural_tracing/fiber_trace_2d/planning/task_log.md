# Task Log: robust sampled-direction anchor refinement

## 2026-08-17 Planning

- Direction predictions are sampled network outputs. Prior component axes may
  condition competitive assignment and robust outlier detection, but updated
  axes must be direct principal aggregates of retained current samples; angular
  interpolation is removed.
- Cells may contain up to two close directions. Hard assignment uses combined
  spatial Gaussian and projective direction agreement to prevent soft
  component collapse.
- Adaptive weighted median/MAD detection identifies inconsistent or blended
  predictions. Trimming is capped at 20% of presence/spatial mass and is not
  compulsory: coherent components retain all evidence.
- Components without a unique retained sampled-direction axis are removed, not
  assigned stale state.
- Backtracking changes transverse position only, tests no finer than the first
  candidate at or below the 0.5 prediction-voxel peak-grid spacing, and leaves
  finer positioning to the later peak stage.
- Exact old/new numeric identity is not an acceptance requirement. The plan
  requires deterministic new runs, geometric/population comparisons,
  downstream replay metrics, performance measurements, and user visual review.
- No fitting-code implementation is part of this planning-only turn.
- Independent review required final robust membership to govern centroid,
  spatial objective, peak search, and final support so trimmed samples cannot
  re-enter; explicit convergence after unconditional direction updates;
  propagation of component removal through all downstream loops/diagnostics;
  validation that trim configuration cannot exceed 20%; deterministic
  histogram edge semantics; profile schema version 4; and an explicit warmup
  plus `--threads 32` benchmark protocol. The plan was corrected accordingly.
- A second review found that the current pre-refinement 10-degree merge can
  erase close modes before robust competition. The plan now removes that merge,
  preserves supported close components through refinement, and relies on
  downstream NMS for duplicates. It also defines configurable trim quantiles,
  existing principal-axis degeneracy criteria, retained-only fitting/peak
  objectives versus all-site final support normalization, quantitative close
  mode/noise fixtures, and symmetric old/new benchmark warmups and repetitions.
- A blended prediction inside a coherent angular core is not identifiable from
  direction samples alone when the network provides no uncertainty output.
  Spatial competition and measured/visual quality checks mitigate but cannot
  eliminate that limitation.

## 2026-08-17 Implementation

- Implemented competitive assignment with
  `gaussian * presence * abs(dot)^2`, deterministic 256-bin weighted
  median/MAD trimming capped at 20% mass, and direct retained sampled-direction
  PCA. Axial sign remains irrelevant.
- Removed the pre-refinement merge path. Supported close components reach
  robust fitting independently; ordinary NMS handles duplicates. Legacy merge
  fields remain inert. New robust artifacts use schema version 2 while strict
  version-1 loading preserves the original parameter shape.
- Implemented position-only halving through the first displacement at or below
  the peak-grid step. Trimmed samples remain excluded from positive centroid,
  spatial, peak, and final-support evidence.
- Plan clarification after implementation review: spatial and peak denominators
  use every sampled lattice site independent of its presence/direction state,
  while only retained evidence enters the numerator. A retained-only denominator
  makes constant response position-independent. Excluding trimmed positive
  sites from the denominator creates geometric holes that attract a widened
  peak even though their numerator is zero; focused axial-membership tests
  caught this. Uniform all-site normalization avoids both failures.
- Plan deviation: exact assignment/inlier-set convergence was removed. Hard
  component and histogram boundaries flickered even when geometry was stable,
  causing thousands of cells to hit 64 passes. The new default is two bounded
  robust assignment/update passes, with the existing CLI retaining experimental
  control.
- Added profile schema version 4, robust CLI controls, strict parameter
  serialization/loading, stable diagnostic IDs through compaction, explicit
  degenerate transitions for removed components, and focused robust/close-mode
  tests.

## 2026-08-17 Validation So Far

- Regular GCC build:
  `cmake --build volume-cartographer/build --target vc_fiberlets test_fiber_anchors test_fiberlet_paths test_fiber_replay -j 8`.
- Performance GCC build:
  `cmake --build volume-cartographer/build/fiberlet-perf --target vc_fiberlets test_fiber_anchors test_fiberlet_paths test_fiber_replay -j 8`.
- Both build trees pass focused CTest for `test_fiber_anchors`,
  `test_fiberlet_paths`, and `test_fiber_replay`; the anchor suite has 58 test
  cases.
- The existing Clang `ci-fast-core` tree also builds and passes the same three
  tests; the final focused anchor rebuild is warning-free.
- Canonical 5,000-base-voxel exploratory results, 32 threads:

  | robust passes | total wall | anchor wall | anchors | greedy failures | fiberlet failures |
  |---:|---:|---:|---:|---:|---:|
  | 64 | 50.95 s | 45.87 s | 2324 | 2 | 1 |
  | 4 | 17.49 s | 12.57 s | 2298 | 2 | 1 |
  | 2 | 14.57 s | 9.69 s | 2315 | 2 | 1 |

- The measured two-pass run used
  `vc_fiberlets fiberlet-replay fiber_s1_002.lasagna.json dj_20260805T025256484_000003.json ... --normal-manifest las_008.lasagna.json --threads 32 --length 5000 --maximum-iterations 2`.
- Finished-code default runs, including same-cell mode preservation and final
  membership refresh, were fully deterministic:

  | run | wall | user CPU | system CPU | anchor wall | fiberlet wall |
  |---:|---:|---:|---:|---:|---:|
  | 1 | 29.14 s | 401.12 s | 4.16 s | 16.99 s | 10.83 s |
  | 2 | 29.15 s | 400.32 s | 3.95 s | 16.87 s | 11.21 s |
  | 3 | 29.11 s | 402.55 s | 4.07 s | 17.10 s | 10.92 s |

- All three produced 2516 anchors, 48,722 searched / 24,375 accepted fiberlets,
  170,498 sampled voxels, 47,710,210 evaluated DP nodes, 2 greedy failures,
  and 1 fiberlet failure. `fiber_replay.json` SHA-256 was
  `bf6d51fa9c357897536d820798b0b0ca5e72eac3f8b26581fde20db7677138dd`
  for every run.
- The final measurements ran while host load average exceeded 40. Their wall
  distribution is valid for repeatability but not an idle-host comparison to
  the earlier roughly 22-second baseline. The two-pass exploratory run before
  that load increase completed in 14.57 seconds, but it still used the old NMS
  behavior and is not a final apples-to-apples result.
- A subsequent idle-host canonical run (load average 0.16 at launch) completed
  in 18.17 seconds wall and 512.77 seconds CPU (509.30 user + 3.47 system),
  versus the immediately preceding implementation's 22.09-second median wall
  and 633.96-second median CPU. This is a 17.7% wall-time and 19.1% CPU-time
  reduction. Anchor extraction took 11.59 seconds and fiberlet extraction took
  5.95 seconds. A second repetition was excluded after host load rose to 40.7;
  it completed in 28.63 seconds.
- The idle-host run produced 2516 anchors, 48,722 searched / 24,375 accepted
  fiberlets, 170,498 sampled voxels, 47,710,224 evaluated DP nodes, 2 greedy
  failures, and 1 fiberlet failure. The small DP-node difference from the
  high-load measurements did not change the accepted population or replay
  failure counts.
- Preserving same-cell close modes and refreshing final memberships raises the
  canonical population from the exploratory 2315 anchors to 2516 and
  correspondingly increases downstream fiberlet work. Replay failure counts
  remain unchanged.
- The canonical workload removed no non-unique robust components. The removal,
  stable-compaction, and diagnostic paths are implemented and reviewed, but a
  production-level fixture that forces exactly one of two initialized
  components through that path remains a test limitation.
- User visual review remains pending.
- A separate `--vis` replay wrote three reloadable failure manifests under
  `/tmp/fiberlet-replay-robust-final-vis`: two greedy and one fiberlet view.

## 2026-08-17 Follow-up Anchor Profile

- Committed the robust sampled-direction checkpoint as `d89b0aba0`.
- The idle-host profile attributes 122.40 worker-seconds to local refinement:
  54.04 seconds to robust tensor proposals, 60.96 seconds to spatial state
  evaluation, 1.63 seconds to centroids, and 5.77 seconds to control overhead.
- Outside local refinement, the largest anchor costs are prediction sampling
  (81.86 worker-seconds), observation construction (65.93 worker-seconds), and
  peak search (47.29 worker-seconds).
- The first follow-up optimization will remove repeated normalization,
  Gaussian, residual, and component-scan work inside each robust proposal and
  omit the unused PCA tensor from final membership refresh. It intentionally
  does not change the estimator or spatial search.

## 2026-08-17 Follow-up Anchor Optimization

- Canonical command:
  `vc_fiberlets fiberlet-replay fiber_s1_002.lasagna.json dj_20260805T025256484_000003.json /tmp/fiberlet-replay --normal-manifest las_008.lasagna.json --threads 32 --length 5000 --maximum-iterations 2`.
- The committed `d89b0aba0` checkpoint completed in 18.17 seconds wall and
  512.77 seconds CPU. Anchor extraction used 11.59 seconds wall and 349.78
  seconds CPU.
- Fusing robust residual histograms with retained tensor accumulation, omitting
  the unused final-membership tensor, caching gradients once per tile, pairing
  baseline/first-candidate spatial objectives, and representing peak geometry
  in transverse 2D reduced the canonical run to 17.11 seconds wall and anchor
  extraction to 10.37 seconds with four-cell tiles.
- Six-cell tiles were the best measured balance between halo duplication and
  32-worker load balance: 16.96 seconds total wall, 461.19 seconds total CPU,
  10.34 seconds anchor wall, and 298.57 seconds anchor CPU. Submitted anchor
  samples fell from about 69.9 million to 39.7 million and physical gradient
  computations to 35.8 million.
- The final run produced 2520 anchors, 48,972 searched / 24,518 accepted
  fiberlets, 47,924,048 evaluated DP nodes, 2 greedy failures, and 1 fiberlet
  failure. Small population and DP differences are accepted for the regrouped
  floating-point reductions; visual quality review remains required.
- Eight-cell tiles reduced aggregate CPU but regressed wall time from poorer
  load balance. Sorting those tiles by descending size regressed further.
  Explicit Gaussian caches and a linked-bin peak broad phase also regressed and
  were removed; the latter cut response visits but lost locality. Replacing
  compensated hot-loop sums with ordinary double reductions was neutral to
  slightly slower (10.45 seconds anchor wall, 298.30 seconds anchor CPU) and
  was also removed.
- Converting only the transient peak-search record to float32 reduced the run
  to 16.57 seconds total wall, 10.05 seconds anchor wall, and 288.29 seconds
  anchor CPU. Extending float32 through radial distance, Gaussian, gradient,
  and vote calculations reduced it further to 16.37 seconds total wall, 9.76
  seconds anchor wall, and 277.38 seconds anchor CPU. Both runs retained 2520
  anchors, 24,518 accepted fiberlets, 2 greedy failures, and 1 fiberlet failure.
