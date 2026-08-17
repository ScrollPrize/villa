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
