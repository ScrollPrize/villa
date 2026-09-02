# Task log: oracle winding inlier search

## Baseline

- Parent working state: uncommitted `conditioned-inliers` implementation on
  revision `4d2c6c51cbc5a43adbdaa194c079117313bb31b6`.
- Untouched 1024 solve: 16 exact, 9 wrong, 1 missing; 74.829% agreement.
- Sign-consistent direct inliers: 1054/1360 retained pieces, 20 exact, 5
  wrong, 1 missing; 84.621% agreement and zero admitted sign conflicts.
- Fresh solve on that graph: 13 exact, 12 wrong, 1 missing; 70.023% agreement.

## Decisions

- Use the canonical calibrated reference estimator as the oracle objective.
- Never replace an exact reference with a wrong or missing one to improve the
  aggregate count.
- Re-solve the fixed-reference problem after accepted removal batches.
- Treat sign evidence as authoritative and magnitude evidence as tunable
  ranking support.

## Plan review

- Freeze the initially supported reference set; otherwise wrong-to-missing
  deletion can look like an improvement. Rank missing before wrong after exact.
- Counterfactual observation deletion is only a proposal. Every accepted batch
  requires a converged conditioned re-solve and realized benchmark improvement.
- Reapply sign-consistent closure after re-solving because new Defects or sign
  conflicts may appear.
- Maintain one original-piece active mask across rounds and rebuild all ordinary
  and cross subsets from it; never reuse stale local indices.
- Keep direct and fresh artifacts distinct and benchmark both through induced
  cross constraints without reading fixed reference states.
- If direct reference-adjacent candidates stall, a bounded ordinary-factor
  neighborhood is required before declaring that no improving candidate exists.

## Implementation notes

- Added `oracle-inliers` as a separate supervised pruning policy.
- The core scorer reuses canonical reference calibration and freezes the
  initial reference universe. Candidate selection tests singles, bounded
  pairs, aggregate wrong-label advantage, and calibrated raw-candidate
  residuals.
- The CLI always rebuilds ordinary and reference-cross subsets from original
  piece IDs, runs a fresh fixed-reference conditioned solve, reapplies exact
  sign closure, and rolls back regressions.
- Neutral removal rounds are allowed because collective false support can
  remain unchanged until several pieces have been peeled.
- Removed pieces publish as `_oracle`; a separate reference-free solve
  publishes as `_oracle_fresh`.

## Experiments

- Command: `/tmp/vc_direction_ablation_runner.sh reference-prune oracle 1`.
- Dataset: 1024 crop, 500/1998 quality-filtered input fibers, 1360 pieces,
  26 reference fibers from `2026-09-01_fiber_stack2`.
- Initial sign-consistent state: 1030 pieces, 20 exact, 5 wrong, 0 missing in
  the frozen 25-reference universe.
- Accepted trajectory: `20/5/0 -> 20/4/1 -> 21/3/1 -> 22/2/1 -> 23/1/1 ->
  23/1/1 -> 24/0/1` for exact/wrong/missing.
- Final frozen objective: 997/1360 pieces retained, 24 exact, 0 wrong, 1
  missing, 7009/8090 reference constraints correct, terminal
  `zero_wrong_with_missing`.
- Full 26-reference report: 24 exact, 0 wrong, 2 missing. Fresh reference-free
  report: 12 exact, 11 wrong, 3 missing.
- Timing: 57.74 s wall, 1245.97 s user, 3.53 s system, Release build.

### 2048-crop validation

- Dataset: `fiber-crop-2048/crop_traces.zarr`, 500/1999 quality-filtered
  input fibers, 2524 pieces after main-component filtering, and the same 26
  reference fibers.
- Strict runs at 500 and 2000 messages did not produce a converged conditioned
  state. Lowering damping from 0.5 to 0.25 also failed to converge.
- Diagnostic run used `--reference-oracle-accept-message-limit` at the normal
  500-message limit. The initial sign-consistent closure retained 1842 pieces
  and scored 25 exact, 1 wrong, 0 missing.
- Accepted oracle trajectory: `25/1/0 -> 25/1/0 -> 26/0/0` for
  exact/wrong/missing, ending with 1838/2524 pieces retained and terminal
  `zero_errors`. All three conditioned rounds had status `message_limit`.
- Direct conditioned constraint agreement was 9437/11254 (83.855%), with zero
  retained perpendicular- or parallel-sign errors.
- The reference-free solve on the retained graph remained poor: 11 exact, 15
  wrong, 0 missing and 6090/10078 (60.429%) constraint agreement. This confirms
  that the 26/26 result is supervised and does not make the retained graph
  independently self-solving.
- Timing: 154.19 s wall, 3851.07 s user, 5.02 s system, Release build. Full log:
  `/tmp/reference-prune-2048.log`.

## Validation

- GCC Release `test_fiber_trace_winding_bp`: 95 cases passed.
- Clang `test_fiber_trace_winding_bp`: 95 cases passed.
- `test_view_fiber_windings.py`: 41 cases passed with external pytest plugin
  autoload disabled.
- Full 1024 oracle diagnostic completed and refreshed `_oracle` and
  `_oracle_fresh` artifacts.

## Deviations and limitations

- The final reference at virtual winding 4.0 is missing. Its retained
  observations supported winding 0 rather than winding 4; deletion cannot
  create absent positive support, so the oracle removes the false support and
  reports the reference as missing.
- The direct conditioned result is supervised. The fresh reference-free solve
  remains unstable and is diagnostic only.
- Graph-neighbor alternatives did not improve the 23/25 plateau; collective
  raw-candidate residual peeling produced the final improvement.
