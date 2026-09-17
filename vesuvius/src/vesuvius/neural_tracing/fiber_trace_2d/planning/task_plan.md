# Plan

1. Lock the comparison to EMA999 u22000 (`67edd3a1...`), corrected mini5 v3
   manifest SHA `ed031e2f...` (5 cases/16.887346mm), and short200 manifest SHA
   `7c6911dc...` (200 cases/662.089848mm). Keep the established 40/80
   ellipse, 160-base grace, and 80-base restart grouping unchanged.
2. Re-establish the native fiberlet baseline with legacy production anchors and
   graph settings beam16/lookahead384/step48. Do not mix the unsuccessful weighted
   anchor experiment into the graph-parameter comparison.
3. Add a resumable matrix mode behind the existing approved evaluation runner.
   Bind matrix provenance to manifest/model/native hashes and the full ordered
   candidate list. Every candidate gets immutable attempt directories plus an
   atomic completed-result pointer; an interrupted attempt is retained and a new
   numbered attempt is started. Validate completed candidates before skipping.
   Every candidate reports total
   restarts, coverage, mean/median/P05/P95 segment length, per-case first failure,
   and wall time.
4. On mini5, search in stages rather than an uncontrolled Cartesian explosion.
   Stage1 is beam16 with step `{16,24,32,48,64,96}` and lookahead
   `{96,192,384,768}` base voxels (multiples of the8-base prediction lattice,
   lookahead >= step):24 candidates. Stage2 takes the best3 distinct step/lookahead
   pairs and adds beam widths `{8,32,64}` (beam16 is already measured). Stage3
   takes the best2 and tests valid step neighbors `s±8` and lookahead neighbors
   `l±64`, deduplicating every earlier candidate. Cap each candidate at the
   approved runner timeout and retain failed/capped attempts. “Retain choices”
   means a cumulative result set, not monotone output from wider searches.
5. Promote only candidates that materially improve mini5 (fewer failures, or a
   deterministic later-first-failure shift without coverage loss) to the 200-break
   set. A no-failure case maps to its reference end. At equal failures, require
   coverage within0.01percentage point of baseline, total first-failure gain at
   least160base voxels, and no individual case regression over64base voxels.
   Always rerun the old fiberlet settings in the same matrix for a fair baseline.
6. Compare the selected fiberlet result with both the prior 67-failure fiberlet
   baseline and the 62-failure regular u22000 beam result. The same-matrix legacy
   rerun is authoritative: rank fewer failures first, reject coverage loss over
   0.01percentage point, then compare failures/mm, mean/quantiles/per-scroll and
   runtime. Historical results are references only. Record failures,
   failures/mm, mean distance, quantiles, coverage, and per-scroll behavior.
7. Archive or promote the tuning based on the 200-break result, document all
   tested settings/results, update changelog/status/log, validate with the
   pre-approved native replay test runner, and commit.

## Spec update

No tracing/evaluation protocol change. If a fiberlet parameter set wins on the
200-break benchmark, document it as the current downstream fiberlet default;
otherwise retain the existing default and archive the sweep.

## Docs updates

Add a benchmark report containing the complete mini5 matrix, promoted 200-case
results, exact manifests/checkpoint/commands, and interpretation. Update
`docs/fiberlet_evaluation.md` only if a new default is promoted.

## Testing

Use only the existing approved entry points:

- `run_fiber3d_native_build` if native changes become necessary;
- `run_fiber3d_native_replay_tests` for regression tests;
- `run_classic_fiber_evaluation.py benchmark-full --config ...` for all mini5
  and short200 runs.

Validate immutable identities, matrix resumption, metric recomputation, equal
case counts/reference lengths, and no coverage regression. Unit coverage includes
ordered variants, legacy/no-anchor-fit binding, interrupted-attempt recovery,
completed-row validation/provenance rejection, deterministic ranking, and
per-case metric extraction. Prewarm production graph chunks once, then repeat the
legacy row after warmup; report end-to-end times as cache-contaminated otherwise.
