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

## Continuation: construction-DP tuning

The completed graph-replay matrix was a negative control, not completion of the
requested DP retuning. The sharper prediction principally changes the discrete
DP that constructs each fiberlet between anchors. Continue as follows:

1. Expose a strict nested `fiberlet.path` configuration in the native evaluation
   CLI for the twelve persisted semantic `FiberletPathConfig` fields: cell
   radius, neighborhood margin, longitudinal/transverse step, endpoint/prediction
   angles, corridor radius, invalid-prediction cost, and the four smoothness
   fields. Never expose the execution-only sampling-batch or thread fields. Feed
   that same object into metadata/fingerprint generation, on-demand construction,
   and cached replay; do not introduce a second implementation or bypass
   canonical construction.
2. Extend the resumable approved-runner sweep machinery with a path-parameter
   mode while preserving the already-published graph-sweep identity. Resolve
   partial candidate overrides against native defaults, hash the complete
   resolved setting, retain immutable attempts, and reject duplicate settings.
3. Add native/CLI regression coverage proving that path settings reach persisted
   construction metadata, alter construction cache identity, and remain separate
   from replay-only settings. Add Python coverage for validation, duplicate
   detection, resumption, and dispatch.
4. Run a corrected-mini5 lattice stage using production anchors and unchanged
   graph replay: longitudinal step `{1,1.5,2,3,4}` prediction voxels crossed with
   transverse step `{0.5,1}`. Then test transverse step `0.25` at the best two
   distinct longitudinal steps, chosen deterministically by failures, coverage,
   summed first-failure progress, and finally declared candidate order. Include
   explicit same-run baseline/prewarm and candidate caps. Rank failures first,
   then coverage and per-case first-failure progress.
5. If a lattice candidate is credible, refine corridor radius and the local
   direction/smoothness balance around it. If no lattice point removes a failure,
   still test a bounded, documented second stage over corridor and prediction
   deviation because those directly control whether the sharper ridge remains
   reachable. Record its exact candidates and selection count in the task log and
   matrix provenance before starting it. Preserve raw corridor radius `0` as the
   backward-compatible baseline (it resolves to the 4-prediction-voxel anchor
   cell size), and exclude explicit radius `4` as a behaviorally duplicate but
   differently hashed identity. Avoid an uncontrolled Cartesian product.
6. Only then consider stepped cost-profile graph replay and bounded-search
   approximations. These are secondary because they select among constructed
   fiberlets; they cannot recover geometry excluded by construction DP.
7. Promote to short200 only for fewer failures, or the existing deterministic
   first-failure gate without coverage loss. On short200 compare against both the
   same-matrix production fiberlet baseline and the regular u22000 tracer. Retain
   production defaults if no setting passes.

## Spec update

Add an optional, strict `fiberlet.path` evaluation configuration that maps
one-to-one onto the existing native construction DP. No metric/protocol change.
If a fiberlet parameter set wins on the 200-break benchmark, document it as the
current downstream fiberlet default; otherwise retain the existing default and
archive the sweep.

## Docs updates

Extend the benchmark report with the complete construction-DP matrix, promoted
200-case results, exact manifests/checkpoint/commands, and interpretation.
Document the new `fiberlet.path` keys. Update `docs/fiberlet_evaluation.md` only
if a new default is promoted.

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

For construction-DP candidates, cache prewarming is per resolved path identity:
the first run includes construction and is the relevant end-to-end cost. Do not
claim replay-only timing parity. Use the approved native build and replay-test
runners after native parsing changes, and the approved benchmark runner for all
measurements.

The existing dataset fingerprint names a common anchor-and-fiberlet root, so each
distinct path identity currently duplicates the unchanged anchor cache as well as
constructing distinct fiberlets. This is accepted for the bounded sweep and must
be included in storage/runtime interpretation; do not claim cross-candidate anchor
reuse.
