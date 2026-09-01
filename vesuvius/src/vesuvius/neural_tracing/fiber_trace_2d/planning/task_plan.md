# Plan: retune hard continuation and alignment falloff

## Experiment contract

1. Keep `--split-continuity hard`, fixed orientation, phase `0.5`, scale
   `0.822`, 512-base-voxel pieces, quality fraction `0.25`, both sign classes,
   parallel cutoff `0.5`, and 500 maximum winding messages in every scenario.
2. Treat `normal=none`, `normal=linear`, and `normal=cosine` as three separate
   model families. Give each family its own complete refinement from the same
   current defaults; never transfer a selected scalar/weight tuple and call the
   receiving family tuned.
3. Tune and compare on `data/workdir3/crop_traces.zarr` (1024). Do not run the
   2048 crop in this task; larger-context validation is explicitly deferred.
4. Establish one fixed pre-solve eligible-reference denominator and retained
   piece population from the common 1024 extraction. Report right, wrong,
   neutralized/abstained, and missing observations against that fixed
   denominator for every candidate. A candidate is ranking-eligible only when
   it evaluates at least 90% of eligible reference observations and leaves at
   least 90% of retained pieces active. This coverage gate is declared before
   the search and must not move in response to results.
5. Rank coverage-eligible deterministic tuning results lexicographically by:
   - converged before message-limit;
   - more exact reference windings;
   - fewer missing, then fewer wrong reference windings;
   - more right reference constraints against the fixed denominator;
   - more evaluated reference constraints and active pieces, then fewer wrong
     constraints;
   - lower continuation and aggregate active-constraint infringement rates.
   Exact metric ties retain the earlier tuple instead of optimizing numerical
   message-residual noise.
   Coverage-ineligible rows remain in the report but cannot be selected.

## Per-family refinement

1. Establish the family baseline with current class weights `8,1,2,2,1`, sign
   cost `44`, Defect cost `100`, BP temperature `1.25`, decision confidence
   `legacy`, and hard-sign angle 30 degrees.
2. Record the mandatory hard/30-degree controlled anchor before tuning.
3. Coarsely scan decision mode `legacy,linear,cosine`, holding the other
   starting parameters fixed. Retain the best deterministic row for that family.
4. Refine one complete tuple with bounded deterministic best-improvement
   coordinate descent. Evaluate blocks in this fixed order: decision mode,
   five class weights, sign cost, Defect cost, then BP temperature. After
   any accepted move, begin a new complete block pass. Stop only after one full
   pass accepts no move.
5. Use finite absolute candidate sets:
   - decision modes from step 3;
   - each class weight at zero or original-seed times powers of two with
     exponents `[-4,+4]`; a zero coordinate reactivates at its original seed;
   - sign cost `0` or `44 * 2^e`, `e in [-3,+3]`;
   - Defect cost `100 * 2^e`, `e in [-3,+3]`;
   - BP temperature `1.25 * 2^e`, `e in [-2,+2]`.
   Cache every complete tuple globally per family, evaluate neighbors in the
   stated order, select the deterministic best improvement rather than the
   first, and fail at 500 unique scenarios per family instead of silently
   returning a partial optimum. The zero-weight cases cover the previously
   promising disabled far-magnitude factors; the built-in positive-only local
   weight search is insufficient by itself.
6. This remains path-dependent local optimization from the declared common
   seed after the coarse structural scan; do not claim a global optimum.
7. Repeat the selected 1024 row once and require identical solver/reference
   metrics before treating the family as tuned.

## Family comparison

1. Freeze all three independently selected family tuples on the 1024 crop.
2. Also retain the common hard/30-degree current-default row as the untuned anchor.
3. Compare convergence, exact references, right/evaluated reference
   constraints, active/Defect pieces, continuation and total infringements,
   winding solve seconds, and runner wall seconds on the 1024 crop.
4. Repeat the three selected tuples three times in rotated order.
   Require deterministic quality metrics and report winding-solve and runner
   wall min/median/max. Timing repetitions are distinct from the earlier
   quality-determinism repeat.
5. Leave final 1024 visualization artifacts for the selected families under
   the tuning output root; retain all logs under `/tmp`.

## Validation

- Use the Release `vc_fiber_trace_chunk` binary and the previously approved
  `/tmp/vc_direction_ablation_runner.sh` command surface.
- Record exact commands, revision, inputs, build type, and all accepted/rejected
  results in `planning/task_log.md`.
- Log the complete tuple and fixed-orientation prepass output for every
  candidate. `--bp-temperature` controls the H/V/Mixed prepass and scales the
  winding Defect unary; do not describe it as only a winding temperature.
- If the experiment requires code changes, rebuild the CLI and run the focused
  winding/crop tests plus `git diff --check`; otherwise state explicitly that
  the task changed only experiment records.

## Spec Update

No solver-semantics change is planned. If a tuned result is later promoted to a
default, update `planning/specs.md` with the selected complete tuple; otherwise
leave the specification unchanged.

## Docs Update

Record the reproducible tuning protocol and results in
`volume-cartographer/docs/fiber_chunk_tracing.md` only if they establish a new
recommended tuple. Do not present an unvalidated 1024-only optimum as a general
default.

## Changelog Update

Add a concise tuning-result entry after held-out validation. Mention a default
change only if one is actually selected after the comparison.
