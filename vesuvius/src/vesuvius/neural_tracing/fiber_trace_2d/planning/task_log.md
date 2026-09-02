# Task log: iterative ordered-winding offender removal

## 2026-09-02

- Started from the uncommitted ordered-cut solver implementation after the
  `piece-length=384` reference-oracle run selected 19 exact, 6 wrong, and 1
  missing reference at split 27.
- The ranking unit is a complete source trace. This prevents partial removal
  from hiding the observed within-fiber jumping problem.
- A violated cross-trace factor counts once against both endpoint traces; a
  within-trace factor counts once. Selection uses exact integer cross-products
  for percentages to avoid floating-point tie ambiguity.
- The diagnostic will be opt-in because repeated Ceres solves may be expensive
  and because removing all conflict participants is not yet a production
  defect model.
- Independent review required a separate active mask, comparable surviving-edge
  before/after counts, explicit failed/empty-solve handling, overflow-safe exact
  percentage comparisons, and stronger proof that every iteration really
  re-solves. The implementation plan was updated before code changes.
- Implemented reusable whole-source trace infringement aggregation and exact
  overflow-safe percentage ranking. Ties use violated count and then current
  trace index; current-to-original trace mapping is printed by the CLI.
- Added `--ordered-prune-offenders`. Each progress row reports the selected
  trace's percentage plus old-full, surviving-before-refit, and
  surviving-after-refit violations. Removed split pieces remain inactive for
  the final cut scan.
- GCC Release and Clang Debug builds completed. Both
  `test_fiber_trace_winding_bp` binaries passed all 89 test cases.
- Release validation command:

  ```bash
  /tmp/vc_direction_ablation_runner.sh ordered-prune
  ```

  Dataset: 1024 crop, 25% quality cohort, piece length 384, complete
  `2026-09-01_fiber_stack2` references. Initial graph: 499 source traces, 1925
  pieces, and 94699 admitted continuous sign factors. The diagnostic removed
  122 traces and reduced violations from `954/94699` (1.01%) to `0/54154`.
  Wall/user/system times were 59.00/1018.19/4.66 seconds.
- The reference-oracle discrete checkpoint after pruning had 16 exact, 8 wrong,
  and 2 unsupported reference windings with `8923/12398` (71.97%) correct
  reference constraints. The unpruned piece-length-384 result was 19 exact, 6
  wrong, and 1 unsupported. Iteratively eliminating every continuous conflict
  is therefore too aggressive and remains explicitly diagnostic.
