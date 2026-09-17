# Fiberlet graph retuning log

Task started 2026-09-16. Locked model is tracing-best refinement EMA999 u22000,
SHA `67edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742`.

Existing comparable short200 results:

- Native fiberlet graph, production anchors, beam16/lookahead384/step48:
  67 failures, 9.736615mm mean, 1.473118mm median, 0.144118mm P05,
  7.847040mm P95 over 662.089848 directedmm/200 cases.
- Regular u22000 beam tracer:62 failures,10.509363mm mean,1.650440mm median,
  0.161748mm P05,7.907840mm P95 on the identical benchmark.

The previous weighted-anchor mini5 run retained all5 failures and did not improve
the downstream result. This task therefore holds production anchor extraction
fixed and retunes graph selection only. Initial exposed controls are beam width,
lookahead distance, and beam decision step. Native graph code also has stepped
cost/search/pruning controls; expose/test those only if the primary matrix does
not produce a useful candidate.

All commands requiring authorization will use existing pre-approved runners only.
Read-only inspection and repository edits do not introduce executable entrypoints.

Independent review required corrections before implementation. The authoritative
mini5 is the corrected v3 frozen manifest at
`.../mini5_corrected_weighted_i4_vis_20260916_v3_inputs/manifest.json`, SHA
`ed031e2f07dafa32b0393186ae531422aee0909898d1e1f621fba69301ddddbe`,
5cases/16.8873457254mm. It is used directly without regenerating corrections.
Short200 SHA is `7c6911dc...`,200cases/662.089848261mm; its pieces000/001 remain
the original shorter inputs, so corrected mini5 is deliberately not a byte-identical
subset. The detailed plan now fixes the staged matrix, deterministic promotion,
attempt-preserving resume, provenance, cache warmup, and test coverage.

Implemented `fiberlet_parameter_sweep.py` behind the existing approved
`benchmark-full` entry point. It rejects weighted/experimental anchor settings,
requires explicit 8-base-lattice graph parameters, preserves numbered attempts,
atomically publishes completed rows, validates model/manifest/native/ordered-matrix
identity on resume, and records per-case first-failure diagnostics. A repeated
stage-3 invocation validated and skipped every completed candidate.

Approved regression runner completed successfully: native suites passed 23, 85,
and 17 cases; Python inspection suite passed 269 with 8 optional skips.

Mini5 graph search completed in three stages: 24 beam16 step/lookahead settings,
9 beam-width follow-ups, and 10 local-neighborhood settings, plus same-matrix
legacy baseline checks and excluded prewarm rows. All 42 completed unique settings
retained 5 failures and 100% coverage. `b16_s016_l768` alone hit the 180-second
cap and was retained as a failed attempt. The largest first-failure-total gain
was only +22.762 base voxels (`b16_s048_l096`) with a -3.089-base worst
individual case, far below the +160-base promotion gate. Beam widths 8/32/64 and
the local grid did not remove a failure.

Decision: keep production beam16/lookahead384/step48. Do not run short200 because
no candidate met the predeclared mini5 promotion condition. Historical comparable
short200 values remain 67 failures/9.736615mm for fiberlets versus 62
failures/10.509363mm for the regular u22000 beam tracer. Full matrix, artifacts,
commands and interpretation are recorded in
`docs/benchmarks/fiberlet_graph_retuning_u22000_20260917.md`.

Completion audit found that the first matrix tuned graph replay beam controls,
but the user specifically called out the fiberlet DP step size in the context of
sharper predictions. Native `FiberletPathConfig` separately controls the actual
construction lattice: longitudinal step 2 prediction voxels, transverse step
0.5, default corridor radius one 4-voxel anchor cell, direction deviation 25
degrees, and local presence/direction/smoothness costs. None of these were
varied. The prior “complete” conclusion was therefore too narrow. Reopened the
same task to expose and tune construction DP before deciding whether fiberlets
can improve on current-best data.

Independent continuation review approved the scope after four constraints were
made explicit: only the twelve persisted semantic path fields are configurable;
fine-lattice follow-ups use deterministic top-two selection; raw corridor radius
zero remains the backward-compatible effective-four baseline while explicit four
is excluded as a duplicate identity; and each distinct path identity currently
duplicates the common anchor cache as well as constructing new fiberlets. These
constraints are now part of the task plan.

Implemented strict native `fiberlet.path` parsing into the canonical
`FiberletPathConfig`, excluding operational batching/thread controls. Added a
separate resumable `fiberlet_path_sweep` mode without changing the archived graph
sweep identity. Partial candidates resolve to the complete native semantic
default, duplicate identities and the radius-four alias are rejected, and each
row persists the fully resolved construction setting.

Construction-DP lattice stage1 completed 10 measured settings plus prewarm on
corrected mini5. Every setting retained 5 failures; production 2/0.5 remained
best at 100% coverage and 6747.054 summed first-failure base voxels. The
deterministic top two distinct longitudinal values are 2 (2/0.5=6747.054;
2/1=6745.975) and 4 (4/1=6746.110; 4/0.5=6704.727). Stage2 is therefore fixed,
before execution, to transverse 0.25 at longitudinal 2 and 4, plus a same-matrix
production prewarm/baseline. No other fine-lattice settings will be added.

Stage2 found one positive upper-bound signal: longitudinal4/transverse0.25 kept
5 failures and 100% coverage, and improved every individual case for a summed
first-failure gain of +157.169 base voxels (6904.223 versus 6747.054). It misses
the predeclared +160 promotion gate by 2.831 base voxels and is not promoted yet.
The next matrix is fixed before execution to 7 measured settings around 4/0.25:
its raw-radius0/deviation25 baseline; corridor radii 2,3,6; and prediction
deviations 15,20,35 degrees. Explicit radius4 remains excluded as the native
radius0 alias. A separate scoring stage, if warranted, is fixed to six one-axis
changes around the best reach setting: invalid cost 2/8, total smoothness 1/4,
and tangent smoothness 5/20. Candidate selection for that stage is failures,
coverage, summed first-failure progress, then declared order; one reach setting
will be selected.

Stage3 reach results selected the unchanged 4/0.25 setting: corridor2,
corridor3, corridor6, deviation15, and deviation20 were metric-identical to the
6904.223 baseline; deviation35 scored 6903.932 and was slower. Stage4 therefore
uses the already-declared six one-axis local-cost candidates around unchanged
4/0.25, plus same-matrix prewarm/baseline. No adaptive additions will be made.

Stage4 completed. Invalid-prediction cost 2/8, total smoothness 1/4, and tangent
smoothness 5 were identical to the 4/0.25 baseline; tangent20 scored 6903.932,
0.291 base voxel worse. Across all four construction matrices, no setting removed
a failure. The best was unchanged 4/0.25 with +157.169 summed first-failure base
voxels and no per-case regression, but it missed the locked +160 gate by 2.831
and took 364.975s cold versus 14.645s for cached production replay. It was not
promoted to short200. Production path defaults and the historical short200
comparison (fiberlet67 failures/9.736615mm versus regular u22000 62/10.509363mm)
remain authoritative. Full results are in
`docs/benchmarks/fiberlet_construction_dp_retuning_u22000_20260917.md`.

Final validation: approved native build succeeded; native suites passed
23/85/17 and the Python/replay suite passed 270 with 8 optional skips. A second
approved invocation of stage4 validated provenance and skipped all eight
completed rows without recomputation. No unapproved executable command was used.
