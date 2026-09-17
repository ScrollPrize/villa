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
