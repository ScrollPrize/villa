# Seven intact GT fibers: shared-phase correction

Corrected coarse event difference:
`kB - kA + fixed_event_shift + learned_phase_B - learned_phase_A + (familyB-familyA)/2`.
The fixed shift already represents original anchor fractions and original
along-fiber turns. Fine per-hypothesis offsets and fitted widths are excluded.
The same corrected matrix drives distance loss, sign hinge, fixed sign charge,
and the saved-state sign diagnostic. Shared phases receive gradients.

Opt-in `bucket_shared_phase=true` in run_bbox_gt_v3.json; old saved configurations
retain their historical meaning. Not yet a change to the corpus baseline config.

| Metric | Previous v2 | Corrected v3 |
|---|---:|---:|
| Retained whole fibers | 7/7 | 7/7 |
| Retained constraints | 46/46 | 46/46 |
| Coarse sign violations | 5 (10.8696%) | 0 (0%) |
| Fitted-position sign violations | 0 | 0 |
| Mean absolute winding-distance mismatch | 0.983133 | 0.206087 |
| Final online objective (last80 updates) | 256.371918 | 0.214366 |
| Solve seconds | 127.48 | 128.33 |

Both: fresh seed0,151hypotheses,CCW,10080full-passupdates,same LR/loss weights,
one managed nonexclusive GPU, no splits, no rejection. Objective formula changed
by correction; mismatch compares against the same extracted distance evidence,
not independent ground-truth winding accuracy. V3 reached zero coarse violations
at1000 and retained zero at every later logged checkpoint.

V5–V6 parallel separation: previously coarse4turns; corrected2.10465turns.
Fitted-position separation~2.1573turns; measured1.956–2.106turns.

Approved runner:
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_bbox_gt_v3.json
```
Same entrypoint with run_bbox_gt_v3_vis.json renders TIFF/SVG/PDF.
Artifacts: data/bbox_gt_solve_v3/report.json, uv_whole/{solution.npz,summary.json,history.json},
vis/seven_fibers/layout.{tif,svg,pdf}.

Independent plan/code review passed.25CPUtests passed (adapter,bucket split,sign
charge), including differentiable phase, common-shift invariance and missing-state
guard; native feature/prior tests and CUDA coordinate checks passed.
The compiled objective uses automatic differentiation for the dynamic shift.
No input fibers changed. Cross-solution scoring remains the separate pending task.
