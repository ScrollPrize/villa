# Seven reviewed fibers: expanded solve audit

Inputs: all seven unchanged JSONs in data/bbox_gt. Output:
data/bbox_gt_solve_v2. Previous four-fiber artifacts remain untouched.

Run (approved entrypoint):
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_bbox_gt_v2.json
```
Visualization uses the same command with run_bbox_gt_v2_vis.json.

All seven retained by preprocessing: two H, five V. 46 events (40 perpendicular,
six parallel); no isolated fibers. UV: seven intact pieces, zero splits or
continuations, structurally zero rejection probability. 10080 full-batch updates,
127.48 seconds, same canonical loss/LR settings as the four-fiber pilot.

| Audit | Result |
|---|---:|
| Retained fibers/events | 7/7, 46/46 |
| Bucket-sign violations | 5/46 (10.8696%) |
| Fitted-position sign violations | 0/46 (0%) |
| Mean bucket-distance error against extracted measurements | 0.983133 turns |
| Final online objective, last 80 updates | 256.371918 |

The five violated bucket events are IDs7,12,13,30,31, involving pairs1–4,1–5,
2–5. Shared phase shifts allow positional signs to agree while discrete bucket
choices disagree. This is NOT a validated new winding GT, and no cross-solution
accuracy score has been produced. Do not substitute positional sign agreement
for the requested exact induced-winding accuracy.

| Fiber suffix | Family | Bucket | Actual anchor q | Actual endpoint q |
|---|---|---:|---:|---:|
| 000001 | H | 19 | 22.33729 | 15.17953 |
| 000002 | H | 11 | 13.14771 | 22.15005 |
| 000003 | V | 14 | 16.90809 | 16.90809 |
| 000004 | V | 15 | 15.96163 | 15.96163 |
| 000005 | V | 13 | 17.98786 | 17.98786 |
| 000006 | V | 17 | 20.16003 | 20.16003 |
| 000007 | V | 20 | 21.65120 | 21.65120 |

Absolute origin remains arbitrary. Detailed report: data/bbox_gt_solve_v2/report.json.
TIFF/SVG/PDF: data/bbox_gt_solve_v2/vis/seven_fibers/layout.*.

Validation: 12 adapter/bucket tests passed; native feature/prior tests passed;
CUDA coordinate value/gradient/repeat checks passed. Two new JSONs lack shape
metadata; importer verifies named scan provenance and bounds instead, rejects
explicit mismatching shapes, never changes source files.

Pending: confirm two comparison artifact identities and disabled-piece score
convention (questions already sent); resolve GT bucket/position discrepancy
before freezing targets or interpreting comparisons. Plan bbox_gt_eval_plan.md.
