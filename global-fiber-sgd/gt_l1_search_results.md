# Enabled GT constraint L1 refinement

Implementation f1bcf1e2f; previous results 016cdf185. Plan gt_l1_search_plan.md.
Frozen reviewed-fiber GT v3; same6735cross events. Minimize continuous winding
L1 of final enabled test-piece events only, equal weight/event, one L1-optimal
integer global correction. All types share this correction. No GT training
constraints; this is tuning, not held-out validation. Disabled pieces excluded
from objective but reported; changes in coverage can improve L1.

| Run | Enabled GT L1 | GT event coverage | Retained length | Spiral L1 | Width factor | Online loss | Solve s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Corrected fresh baseline | .577787 | 80.4603% | 79.3598% | .946903 | .900953 | 12.386964 | 184.87 |

Baseline parallel/perpendicular L1 .574301/.607163;5419enabled events, offset-2.
Matches original saved solve.39CPUtests pass; distributed6GPU gradient and
coordinate checks pass. Search active, first change distance_weight .25->.125.

Every candidate is fresh seed0,10080minibatch updates, globalbatch24576,
151hypotheses,CCW, six managed exclusive GPUs. Coordinate factors .5/1/2 then
.25/.5/1/2/4 relative to baseline; restart parameter list after each acceptance.
Acceptance threshold1e-4 L1; no coverage penalty. Optimizer losses have different
weights across trials, so their raw magnitudes are not directly comparable.

Live results: data/uv_gt_l1_search_v1/results.json; best.json and acceptances.json.
Completed trial solutions/configs retained under trials/. Root summary.json
certifies finite-grid completion; no global-optimum claim.

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_gt_l1_search.json
```
Reusing the command resumes matching completed trials; do not run a second copy
while the existing controller is active.
