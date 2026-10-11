# Reference-initialized CCW full joint solve

Completed 10 exhaustive passes successfully (exit 0), 1394.68 seconds to final diagnostic.
Config: run_uv_reference_init151_ccw.json. Output: data/uv_reference_init151_v1/ccw_joint.
151 winding hypotheses; every global variable optimized from update one, including
broken probabilities; fixed rejection cost; no warmup or frozen geometry. Reference
only initializes winding logits and supplies evaluation; unmatched fibers use radial
initialization. Upstream signs/HV/directions remain preprocessing inputs.

| Pass | Mean objective | Sign violations | Retained length | Winding L1 | Within ±2 | Fit q10–q90 | Reference q10–q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | — | 21.3921% | 100.0000% | 0.000439 | 100.0000% | 42–124 | 42–124 |
| 1 | 844.1183 | 7.4528% | 91.2974% | 0.834117 | 94.0543% | 42–124 | 42–124 |
| 2 | 706.0184 | 5.5369% | 86.0172% | 0.785325 | 95.3285% | 42–124 | 42–124 |
| 3 | 249.1383 | 4.1958% | 84.5944% | 0.761410 | 95.7904% | 42–124 | 42–124 |
| 4 | 81.7946 | 3.1415% | 84.1384% | 0.781679 | 95.4627% | 42–124 | 42–124 |
| 5 | 33.9290 | 2.4177% | 83.9663% | 0.807627 | 94.9659% | 42–124 | 42–124 |
| 6 | 17.5435 | 2.0901% | 83.8719% | 0.828627 | 94.5408% | 42–124 | 42–124 |
| 7 | 10.1512 | 1.7818% | 83.8452% | 0.847085 | 94.1551% | 42–124 | 42–124 |
| 8 | 6.3685 | 1.5542% | 83.8490% | 0.862602 | 93.9060% | 42–124 | 42–124 |
| 9 | 4.2994 | 1.5656% | 83.8523% | 0.872689 | 93.7300% | 42–124 | 42–124 |
| 10 | 3.2551 | 1.4995% | 83.8699% | 0.880643 | 93.6253% | 42–124 | 42–124 |

Objective is the mean online objective during a pass, not a frozen-state full-objective
measurement. Passes 0–9 audit the fixed 100k event sample; pass 10 audits all 6,606,100
events. Winding metrics are equal-source-fiber statistics among retained reference-matched
fibers; global offset correction is zero throughout. Final matched fibers:361194.
Final loss splits: pair0.996477, regularization2.258641. Earlier radial CCW objective
6.363428 is not an initialization-only control: it used delayed broken updates and
degree-weighted rejection, unlike this fixed-cost/no-delay run.

Run command:
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_reference_init151_ccw.json
```

