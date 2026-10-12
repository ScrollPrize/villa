# Corrected joint solves against reviewed-fiber GT

Implementation/GT commit: e2ba3415a. Plan: gt_constraint_score_plan.md.

## Protocol
Same canonical test subset: 2192 source fibers / 2199 pieces, 24771 internal
constraints, spiral selection IDs14–20 and base Z30000–50000. Same best weights,
seed0,151hypotheses,CCW,10080sampled minibatch updates (~10000 pass equivalents),
globalbatch24576, six exclusive managed GPUs. Both enable bucket_shared_phase.
Fresh starts from radial prior; reference initializes winding and widths from
spiral surface. All variables optimized from start; reviewed GT never in training.

Frozen seven-fiber target: data/bbox_gt_solve_v3/uv_whole/solution.npz.
Cross extraction on complete original source curves, then filter test endpoints
to solved pieces. No new cuts or constraints alter either solution. Native
extraction1.88s + orientation1.22s; frozen GT normal export0.17s.

## New primary score
Each cross event casts one nearest-integer winding-offset vote. Exact success
means that vote equals one modal global integer offset (tie: minimum mean absolute
residual, then smallest absolute offset, then smaller integer). Round half toward
positive infinity. All disabled pieces remain in denominator at their strongest
nonbroken hypothesis. No per-fiber offset optimization or GT correction.

A=test,B=GT: coarse delta = qGT-qTest-wrapped_original_angle_difference
+0.5*(familyGT-familyTest). Required additive test offset = delta-signed distance.
Coarse q uses selected bucket + shared phase + original angular travel, no fine
per-bucket offset or width. Positive measured distance means GT is outward.
Signs use GT endpoint normals, aligned to the frozen original GT normal sweep.
This deliberately does not require tested-fiber normal agreement. The native
curve export checks all original GT endpoint normals before use; zero combined
sweep gauge corrections were needed on these inputs.

| Metric | Fresh | Reference initialized |
|---|---:|---:|
| **GT score, all constraints (%)** | 49.6214 | 55.0705 |
| Correct / total | 3342/6735 | 3709/6735 |
| Global additive offset | -2.0000 | -2.0000 |
| Enabled-event score (%) | 58.4056 | 55.9680 |
| Disabled-event score (%) | 13.4498 | 38.7464 |
| Disabled-event count | 1316.0000 | 351.0000 |
| Parallel score (%) | 50.0670 | 55.1926 |
| Perpendicular score (%) | 46.1438 | 54.1176 |
| GT mean absolute continuous residual (turns) | 0.9569 | 0.5389 |
| Retained length (%) | 79.3598 | 93.0417 |
| Spiral reference L1 | 0.9469 | 0.5278 |
| Spiral within two windings (%) | 92.3477 | 98.1490 |
| Width factor fit/reference | 0.9010 | 0.8897 |
| Internal retained sign violations (%) | 0.1161 | 0.2326 |
| Online mean objective | 12.3870 | 7.2351 |
| Solve seconds | 173.2322 | 173.9896 |

## Per GT fiber

| JSON suffix | Constraints | Fresh score (%) | Reference score (%) |
|---|---:|---:|---:|
| 000001 | 1425 | 53.7544 | 56.6316 |
| 000002 | 1063 | 41.6745 | 53.9040 |
| 000003 | 853 | 42.2040 | 54.3962 |
| 000004 | 1075 | 72.1860 | 50.9767 |
| 000005 | 941 | 37.3007 | 51.5409 |
| 000006 | 839 | 52.6818 | 66.7461 |
| 000007 | 539 | 37.8479 | 50.4638 |

## Coverage / limitations
6735 common events:5970parallel,765perpendicular.1119near-tangent connectors
have no reliable signed distance and are excluded identically;7events outside
solved spans excluded. No invalid frozen normals, no family disagreements;
all seven GT fibers supported.25923non-cross events are not evaluation data.
This score weights events equally, not families or fibers; dense parallel
encounters dominate. Its exact winding-vote test is stricter than the old
within-two statistic, and the old metric only covers retained fibers. Do not
compare their percentages as the same metric or attribute every failed vote to
a tracing error: extracted distances and inferred GT layout are approximate.

## Reproduction / artifacts
Approved runner prefix:
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_gt_phase_suite.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_gt_constraint_score.json
```
Solutions: data/uv_gt_phase_{fresh,reference}_v1/solution.npz and summary.json.
Score/provenance: data/gt_constraint_score_v1/results.json. Individual votes:
uv_gt_phase_{fresh,reference}_v1_events.json in that directory, including source,
piece, arc, normal cosine, signed measurement, vote and corrected residual.
Frozen GT and solved states unmodified by evaluation. Hashes saved with results.

Validation:30CPUtests passed; sixGPU distributed gradient checks and coordinate
value/gradient/repeat checks passed for both solves. Independent metric review
approved including normal gauge fix. Exact scoring rerun checked separately.

## Detailed saved-event breakdown

Same global offset -2 throughout; distances are measured Lasagna windings.

| Absolute measured distance | Events | Fresh correct % | Ref correct % | Fresh L1 | Ref L1 |
|---|---:|---:|---:|---:|---:|
| 0–<0.5 | 2936 | 52.76 | 57.12 | .897 | .534 |
| .5–<1 | 1109 | 54.46 | 50.14 | .928 | .528 |
| 1–<1.5 | 1722 | 44.72 | 53.25 | 1.043 | .510 |
| 1.5–<2 | 466 | 52.36 | 58.37 | .842 | .606 |
| 2–<3 | 393 | 31.81 | 55.98 | 1.215 | .630 |
| >=3 | 109 | 45.87 | 61.47 | 1.066 | .621 |

Rounded errors 0/1/2/>=3: fresh 49.6214/35.3823/4.2019/10.7944%;
reference 55.0705/41.7075/2.9547/.2673%.
Continuous residual median/p95: fresh .5047/4.6843, reference .4670/1.1983.
Enabled L1: fresh .5777874823 (5419 events), reference .5108947290 (6384).
Disabled L1: fresh 2.5180693541 (1316), reference 1.0487780533 (351).
Parallel L1 .968164/.534981; perpendicular .869110/.569718 (fresh/reference).
583 of fresh's 727 >=3-winding errors touch disabled pieces.
