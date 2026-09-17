# Fiberlet construction-DP retuning on refinement EMA999 u22000

## Outcome

Retain the existing production construction defaults. A staged 25-setting
construction-DP search (plus same-matrix prewarms) on corrected mini5 did not
remove any of the five failures. The strongest result was longitudinal step 4
and transverse step 0.25 prediction voxels: 100% coverage and summed
first-failure progress 6904.223 base voxels, +157.169 over production's
6747.054. It improved every individual case but missed the predeclared +160
promotion gate by 2.831 base voxels. It was therefore not promoted to short200.

The candidate is also substantially more expensive to construct: 364.975s cold
on mini5 versus a 14.645s cached production replay. Its separate path identity
includes cold fiberlet construction and currently duplicates the unchanged
anchor cache. The times are consequently useful end-to-end evidence, not a pure
DP-kernel comparison.

## Fixed protocol

- Model: refinement EMA999 u22000, SHA
  `67edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742`.
- Benchmark: corrected mini5 v3, manifest SHA
  `ed031e2f07dafa32b0393186ae531422aee0909898d1e1f621fba69301ddddbe`,
  5 directed cases / 16.8873457254mm.
- Production anchors; graph replay fixed at beam16, 48-base decision step and
  384-base lookahead.
- 40/80-base acceptance ellipse, 160-base excursion grace, and 80-base restart
  grouping unchanged.
- Storage chunks 128 base voxels and 224 native construction workers. All
  prediction volumes were cached (`tiles=0`).
- Ranking: failures, coverage, summed first-failure-or-end progress, then
  declared order. Promotion at equal failures required 100% coverage, at least
  +160 summed base voxels, and no individual regression above 64 base voxels.

## Lattice stages

Stage 1 crossed longitudinal `{1,1.5,2,3,4}` with transverse `{0.5,1}`.
Stage 2 deterministically selected longitudinal 2 and 4 and tested transverse
0.25. Prewarm rows are omitted.

| Stage | Longitudinal | Transverse | Failures | Coverage % | First total | Delta vs production | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | 1 | 0.5 | 5 | 100.000000 | 6630.833 | -116.221 | 275.353 |
| S1 | 1 | 1 | 5 | 99.671241 | 6622.635 | -124.419 | 218.950 |
| S1 | 1.5 | 0.5 | 5 | 100.000000 | 6657.729 | -89.325 | 291.340 |
| S1 | 1.5 | 1 | 5 | 99.744366 | 6657.729 | -89.325 | 230.081 |
| S1 | 2 | 0.5 | 5 | 100.000000 | 6747.054 | +0.000 | 14.756 |
| S1 | 2 | 1 | 5 | 100.000000 | 6745.975 | -1.079 | 228.027 |
| S1 | 3 | 0.5 | 5 | 100.000000 | 6495.679 | -251.375 | 268.580 |
| S1 | 3 | 1 | 5 | 100.000000 | 6705.135 | -41.919 | 234.915 |
| S1 | 4 | 0.5 | 5 | 100.000000 | 6704.727 | -42.327 | 261.411 |
| S1 | 4 | 1 | 5 | 100.000000 | 6746.110 | -0.944 | 241.768 |
| S2 | 2 | 0.5 | 5 | 100.000000 | 6747.054 | +0.000 | 14.645 |
| S2 | 2 | 0.25 | 5 | 100.000000 | 6622.893 | -124.161 | 399.096 |
| S2 | 4 | 0.25 | 5 | 100.000000 | 6904.223 | **+157.169** | 364.975 |

The 4/0.25 candidate improved all five first-failure positions by +0.516 to
+58.529 base voxels. It nevertheless retained every failure and narrowly missed
the locked aggregate threshold.

## Reach and objective stages

The bounded follow-up held 4/0.25 fixed. Raw corridor radius zero is the native
backward-compatible setting and resolves to the 4-prediction-voxel anchor cell;
explicit radius 4 was deliberately excluded as a behaviorally duplicate but
differently hashed identity.

| Variant | Failures | Coverage % | First total | Wall s | Result |
|---|---:|---:|---:|---:|---|
| 4/0.25 baseline | 5 | 100.000000 | 6904.223 | 14.107 | reference |
| corridor 2 | 5 | 100.000000 | 6904.223 | 291.357 | identical |
| corridor 3 | 5 | 100.000000 | 6904.223 | 328.582 | identical |
| corridor 6 | 5 | 100.000000 | 6904.223 | 437.082 | identical |
| deviation 15° | 5 | 100.000000 | 6904.223 | 303.458 | identical |
| deviation 20° | 5 | 100.000000 | 6904.223 | 336.534 | identical |
| deviation 35° | 5 | 100.000000 | 6903.932 | 458.476 | -0.291 |
| invalid cost 2 | 5 | 100.000000 | 6904.223 | 366.918 | identical |
| invalid cost 8 | 5 | 100.000000 | 6904.223 | 362.147 | identical |
| smoothness 1 | 5 | 100.000000 | 6904.223 | 366.225 | identical |
| smoothness 4 | 5 | 100.000000 | 6904.223 | 365.343 | identical |
| tangent smoothness 5 | 5 | 100.000000 | 6904.223 | 361.968 | identical |
| tangent smoothness 20 | 5 | 100.000000 | 6903.932 | 363.368 | -0.291 |

Reach and scalar local costs are not limiting these cases. The only positive
signal came from the discretized 4/0.25 lattice, and it was too small to justify
its construction cost or a short200 promotion. Historical short200 therefore
remains 67 failures / 9.736615mm mean for production fiberlets versus 62 failures
/ 10.509363mm for the regular u22000 tracer.

## Configuration surface

The evaluator now accepts strict `fiberlet.path` overrides for exactly the twelve
persisted semantic `FiberletPathConfig` fields:

`cell_radius`, `neighborhood_margin_cells`,
`longitudinal_step_prediction`, `transverse_step_prediction`,
`maximum_endpoint_angle_degrees`, `maximum_prediction_deviation_degrees`,
`corridor_radius_prediction`, `invalid_prediction_cost_per_voxel`,
`smoothness_weight`, `smoothness_normal_weight`,
`smoothness_tangent_weight`, and `smoothness_free_angle_degrees`.

Execution-only sampling-batch and thread fields are intentionally rejected.
Overrides feed the canonical native metadata, fingerprint, on-demand
construction, and cached replay path; there is no Python approximation.

## Artifacts and reproduction

Immutable matrices, attempts, full metrics, and per-case diagnostics:

- `/ephemeral/hendrik/fiber3d_train2/evaluations/mini5_fiberlet_path_lattice_stage1_20260917_v1`
- `/ephemeral/hendrik/fiber3d_train2/evaluations/mini5_fiberlet_path_lattice_stage2_20260917_v1`
- `/ephemeral/hendrik/fiber3d_train2/evaluations/mini5_fiberlet_path_reach_stage3_20260917_v1`
- `/ephemeral/hendrik/fiber3d_train2/evaluations/mini5_fiberlet_path_cost_stage4_20260917_v1`

Run any checked-in config through the approved command:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_fiberlet_path_lattice_stage1_20260917_v1.json
```

Replace the config filename for stages 2–4. Completed candidates validate their
bound model/manifest/native/matrix identity and resume without recomputation.
The approved build and replay-test runners passed native suites 23/85/17 and
Python 270 passed / 8 optional skips.
