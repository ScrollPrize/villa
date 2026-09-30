# Density-modulated winding disablement

Current reverse-v2 2048Paris4 traces, unchanged geometry, normals, weights and
500message cap. No new inference or tracing. Initial normal/orientation BP is
unchanged. Reference-removal protocol is the prior oracle-inliers one-round
diagnostic, accepting the historical message-limit condition. Independent
readout is computed from the original frozen BP, not the conditioned/pruned one;
existing GT global gauge alignment remains for scoring.

## Modifier

For piece-local statistic d, measured in forward tracer loss/prediction voxel:

`multiplier = max(0.1, 0.5/(0.5+d))`

`defect penalty = 100 * incident winding measurements * multiplier`

Mean and median use overlap arc-length weights. Worst is max segment loss density.
The same fixed mapping is used for all variants; it is not fitted to removal
labels. Control uses multiplier1. High loss only makes disabling cheaper; it
never increases the original penalty. Average multipliers across pieces:
control1,mean0.6535984,median0.6706050,worst0.5212913. Thus strength and spatial
allocation both differ between variants; this is not an equal-mean-penalty
ablation. A matched uniform-penalty control would be a separate follow-up.

## Validation

Optional multiplier attaches to constraint-piece descriptors and is copied into
the identity piece-to-node model. All5winding unary charge paths share a helper.
Subsets and conditioned ordinary pieces preserve it; reference pieces remain1.
Default behavior unchanged. Optional input validates trace artifact path,
original trace/arc identity, nonempty complete ordered coverage and (0,1]values.

104native winding tests and32Python tests passed. Tests cover default equivalence,
uniform multiplier vs globally scaled unary, invalid values, subset preservation,
weighted statistics, bounds and partial-step handling. The initial native test
attempt stopped before benchmarks on a test comparing invalid NaN coordinates;
fixed to compare validity/discrete labels and valid coordinates. No solver fix
was needed for that failure.

Control reproduces all initial classes/winding assignments and byte-identical
factor CSV from previous baseline, plus38.46%independent accuracy and
-110.6766033%removal score. Code review confirmed readout-only flag cannot feed
independent assignments back into conditioning.

## Reproduce

```sh
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python volume-cartographer/scripts/prepare_defect_density_inputs.py /ephemeral/hendrik/fiber_crop_bench/20260924_cost_audit/beam-crop-new2048-coverage40x80-angle45-reverse-v2_step_costs.csv /ephemeral/hendrik/fiber_crop_bench/20260923_beam2048_v1/bp-beam-new2048-coverage40x80-angle45-reverse-v2_consistency.csv /ephemeral/hendrik/fiber_crop_bench/20260923_beam2048_v1/beam-crop-new2048-coverage40x80-angle45-reverse-v2.zarr volume-cartographer/docs/fiber_benchmark_runs/2026-09-24-defect-density-inputs
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_defect_density_20260924.json
```

Records/logs/OBJ/CSV: `/ephemeral/hendrik/fiber_crop_bench/20260924_defect_density/reverse-density-{control,mean,median,max}`.
Input JSONs are in `2026-09-24-defect-density-inputs/` beside this report.
Build RelWithDebInfo,128workers, one run per variant; no default promotion.

## Results

All four runs completed successfully. No variant improves the benchmark; retain
the unmodulated default. Independent accuracy is 10/26 (38.4615%) throughout;
reference-conditioned accuracy is 26/26 (100%) throughout and is not independent
accuracy.

| Modifier | Removal score (%) ↑ | Initial defect pieces | Reference-removed pieces | Runtime (s) |
|---|---:|---:|---:|---:|
| Control | -110.6766 | 1059/6790 | 1084/6790 | 251.68 |
| Mean | -113.7082 | 1050/6790 | 1103/6790 | 273.25 |
| Median | -113.3489 | 1098/6790 | 1094/6790 | 265.80 |
| Worst | -116.8721 | 1056/6790 | 1135/6790 | 293.82 |

The existing removal score is `-100 * problematic / retained_fulfilled`;
closer to zero is better. It is a ratio, not a fraction of constraints removed,
and can exceed 100% in magnitude. Problematic includes removed-incident and
retained-infringed constraints (retained-defect is zero for all four).

| Modifier | Removed-incident | Retained-infringed | Problematic | Retained-fulfilled |
|---|---:|---:|---:|---:|
| Control | 39199 | 45452 | 84651 | 76485 |
| Mean | 40528 | 45208 | 85736 | 75400 |
| Median | 40309 | 45300 | 85609 | 75527 |
| Worst | 41855 | 44981 | 86836 | 74300 |

All initial solves hit the unchanged 500-message cap. Conditioned control
converged; all three modified conditioned solves hit the message cap, accepted
under the existing benchmark protocol. These are finite-budget experimental
outcomes, not proofs about converged optima. H/V preprocessing classes are
identical and all four factor CSVs have SHA256
`61155ab3d25ebbb10e18372cc0a04907503178253b93fd0594371aac64411b43`.

### Independent rank-one ladder

| Reference | Control | Mean | Median | Worst |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| 1 | 1 | 1 | 1 | 1 |
| 1.5 | 1.5 | 1.5 | 1.5 | 1.5 |
| 2 | 2 | 2 | 2 | 2 |
| 2.5 | 1.5 | 1.5 | 1.5 | 1.5 |
| 3 | 2 | 2 | 2 | 2 |
| 3.5 | 2.5 | 2.5 | 2.5 | 2.5 |
| 4 | 4 | 4 | 4 | 4 |
| 4.5 | 4.5 | 4.5 | 4.5 | 4.5 |
| 5 | 5 | 5 | 5 | 5 |
| 5.5 | 5.5 | 5.5 | 5.5 | 5.5 |
| 6 | 6 | 6 | 6 | 6 |
| 6.5 | 5.5 | 5.5 | 5.5 | 5.5 |
| 7 | 6 | 6 | 6 | 6 |
| 7.5 | 6.5 | 6.5 | 6.5 | 6.5 |
| 8 | 6 | 6 | 6 | 6 |
| 8.5 | 6.5 | 6.5 | 6.5 | 6.5 |
| 9 | 7 | 7 | 7 | 7 |
| 9.5 | 6.5 | 7.5 | 6.5 | 7.5 |
| 10 | 7 | 8 | 7 | 8 |
| 10.5 | 7.5 | 8.5 | 7.5 | 8.5 |
| 11 | 8 | 9 | 9 | 9 |
| 11.5 | 9.5 | 9.5 | 9.5 | 9.5 |
| 12 | 10 | 10 | 10 | 10 |
| 12.5 | 9.5 | 9.5 | 9.5 | 9.5 |

Mean/worst move some upper windings closer but make none exactly correct.
This tests one shared mapping, not an exhaustive penalty-strength sweep. A
uniform penalty matched to each mean multiplier would separate overall penalty
reduction from piece-local targeting; it was not part of this matrix.
