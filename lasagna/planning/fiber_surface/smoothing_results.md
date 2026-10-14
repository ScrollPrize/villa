# Bending and smoothness sweep — 2026-10-14

Conclusion: 3× smoothness helps, stronger bending does not. Local folds remain,
including cells containing fiber attachments. No default promotion or explicit
repair implementation yet. Baseline definitions/results: joint_uv_results.md.

## Matched protocol

849 retained H pieces in both layouts. Baseline bend5, smooth0.5, step0.5;
grid100base voxels,600joint steps, meshLR5, UVLR0.1, attachment ramp1→20.
All other settings unchanged. Each input's initializer and full targets arrays
were verified identical across its trials, with identical solution/tensor/deformation
hashes. Source input files remain unchanged. Both layout selections match.

Eight new runs on H100 GPUs0–3 with non-exclusive8GiB/8CPU grants, released on
completion. Solve times8.01–9.13s; shared hardware, not isolated performance profiling.
No GPUs6/7, new inference or volume reads. No full-scroll run: these are the supplied
regional layouts. Single trial per setting; no deterministic-repeat claim.

## Results (all distances in base voxels)

Fold count is the existing opposing-triangle criterion, not a global collision
test. All candidates had zero degenerate quads and zero degenerate attachment
normals. XYZ means are arc weighted; p95/max are sample statistics.

| Input | Bending × | Smoothness × | Mean XYZ | p95 XYZ | Maximum XYZ | Normal RMS | Folds | Folds with attachments | Solve seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 1 | 1 | 8.103 | 21.380 | 931.59 | 2.521 | 301 | 73 | 7.50 |
| Original | 3 | 1 | 8.062 | 21.512 | 724.90 | 2.291 | 411 | 75 | 8.24 |
| Original | 10 | 1 | 8.595 | 23.136 | 519.27 | 2.743 | 550 | 90 | 9.13 |
| Original | 1 | 3 | 7.931 | 21.839 | 787.03 | 2.313 | 225 | 57 | 8.26 |
| Original | 1 | 10 | 8.169 | 23.228 | 496.43 | 2.638 | 231 | 49 | 8.13 |
| Deformed | 1 | 1 | 7.916 | 20.608 | 890.83 | 2.506 | 369 | 85 | 7.48 |
| Deformed | 3 | 1 | 7.932 | 20.929 | 778.11 | 2.403 | 427 | 75 | 8.80 |
| Deformed | 10 | 1 | 8.193 | 22.070 | 644.51 | 2.380 | 565 | 64 | 9.00 |
| Deformed | 1 | 3 | 7.995 | 21.575 | 894.67 | 2.687 | 233 | 57 | 8.39 |
| Deformed | 1 | 10 | 8.249 | 23.770 | 574.42 | 2.665 | 249 | 47 | 8.01 |

At3×smoothness, total folds fall25.25%/36.86% for original/deformed. Of the
remaining225/233,168/176 do not contain a final attachment. That does not necessarily
mean extrapolated: many empty cells lie between fibers. Separately,123/127 folded
cells have all four vertices in the initially H-bracketed support class. Neither
classification proves the surface correct or incorrect elsewhere.

The10×smoothness trials have fewer attachment-cell folds but slightly more total
folds and higher mean/p95 error than3×. Strong bending reduces some maximum errors
but increases total folds, so it was not combined with the smoothness trials.

## Interpretation

Existing bend loss penalizes sharp turns along vertex triplets (rows, columns,
diagonals); it is not a direct penalty on opposing normals within each quad.
Smoothness penalizes deviation from the four-neighbor average. Increasing either
weight does not guarantee untangling. The observed tradeoff suggests an explicit
local orientation/area repair term is a sensible next experiment, but it has NOT
been added in this task. Convergence/learning-rate effects are not ruled out by
these fixed600step trials.

## Outputs and reproduction

Best total-fold candidates from this sweep:

- `/home/hendrik/villa4/lasagna/data/smoothing_original_b5_s1.5_v1/surface.tifxyz`
- `/home/hendrik/villa4/lasagna/data/smoothing_deformed_b5_s1.5_v1/surface.tifxyz`

Each trial directory has initial/final surfaces, attachment positions, hashes,
progress and summaries. Exact configurations under `lasagna/configs/`:
`smoothing_{original,deformed}_b{15,50}.json` and
`smoothing_{original,deformed}_s{1.5,5}.json`.

Use the existing approved runner for each configuration, for example:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/smoothing_original_s1.5.json
```

Choose fresh output directories for reruns; do not overwrite completed outputs.
The same runner with `smoothing_report.json` performs CPU-only post-hoc auditing
using the shared geometry-mask helper. Machine-readable results, including all
shift distributions and support counts: [smoothing_metrics.json](smoothing_metrics.json).
Attachment-cell membership uses final moved UVs (floor/border-cell convention),
not the initial UV locations. It is a sample occupancy measure, not a full tube mask.

Validation:52tests passed,1known unrelated speculative-config test excluded.
Regression selection includes the unchanged fixed/sliding surface, optimizer
weights, step-loss and source-adapter tests; new shared-mask regression checks
that reporting uses the same fold definition. Git diff whitespace check clean.
