# Reviewed 13-transform TTA

Frozen u31500, reviewed91fibers/182directions, GPUs0–5. Same beam8 tracing/thresholds as previous evaluation. Mean and median share13forwards per tile and exactly14accumulated channels. Baseline is a separate ordinary7channel run. Strict float32 (TF32 disabled), batch1, float32 accumulation. Previous bf16 row is historical context, not the matched baseline.

Directions reframed analytically in sign-invariant double-angle coordinates; mean or componentwise midpoint median of cos2/sin2, capped to unit disk then re-encoded. Presence ordinary mean/median. Median is not a spherical geometric median. Tile aggregates are then blended by the existing accumulator. No embeddings; pilot limited to output stride1 and one branch.

Mean is combined directed reference length/(restarts+1); quantiles include terminal segments. This reviewed hard-region cohort is not proven to be an independent training holdout.

| Variant | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| previous_bf16 | 24 | 0.01731 | 55.447 | 5.240 | 1.486 | 21.845 |
| baseline | 24 | 0.01731 | 55.447 | 5.230 | 1.793 | 21.364 |
| mean | 24 | 0.01731 | 55.447 | 5.239 | 1.503 | 20.397 |
| median | 19 | 0.01371 | 69.309 | 5.248 | 1.775 | 22.031 |

## Per scroll

| Scroll / variant | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 / previous_bf16 | 9 | 0.03155 | 28.522 | 4.521 | 1.193 | 11.719 |
| PHerc0139 / baseline | 7 | 0.02454 | 35.653 | 4.521 | 1.441 | 11.719 |
| PHerc0139 / mean | 7 | 0.02454 | 35.653 | 4.778 | 1.371 | 11.719 |
| PHerc0139 / median | 6 | 0.02104 | 40.746 | 4.928 | 1.371 | 11.719 |
| PHerc0332 / previous_bf16 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / baseline | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / mean | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / median | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc1299 / previous_bf16 | 6 | 0.04070 | 21.061 | 5.033 | 1.479 | 10.212 |
| PHerc1299 / baseline | 6 | 0.04070 | 21.061 | 5.033 | 1.391 | 10.176 |
| PHerc1299 / mean | 6 | 0.04070 | 21.061 | 5.033 | 0.921 | 10.200 |
| PHerc1299 / median | 2 | 0.01357 | 49.142 | 5.481 | 2.611 | 10.628 |
| PHerc1451 / previous_bf16 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / baseline | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / mean | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / median | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHercParis4 / previous_bf16 | 9 | 0.00966 | 93.190 | 5.376 | 1.928 | 25.336 |
| PHercParis4 / baseline | 11 | 0.01180 | 77.659 | 5.357 | 1.930 | 25.336 |
| PHercParis4 / mean | 11 | 0.01180 | 77.659 | 5.353 | 2.036 | 25.336 |
| PHercParis4 / median | 11 | 0.01180 | 77.659 | 5.357 | 2.036 | 25.336 |

## Reproduction

Set scripts/training_failure_mining_control.json action to reviewed_tta_sweep, then run:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Outputs/configs: /ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/tta13_u31500_v2. Missing rows are not yet completed. Mean fills the shared two-product cache; median reuses it and only infers missing coverage. Their wall times are not independent TTA inference benchmarks.
