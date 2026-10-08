# Reviewed cohort: halfsigma_sideband_best_ema99_20261002

Snapshot: `/ephemeral/hendrik/fiber3d_train2/runs/5scroll_json_eligible_fullres_refinement_gaussian_halfsigma_sideband_reviewed32_128_sd1_16x8_ema_20261001_203634/snapshots/best_ema_0p99.pt`, frozen update40800, `test_ema_0p99/loss_total`=0.74116903. SHA256 `6b35ca8c563bbe56f7a8f18ee05c5cd2051220752b3d7a99809b5e65aeb309e6`.

Same corrected reviewedv3 cohort:91fibers,182directions, five scrolls. All directions completed with zero infrastructure failures. No embedding, no fw/bw filtering. Native beam8/lookahead2, step4prediction voxels,40/80base ellipse,160base grace,80base restart grouping. Fullres prediction lattice2base voxels,64³output chunks.192workers, GPUs0–5 demand-managed.

Mean uses the established combined directed reference distance/(restarts+1); quantiles use reference partitions including successful terminal spans. This selected reviewed hard-region cohort is not proven to be an independent training holdout.

| Model | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| lowres | 29 | 0.02092 | 46.206 | 5.035 | 1.445 | 21.856 |
| fullres | 36 | 0.02597 | 37.464 | 4.780 | 1.380 | 23.660 |
| gaussian | 32 | 0.02309 | 42.005 | 4.942 | 1.371 | 19.780 |
| halfsigma_raw | 28 | 0.02020 | 47.799 | 5.025 | 1.447 | 24.319 |
| halfsigma_current_raw_v3 | 24 | 0.01731 | 55.447 | 5.240 | 1.486 | 21.845 |
| halfsigma_current_raw_sideband_reviewed32_v1 | 17 | 0.01226 | 77.010 | 5.248 | 1.619 | 24.738 |
| halfsigma_sideband_best_raw_20261002 | 16 | 0.01154 | 81.540 | 5.248 | 1.925 | 24.738 |
| halfsigma_sideband_best_ema99_20261002 | 22 | 0.01587 | 60.269 | 5.244 | 1.495 | 23.284 |

## Per scroll

| Scroll / model | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 / lowres | 10 | 0.03506 | 25.929 | 4.521 | 1.371 | 10.638 |
| PHerc0139 / fullres | 16 | 0.05610 | 16.778 | 3.593 | 1.365 | 10.480 |
| PHerc0139 / gaussian | 12 | 0.04207 | 21.940 | 3.814 | 1.371 | 11.719 |
| PHerc0139 / halfsigma_raw | 9 | 0.03155 | 28.522 | 3.814 | 1.371 | 11.719 |
| PHerc0139 / halfsigma_current_raw_v3 | 9 | 0.03155 | 28.522 | 4.521 | 1.193 | 11.719 |
| PHerc0139 / halfsigma_current_raw_sideband_reviewed32_v1 | 6 | 0.02104 | 40.746 | 4.778 | 1.236 | 11.719 |
| PHerc0139 / halfsigma_sideband_best_raw_20261002 | 8 | 0.02805 | 31.691 | 4.168 | 1.371 | 11.719 |
| PHerc0139 / halfsigma_sideband_best_ema99_20261002 | 8 | 0.02805 | 31.691 | 4.649 | 1.306 | 11.719 |
| PHerc0332 / lowres | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / fullres | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / gaussian | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / halfsigma_raw | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / halfsigma_current_raw_v3 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / halfsigma_current_raw_sideband_reviewed32_v1 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / halfsigma_sideband_best_raw_20261002 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / halfsigma_sideband_best_ema99_20261002 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc1299 / lowres | 4 | 0.02713 | 29.485 | 4.702 | 1.957 | 15.539 |
| PHerc1299 / fullres | 4 | 0.02713 | 29.485 | 4.811 | 1.535 | 15.539 |
| PHerc1299 / gaussian | 4 | 0.02713 | 29.485 | 4.811 | 1.003 | 15.539 |
| PHerc1299 / halfsigma_raw | 5 | 0.03392 | 24.571 | 4.666 | 1.059 | 11.854 |
| PHerc1299 / halfsigma_current_raw_v3 | 6 | 0.04070 | 21.061 | 5.033 | 1.479 | 10.212 |
| PHerc1299 / halfsigma_current_raw_sideband_reviewed32_v1 | 3 | 0.02035 | 36.857 | 5.033 | 1.661 | 15.957 |
| PHerc1299 / halfsigma_sideband_best_raw_20261002 | 2 | 0.01357 | 49.142 | 5.257 | 2.088 | 16.375 |
| PHerc1299 / halfsigma_sideband_best_ema99_20261002 | 2 | 0.01357 | 49.142 | 5.257 | 2.105 | 16.375 |
| PHerc1451 / lowres | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / fullres | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / gaussian | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / halfsigma_raw | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / halfsigma_current_raw_v3 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / halfsigma_current_raw_sideband_reviewed32_v1 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / halfsigma_sideband_best_raw_20261002 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / halfsigma_sideband_best_ema99_20261002 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHercParis4 / lowres | 15 | 0.01610 | 58.244 | 5.187 | 1.459 | 25.157 |
| PHercParis4 / fullres | 16 | 0.01717 | 54.818 | 5.111 | 1.472 | 25.602 |
| PHercParis4 / gaussian | 16 | 0.01717 | 54.818 | 5.097 | 1.496 | 25.336 |
| PHercParis4 / halfsigma_raw | 14 | 0.01502 | 62.127 | 5.187 | 1.628 | 25.643 |
| PHercParis4 / halfsigma_current_raw_v3 | 9 | 0.00966 | 93.190 | 5.376 | 1.928 | 25.336 |
| PHercParis4 / halfsigma_current_raw_sideband_reviewed32_v1 | 8 | 0.00858 | 103.545 | 5.481 | 1.972 | 25.801 |
| PHercParis4 / halfsigma_sideband_best_raw_20261002 | 6 | 0.00644 | 133.129 | 5.502 | 2.061 | 25.915 |
| PHercParis4 / halfsigma_sideband_best_ema99_20261002 | 12 | 0.01288 | 71.685 | 5.387 | 1.891 | 25.336 |

## Reproduction

Set scripts/training_failure_mining_control.json to `{"action": "reviewed_eval", "reviewed_model": "halfsigma_sideband_best_ema99_20261002"}`, then:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Output: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/halfsigma_sideband_best_ema99_20261002/`. Frozen snapshot, selection metadata, workload, preparation/binding validation and complete per-case artifacts retained there.

Coordinator seconds (cache/inference work differs; not a model-speed comparison): lowres=23.18, fullres=26.50, gaussian=49.92, halfsigma_raw=245.12, halfsigma_current_raw_v3=240.12, halfsigma_current_raw_sideband_reviewed32_v1=254.93, halfsigma_sideband_best_raw_20261002=269.26, halfsigma_sideband_best_ema99_20261002=244.02.

Validation:focused Python tests; independent plan review; identical per-case reference selection/lengths; all182cases complete; effective embedding/reverse settings and saved per-case check counters verified disabled. No native or scheduling changes.
