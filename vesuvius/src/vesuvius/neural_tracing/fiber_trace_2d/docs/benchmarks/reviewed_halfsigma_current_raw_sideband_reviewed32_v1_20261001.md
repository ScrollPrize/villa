# Reviewed cohort: halfsigma_current_raw_sideband_reviewed32_v1

Snapshot: `/ephemeral/hendrik/fiber3d_train2/runs/5scroll_json_eligible_fullres_refinement_gaussian_halfsigma_sideband_reviewed32_128_sd1_16x8_ema_20261001_203634/snapshots/current.pt`, frozen update37000, `test/loss_total`=0.75786823. SHA256 `3a55f20dc6137a384d5b54fd25902263472daefb558a702d170af3ad579dafb6`.

Same corrected reviewedv3 cohort:91fibers,182directions, five scrolls. All directions completed with zero infrastructure failures. No embedding, no fw/bw filtering. Native beam8/lookahead2, step4prediction voxels,40/80base ellipse,160base grace,80base restart grouping. Fullres prediction lattice2base voxels,64³output chunks.192workers, GPUs0–5 demand-managed.

Mean uses the established combined directed reference distance/(restarts+1); quantiles use reference partitions including successful terminal spans. This selected reviewed hard-region cohort is not proven to be an independent training holdout.

| Model | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| lowres | 29 | 0.02092 | 46.206 | 5.035 | 1.445 | 21.856 |
| fullres | 36 | 0.02597 | 37.464 | 4.780 | 1.380 | 23.660 |
| gaussian | 32 | 0.02309 | 42.005 | 4.942 | 1.371 | 19.780 |
| halfsigma_raw | 28 | 0.02020 | 47.799 | 5.025 | 1.447 | 24.319 |
| halfsigma_ema99 | 29 | 0.02092 | 46.206 | 5.016 | 1.371 | 24.055 |
| halfsigma_current_raw_v1 | 32 | 0.02309 | 42.005 | 5.033 | 1.344 | 21.827 |
| halfsigma_current_raw_v2 | 24 | 0.01731 | 55.447 | 5.220 | 1.574 | 20.239 |
| halfsigma_current_raw_v3 | 24 | 0.01731 | 55.447 | 5.240 | 1.486 | 21.845 |
| halfsigma_current_raw_sideband_reviewed32_v1 | 17 | 0.01226 | 77.010 | 5.248 | 1.619 | 24.738 |

## Per scroll

| Scroll / model | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 / lowres | 10 | 0.03506 | 25.929 | 4.521 | 1.371 | 10.638 |
| PHerc0139 / fullres | 16 | 0.05610 | 16.778 | 3.593 | 1.365 | 10.480 |
| PHerc0139 / gaussian | 12 | 0.04207 | 21.940 | 3.814 | 1.371 | 11.719 |
| PHerc0139 / halfsigma_raw | 9 | 0.03155 | 28.522 | 3.814 | 1.371 | 11.719 |
| PHerc0139 / halfsigma_ema99 | 12 | 0.04207 | 21.940 | 3.788 | 0.651 | 11.719 |
| PHerc0139 / halfsigma_current_raw_v1 | 16 | 0.05610 | 16.778 | 3.701 | 0.858 | 11.746 |
| PHerc0139 / halfsigma_current_raw_v2 | 8 | 0.02805 | 31.691 | 4.778 | 1.261 | 11.719 |
| PHerc0139 / halfsigma_current_raw_v3 | 9 | 0.03155 | 28.522 | 4.521 | 1.193 | 11.719 |
| PHerc0139 / halfsigma_current_raw_sideband_reviewed32_v1 | 6 | 0.02104 | 40.746 | 4.778 | 1.236 | 11.719 |
| PHerc0332 / lowres | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / fullres | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / gaussian | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / halfsigma_raw | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / halfsigma_ema99 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / halfsigma_current_raw_v1 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / halfsigma_current_raw_v2 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / halfsigma_current_raw_v3 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / halfsigma_current_raw_sideband_reviewed32_v1 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc1299 / lowres | 4 | 0.02713 | 29.485 | 4.702 | 1.957 | 15.539 |
| PHerc1299 / fullres | 4 | 0.02713 | 29.485 | 4.811 | 1.535 | 15.539 |
| PHerc1299 / gaussian | 4 | 0.02713 | 29.485 | 4.811 | 1.003 | 15.539 |
| PHerc1299 / halfsigma_raw | 5 | 0.03392 | 24.571 | 4.666 | 1.059 | 11.854 |
| PHerc1299 / halfsigma_ema99 | 5 | 0.03392 | 24.571 | 5.033 | 1.068 | 10.915 |
| PHerc1299 / halfsigma_current_raw_v1 | 4 | 0.02713 | 29.485 | 5.257 | 1.008 | 11.039 |
| PHerc1299 / halfsigma_current_raw_v2 | 7 | 0.04748 | 18.428 | 4.589 | 0.797 | 10.253 |
| PHerc1299 / halfsigma_current_raw_v3 | 6 | 0.04070 | 21.061 | 5.033 | 1.479 | 10.212 |
| PHerc1299 / halfsigma_current_raw_sideband_reviewed32_v1 | 3 | 0.02035 | 36.857 | 5.033 | 1.661 | 15.957 |
| PHerc1451 / lowres | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / fullres | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / gaussian | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / halfsigma_raw | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / halfsigma_ema99 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / halfsigma_current_raw_v1 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / halfsigma_current_raw_v2 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / halfsigma_current_raw_v3 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / halfsigma_current_raw_sideband_reviewed32_v1 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHercParis4 / lowres | 15 | 0.01610 | 58.244 | 5.187 | 1.459 | 25.157 |
| PHercParis4 / fullres | 16 | 0.01717 | 54.818 | 5.111 | 1.472 | 25.602 |
| PHercParis4 / gaussian | 16 | 0.01717 | 54.818 | 5.097 | 1.496 | 25.336 |
| PHercParis4 / halfsigma_raw | 14 | 0.01502 | 62.127 | 5.187 | 1.628 | 25.643 |
| PHercParis4 / halfsigma_ema99 | 12 | 0.01288 | 71.685 | 5.270 | 1.721 | 25.336 |
| PHercParis4 / halfsigma_current_raw_v1 | 12 | 0.01288 | 71.685 | 5.353 | 1.790 | 25.336 |
| PHercParis4 / halfsigma_current_raw_v2 | 9 | 0.00966 | 93.190 | 5.357 | 2.061 | 25.336 |
| PHercParis4 / halfsigma_current_raw_v3 | 9 | 0.00966 | 93.190 | 5.376 | 1.928 | 25.336 |
| PHercParis4 / halfsigma_current_raw_sideband_reviewed32_v1 | 8 | 0.00858 | 103.545 | 5.481 | 1.972 | 25.801 |

## Reproduction

Training provenance: the 20261001_203628 launch config uses the original train split for `datasets` and the reviewed32 failure union for `test_datasets` and periodic trace validation. The reviewed32 name does not mean direct training on those 32 annotations. This cohort is monitored during training, so this is not a blind final-test comparison.

Set scripts/training_failure_mining_control.json to `{"action": "reviewed_eval", "reviewed_model": "halfsigma_current_raw_sideband_reviewed32_v1"}`, then:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Output: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/halfsigma_current_raw_sideband_reviewed32_v1/`. Frozen snapshot, selection metadata, workload, preparation/binding validation and complete per-case artifacts retained there.

Coordinator seconds (cache/inference work differs; not a model-speed comparison): lowres=23.18, fullres=26.50, gaussian=49.92, halfsigma_raw=245.12, halfsigma_ema99=234.64, halfsigma_current_raw_v1=239.70, halfsigma_current_raw_v2=245.94, halfsigma_current_raw_v3=240.12, halfsigma_current_raw_sideband_reviewed32_v1=254.93.

Validation:focused Python tests; independent plan review; identical per-case reference selection/lengths; all182cases complete; effective embedding/reverse settings and saved per-case check counters verified disabled. No native or scheduling changes.
