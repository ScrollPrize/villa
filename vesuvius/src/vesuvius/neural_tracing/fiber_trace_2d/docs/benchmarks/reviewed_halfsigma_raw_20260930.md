# Reviewed cohort: half-sigma best raw

Snapshot: `/ephemeral/hendrik/fiber3d_train2/runs/5scroll_json_eligible_fullres_refinement_gaussian_halfsigma_128_sd1_16x8_ema_20260930_155818/snapshots/best.pt`, frozen update18600, raw `test/loss_total`=0.78594075. SHA256 `78533ef2dbf70a8d2af68c4bd700c8061046e86f991a5ada95e46cc0daf0b819`.

Same corrected reviewedv3 cohort:91fibers,182directions, five scrolls. All directions completed with zero infrastructure failures. No embedding, no fw/bw filtering. Native beam8/lookahead2, step4prediction voxels,40/80base ellipse,160base grace,80base restart grouping. Fullres prediction lattice2base voxels,64³output chunks.192workers, GPUs0–5 demand-managed.

Mean uses the established combined directed reference distance/(restarts+1); quantiles use reference partitions including successful terminal spans. This selected reviewed hard-region cohort is not proven to be an independent training holdout.

| Model | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| lowres | 29 | 0.02092 | 46.206 | 5.035 | 1.445 | 21.856 |
| fullres | 36 | 0.02597 | 37.464 | 4.780 | 1.380 | 23.660 |
| gaussian | 32 | 0.02309 | 42.005 | 4.942 | 1.371 | 19.780 |
| halfsigma_raw | 28 | 0.02020 | 47.799 | 5.025 | 1.447 | 24.319 |

## Per scroll

| Scroll / model | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 / lowres | 10 | 0.03506 | 25.929 | 4.521 | 1.371 | 10.638 |
| PHerc0139 / fullres | 16 | 0.05610 | 16.778 | 3.593 | 1.365 | 10.480 |
| PHerc0139 / gaussian | 12 | 0.04207 | 21.940 | 3.814 | 1.371 | 11.719 |
| PHerc0139 / halfsigma_raw | 9 | 0.03155 | 28.522 | 3.814 | 1.371 | 11.719 |
| PHerc0332 / lowres | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / fullres | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / gaussian | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 / halfsigma_raw | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc1299 / lowres | 4 | 0.02713 | 29.485 | 4.702 | 1.957 | 15.539 |
| PHerc1299 / fullres | 4 | 0.02713 | 29.485 | 4.811 | 1.535 | 15.539 |
| PHerc1299 / gaussian | 4 | 0.02713 | 29.485 | 4.811 | 1.003 | 15.539 |
| PHerc1299 / halfsigma_raw | 5 | 0.03392 | 24.571 | 4.666 | 1.059 | 11.854 |
| PHerc1451 / lowres | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / fullres | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / gaussian | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 / halfsigma_raw | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHercParis4 / lowres | 15 | 0.01610 | 58.244 | 5.187 | 1.459 | 25.157 |
| PHercParis4 / fullres | 16 | 0.01717 | 54.818 | 5.111 | 1.472 | 25.602 |
| PHercParis4 / gaussian | 16 | 0.01717 | 54.818 | 5.097 | 1.496 | 25.336 |
| PHercParis4 / halfsigma_raw | 14 | 0.01502 | 62.127 | 5.187 | 1.628 | 25.643 |

## Reproduction

Set scripts/training_failure_mining_control.json to `{"action":"reviewed_eval","reviewed_model":"halfsigma_raw"}`, then:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Output: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/halfsigma_raw/`. Frozen snapshot, selection metadata, workload, preparation/binding validation and complete per-case artifacts retained there.

Coordinator seconds (cache/inference work differs; not a model-speed comparison): lowres=23.18, fullres=26.50, gaussian=49.92, halfsigma_raw=245.12.

Validation:8focused Python tests; independent plan review; identical per-case reference selection/lengths; all182cases complete; effective embedding/reverse settings and saved per-case check counters verified disabled. No native or scheduling changes.
