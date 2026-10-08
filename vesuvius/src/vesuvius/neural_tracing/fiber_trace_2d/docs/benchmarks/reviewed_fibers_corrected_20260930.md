# Corrected reviewed-reference evaluation — 2026-09-30

All91 annotations have `hendrik_hq_approved`, no Lasagna spans, one eligible piece each:182directions. Five scroll counts22/1/11/1/56. Reference length693.087803mm (1386.175605mm directed). This v3 cohort supersedes the earlier v2 comparison; archived v2 results are not directly comparable because reference geometry/eligibility changed.

Output: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3`.
Same frozen models, inference parameters, normal manifests, beam8/lookahead2/step4 prediction voxels,40/80base ellipse,160base grace,80base restart grouping. GPUs0–5 only, unchanged192-worker budget. Mean=directed distance/(restarts+1); quantiles of actual reference partitions including successful terminal segments. No rendering requested.

## Combined results

| Model | Restarts | Mean mm | Median mm | P5 mm | P95 mm | Coordinator s | New tiles |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lowres212757 best EMA999 | 29 | 46.206 | 5.035 | 1.445 | 21.856 | 23.185 | 0 |
| Fullres213753 best EMA999 | 36 | 37.464 | 4.780 | 1.380 | 23.660 | 26.501 | 0 |
| Gaussian092844 EMA999 u17700 | 32 | 42.005 | 4.942 | 1.371 | 19.780 | 49.921 | 157 |

Timings are cache-reusing measurements, not controlled model-speed comparisons. All182/182directions complete,zero infrastructure failures. Lowres retains fewest failures; Gaussian improves relative to previous fullres but does not beat lowres.

## Per-scroll results

| Scroll | Model | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 | Lowres212757 best EMA999 | 10 | 0.03506 | 25.929 | 4.521 | 1.371 | 10.638 |
| PHerc0139 | Fullres213753 best EMA999 | 16 | 0.05610 | 16.778 | 3.593 | 1.365 | 10.480 |
| PHerc0139 | Gaussian092844 EMA999 u17700 | 12 | 0.04207 | 21.940 | 3.814 | 1.371 | 11.719 |
| PHerc0332 | Lowres212757 best EMA999 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 | Fullres213753 best EMA999 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc0332 | Gaussian092844 EMA999 u17700 | 0 | 0.00000 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc1299 | Lowres212757 best EMA999 | 4 | 0.02713 | 29.485 | 4.702 | 1.957 | 15.539 |
| PHerc1299 | Fullres213753 best EMA999 | 4 | 0.02713 | 29.485 | 4.811 | 1.535 | 15.539 |
| PHerc1299 | Gaussian092844 EMA999 u17700 | 4 | 0.02713 | 29.485 | 4.811 | 1.003 | 15.539 |
| PHerc1451 | Lowres212757 best EMA999 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 | Fullres213753 best EMA999 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHerc1451 | Gaussian092844 EMA999 u17700 | 0 | 0.00000 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHercParis4 | Lowres212757 best EMA999 | 15 | 0.01610 | 58.244 | 5.187 | 1.459 | 25.157 |
| PHercParis4 | Fullres213753 best EMA999 | 16 | 0.01717 | 54.818 | 5.111 | 1.472 | 25.602 |
| PHercParis4 | Gaussian092844 EMA999 u17700 | 16 | 0.01717 | 54.818 | 5.097 | 1.496 | 25.336 |

## Reproduction and validation

Set `scripts/training_failure_mining_control.json` to `{"action":"reviewed_eval","reviewed_model":"lowres"}` (or fullres/gaussian), then:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Seven focused Python tests pass. Independent rerun-plan review approved. Exact checkpoint hashes and producer inference settings verified equal to v2; all three current runs share the same frozen reference hashes, normal-manifest hashes and tracer parameters. No native or scheduling changes for this rerun. Gaussian is pinned to frozen u17700 rather than the live training best. Full source/model provenance and reader tests: [initial report](reviewed_fibers_20260930.md). Exact results: [JSON](reviewed_fibers_corrected_20260930.json).

This is a selected, reviewed hard-region cohort, not an independently established training holdout; training/selection overlap is not ruled out. Original annotations and prior results remain intact.
