# Normal/width/HV models with full-prefix FW/BW consistency

91 reviewed fibers, 182 directions, five scrolls. Identical frozen cohort and geometry to the previous raw u40500 baseline. Beam8/lookahead2; forward ellipse40/80 base voxels, excursion grace160. Reverse full prefix every512 base voxels and at endpoints; reverse ellipse20/40, no reversed history, no embedding. New auxiliary model heads are loaded but do not change the existing presence/direction tracing inputs.

Error restarts are geometric episodes grouped within80 base voxels. Filtered stops are ungrouped. Actual stops counts individual geometric failures plus filtered stops, before episode grouping. Mean = retained GT mm / (actual stops +1), treating the cohort as one connected line as requested. Median and P05/P95 describe actual retained spans, including terminal and zero-length spans. GT still controls ordinary failure endpoints/reseeding; terminal failure→filter conversions are oracle-timed diagnostics, not GT-free detection.

| Model | Error restarts | Filtered stops | Actual stops | Retained % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous raw u40500 | 5 | 66 | 71 | 94.7436 | 18.2405 | 4.0507 | 0.3196 | 15.6663 |
| 083922 best-restarts raw u9800 | 7 | 159 | 166 | 87.4005 | 7.2546 | 2.4597 | 0.0000 | 11.8598 |
| 105948 best-loss raw u10800 | 8 | 213 | 221 | 82.4401 | 5.1476 | 2.0352 | 0.0000 | 9.5899 |
| 105948 current raw u11000 | 3 | 233 | 236 | 81.1133 | 4.7442 | 1.2645 | 0.0000 | 9.0643 |

## Per scroll

| Scroll / model | Error restarts | Filtered stops | Actual stops | Retained % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PHerc0139 / Previous raw u40500 | 1 | 20 | 21 | 92.9703 | 12.0532 | 3.2085 | 0.5897 | 9.8238 |
| PHerc0139 / 083922 best-restarts raw u9800 | 2 | 30 | 32 | 89.6558 | 7.7490 | 2.4597 | 0.0000 | 9.9715 |
| PHerc0139 / 105948 best-loss raw u10800 | 1 | 54 | 55 | 81.6310 | 4.1577 | 1.2697 | 0.0000 | 8.9466 |
| PHerc0139 / 105948 current raw u11000 | 0 | 48 | 48 | 83.0354 | 4.8334 | 1.3177 | 0.0000 | 8.8833 |
| PHerc0332 / Previous raw u40500 | 0 | 0 | 0 | 100.0000 | 11.1287 | 5.5644 | 5.5644 | 5.5644 |
| PHerc0332 / 083922 best-restarts raw u9800 | 0 | 5 | 5 | 52.3660 | 0.9713 | 1.2256 | 0.0000 | 1.9138 |
| PHerc0332 / 105948 best-loss raw u10800 | 0 | 5 | 5 | 50.7730 | 0.9417 | 0.8629 | 0.0000 | 2.0015 |
| PHerc0332 / 105948 current raw u11000 | 0 | 7 | 7 | 28.8004 | 0.4006 | 0.0000 | 0.0000 | 1.7158 |
| PHerc1299 / Previous raw u40500 | 2 | 11 | 13 | 91.5965 | 9.6456 | 4.1436 | 0.0000 | 10.3649 |
| PHerc1299 / 083922 best-restarts raw u9800 | 2 | 8 | 10 | 94.0627 | 12.6067 | 4.2399 | 0.1788 | 11.8146 |
| PHerc1299 / 105948 best-loss raw u10800 | 0 | 7 | 7 | 94.9125 | 17.4909 | 4.3715 | 1.4900 | 12.3457 |
| PHerc1299 / 105948 current raw u11000 | 0 | 4 | 4 | 96.5907 | 28.4802 | 4.5889 | 1.5653 | 9.2657 |
| PHerc1451 / Previous raw u40500 | 0 | 0 | 0 | 100.0000 | 10.4958 | 5.2479 | 5.2479 | 5.2479 |
| PHerc1451 / 083922 best-restarts raw u9800 | 0 | 1 | 1 | 88.2434 | 4.6309 | 2.7749 | 1.3926 | 5.0006 |
| PHerc1451 / 105948 best-loss raw u10800 | 0 | 4 | 4 | 56.6966 | 1.1901 | 0.5004 | 0.0000 | 2.4777 |
| PHerc1451 / 105948 current raw u11000 | 0 | 4 | 4 | 61.3923 | 1.2887 | 1.2421 | 0.0505 | 3.2120 |
| PHercParis4 / Previous raw u40500 | 2 | 35 | 37 | 95.6623 | 23.4600 | 4.8808 | 0.9419 | 18.3623 |
| PHercParis4 / 083922 best-restarts raw u9800 | 3 | 115 | 118 | 86.0651 | 6.7399 | 2.4287 | 0.0000 | 12.2693 |
| PHercParis4 / 105948 best-loss raw u10800 | 7 | 143 | 150 | 81.3826 | 5.0226 | 1.8582 | 0.0000 | 9.8349 |
| PHercParis4 / 105948 current raw u11000 | 3 | 170 | 173 | 78.9233 | 4.2269 | 1.2505 | 0.0000 | 9.2063 |

## Snapshots and reproduction

Code: fiber-exp5 after fast-forward merge of fiber-normal-width-hv at5d76a43ad. Neural follower integration was committed separately atba86eea12 on fiber-exp6. All requested raw checkpoints were frozen before evaluation; later training changes do not alter these selections.

Use control action `reviewed_normal_width` in `scripts/training_failure_mining_control.json`, then:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

GPUs0–5, existing manager-exclusive runner. Existing BF16 inference/float16 accumulation retained for matched baseline protocol. Wall time includes preparation, manager/resource waits and new inference; it is not isolated trace speed. Full per-case artifacts and immutable workloads: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3`. Machine-readable companion JSON contains source paths, hashes and full metrics.

| Selection | SHA256 | Wall s | Terminal failure→filtered |
|---|---|---:|---:|
| Previous raw u40500 | historical baseline | — | 8 |
| 083922 best-restarts raw u9800 | b92a76b0412ebe35e066caa2d88269ffee7913407014516aab1a938d00f3d8d9 | 308.2849547410151 | 41 |
| 105948 best-loss raw u10800 | 734fb226bd5ee73fff6ac9cd2f20c001376491780b9105c47b9b50e543351aec | 306.2153179399902 | 45 |
| 105948 current raw u11000 | 2408ece6ffa2251655709cf3987357f1892a573828929934266145f09f9d2eba | 303.5347902539652 | 55 |

Validation:498 native cases,4 queue regressions,5 configuration/report/reverse-prefix tests passed. Completed results additionally validate182cases/91eligible runs, exact source fiber/coordinate identities, effective reverse flags, and retained-span sums. The u9800 training-time selection was22restarts on32pieces/64directions; that is not the full benchmark score.
