# Reverse checks to segment start — raw u40500

Same frozen model/cache, 91 reviewed fibers / 182 directions, beam8, reverse ellipse20/40. Window checks only the latest accepted interval; prefix checks the entire current forward segment. All terminal endpoints are checked unless already checked at the same point. One-point segments cannot be checked. Intervals are forward arc travel in base voxels, checked at the first committed step reaching the interval. No convergence shortcut.

Acceptance uses forward trace geometry only, never GT. However GT still determines ordinary evaluation failure endpoints and reseeding. Thus terminal failure-to-filter conversions are oracle-timed diagnostics, not demonstrated deployable GT-free detection. Retained coverage excludes rejected/skipped intervals. Ordinary restart episodes use existing80-voxel grouping; filtered stops are ungrouped.

| Variant | Ordinary restarts | Filtered stops | Total interruptions | Retained % | Reverse checks | Terminal failure→filtered | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| unfiltered | 16 | 0 | 16 | 100.0000 | 0 | 0 | — |
| window256_original | 13 | 36 | 49 | 98.4381 | 2313 | 0 | — |
| window256_terminal | 8 | 41 | 49 | 98.3309 | 2326 | 8 | 74.6 |
| prefix256_terminal | 5 | 75 | 80 | 96.8330 | 2324 | 8 | 217.1 |
| prefix128_terminal | 5 | 78 | 83 | 98.1783 | 4480 | 3 | 349.8 |
| prefix512_terminal | 5 | 66 | 71 | 94.7436 | 1224 | 8 | 138.9 |

## Per scroll

| Scroll / variant | Ordinary restarts | Filtered stops | Retained % |
|---|---:|---:|---:|
| PHerc0139 / unfiltered | 8 | 0 | 100.0000 |
| PHerc0139 / window256_original | 6 | 12 | 97.5034 |
| PHerc0139 / window256_terminal | 4 | 14 | 97.3249 |
| PHerc0139 / prefix256_terminal | 1 | 26 | 94.9065 |
| PHerc0139 / prefix128_terminal | 1 | 27 | 97.0078 |
| PHerc0139 / prefix512_terminal | 1 | 20 | 92.9703 |
| PHerc0332 / unfiltered | 0 | 0 | 100.0000 |
| PHerc0332 / window256_original | 0 | 0 | 100.0000 |
| PHerc0332 / window256_terminal | 0 | 0 | 100.0000 |
| PHerc0332 / prefix256_terminal | 0 | 0 | 100.0000 |
| PHerc0332 / prefix128_terminal | 0 | 0 | 100.0000 |
| PHerc0332 / prefix512_terminal | 0 | 0 | 100.0000 |
| PHerc1299 / unfiltered | 2 | 0 | 100.0000 |
| PHerc1299 / window256_original | 3 | 6 | 97.3917 |
| PHerc1299 / window256_terminal | 2 | 7 | 97.0159 |
| PHerc1299 / prefix256_terminal | 3 | 11 | 95.2558 |
| PHerc1299 / prefix128_terminal | 2 | 11 | 97.4905 |
| PHerc1299 / prefix512_terminal | 2 | 11 | 91.5965 |
| PHerc1451 / unfiltered | 0 | 0 | 100.0000 |
| PHerc1451 / window256_original | 0 | 0 | 100.0000 |
| PHerc1451 / window256_terminal | 0 | 0 | 100.0000 |
| PHerc1451 / prefix256_terminal | 0 | 0 | 100.0000 |
| PHerc1451 / prefix128_terminal | 0 | 0 | 100.0000 |
| PHerc1451 / prefix512_terminal | 0 | 0 | 100.0000 |
| PHercParis4 / unfiltered | 6 | 0 | 100.0000 |
| PHercParis4 / window256_original | 4 | 18 | 98.8535 |
| PHercParis4 / window256_terminal | 2 | 20 | 98.8081 |
| PHercParis4 / prefix256_terminal | 1 | 38 | 97.5987 |
| PHercParis4 / prefix128_terminal | 2 | 40 | 98.6032 |
| PHercParis4 / prefix512_terminal | 2 | 35 | 95.6623 |

## Run

Set control action `reviewed_reverse_prefix`, then:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Workloads, output and timing: /ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/reverse_prefix_40500_v1. Historical timing is omitted; shared-cache wall times are not standalone inference benchmarks.
