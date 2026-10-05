# Reverse checks to segment start — raw u40500

Same frozen model/cache, 91 reviewed fibers / 182 directions, beam8, reverse ellipse20/40. Window checks only the latest accepted interval; prefix checks the entire current forward segment. All terminal endpoints are checked unless already checked at the same point. One-point segments cannot be checked. Intervals are forward arc travel in base voxels, checked at the first committed step reaching the interval. No convergence shortcut.

History H is a directional prior: negative last H forward edges, oldest to newest, folded with the existing rolling smoothness rule (window4). The most recent edge seeds previous-step direction; model direction remains separate. H0 preserves the cold model-direction seed. History samples do not consume search depth or fabricate path points. Every reverse step is freely inferred and checked. Checkpoints use actual committed vertices. This is not exact reverse curvature history (future forward points beyond terminal endpoints are unavailable). H1 isolates actual-edge initialization. Lower filtered counts may weaken rejection; choose only among H1/4/8 with <=5ordinary failures, then minimize filtered stops, tie-breaking by retained coverage. If none qualifies at256, test the least-error candidate at128/512 as a bounded follow-up, without calling it a winner. Selection is tuned on this cohort, not independent validation.

Acceptance uses forward trace geometry only, never GT. However GT still determines ordinary evaluation failure endpoints and reseeding. Thus terminal failure-to-filter conversions are oracle-timed diagnostics, not demonstrated deployable GT-free detection. Retained coverage excludes rejected/skipped intervals. Ordinary restart episodes use existing80-voxel grouping; filtered stops are ungrouped.

| Variant | Ordinary restarts | Filtered stops | Total interruptions | Retained % | Reverse checks | Terminal failure→filtered | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| unfiltered | 16 | 0 | 16 | 100.0000 | 0 | 0 | — |
| window256_original | 13 | 36 | 49 | 98.4381 | 2313 | 0 | — |
| old_prefix128 | 5 | 78 | 83 | 98.1783 | 4480 | 3 | 349.8 |
| old_prefix256 | 5 | 75 | 80 | 96.8330 | 2324 | 8 | 217.1 |
| old_prefix512 | 5 | 66 | 71 | 94.7436 | 1224 | 8 | 138.9 |
| prefix256_history0 | 5 | 75 | 80 | 96.8330 | 2324 | 8 | 196.5 |
| prefix256_history1 | 7 | 69 | 76 | 97.0755 | 2325 | 6 | 177.7 |
| prefix256_history4 | 6 | 70 | 76 | 96.9926 | 2323 | 6 | 168.5 |
| prefix256_history8 | 6 | 69 | 75 | 97.0404 | 2323 | 6 | 189.4 |
| prefix128_history8 | 5 | 72 | 77 | 98.3165 | 4484 | 5 | 302.2 |
| prefix512_history8 | 6 | 62 | 68 | 95.0853 | 1226 | 8 | 123.6 |

## Per scroll

| Scroll / variant | Ordinary restarts | Filtered stops | Retained % |
|---|---:|---:|---:|
| PHerc0139 / unfiltered | 8 | 0 | 100.0000 |
| PHerc0139 / window256_original | 6 | 12 | 97.5034 |
| PHerc0139 / old_prefix128 | 1 | 27 | 97.0078 |
| PHerc0139 / old_prefix256 | 1 | 26 | 94.9065 |
| PHerc0139 / old_prefix512 | 1 | 20 | 92.9703 |
| PHerc0139 / prefix256_history0 | 1 | 26 | 94.9065 |
| PHerc0139 / prefix256_history1 | 3 | 21 | 95.8703 |
| PHerc0139 / prefix256_history4 | 2 | 22 | 95.6279 |
| PHerc0139 / prefix256_history8 | 2 | 21 | 95.8601 |
| PHerc0139 / prefix128_history8 | 1 | 22 | 97.5141 |
| PHerc0139 / prefix512_history8 | 2 | 18 | 93.6907 |
| PHerc0332 / unfiltered | 0 | 0 | 100.0000 |
| PHerc0332 / window256_original | 0 | 0 | 100.0000 |
| PHerc0332 / old_prefix128 | 0 | 0 | 100.0000 |
| PHerc0332 / old_prefix256 | 0 | 0 | 100.0000 |
| PHerc0332 / old_prefix512 | 0 | 0 | 100.0000 |
| PHerc0332 / prefix256_history0 | 0 | 0 | 100.0000 |
| PHerc0332 / prefix256_history1 | 0 | 0 | 100.0000 |
| PHerc0332 / prefix256_history4 | 0 | 0 | 100.0000 |
| PHerc0332 / prefix256_history8 | 0 | 0 | 100.0000 |
| PHerc0332 / prefix128_history8 | 0 | 0 | 100.0000 |
| PHerc0332 / prefix512_history8 | 0 | 0 | 100.0000 |
| PHerc1299 / unfiltered | 2 | 0 | 100.0000 |
| PHerc1299 / window256_original | 3 | 6 | 97.3917 |
| PHerc1299 / old_prefix128 | 2 | 11 | 97.4905 |
| PHerc1299 / old_prefix256 | 3 | 11 | 95.2558 |
| PHerc1299 / old_prefix512 | 2 | 11 | 91.5965 |
| PHerc1299 / prefix256_history0 | 3 | 11 | 95.2558 |
| PHerc1299 / prefix256_history1 | 3 | 11 | 95.2558 |
| PHerc1299 / prefix256_history4 | 3 | 11 | 95.2558 |
| PHerc1299 / prefix256_history8 | 3 | 11 | 95.2558 |
| PHerc1299 / prefix128_history8 | 2 | 11 | 97.4905 |
| PHerc1299 / prefix512_history8 | 2 | 10 | 92.4312 |
| PHerc1451 / unfiltered | 0 | 0 | 100.0000 |
| PHerc1451 / window256_original | 0 | 0 | 100.0000 |
| PHerc1451 / old_prefix128 | 0 | 0 | 100.0000 |
| PHerc1451 / old_prefix256 | 0 | 0 | 100.0000 |
| PHerc1451 / old_prefix512 | 0 | 0 | 100.0000 |
| PHerc1451 / prefix256_history0 | 0 | 0 | 100.0000 |
| PHerc1451 / prefix256_history1 | 0 | 0 | 100.0000 |
| PHerc1451 / prefix256_history4 | 0 | 0 | 100.0000 |
| PHerc1451 / prefix256_history8 | 0 | 0 | 100.0000 |
| PHerc1451 / prefix128_history8 | 0 | 0 | 100.0000 |
| PHerc1451 / prefix512_history8 | 0 | 0 | 100.0000 |
| PHercParis4 / unfiltered | 6 | 0 | 100.0000 |
| PHercParis4 / window256_original | 4 | 18 | 98.8535 |
| PHercParis4 / old_prefix128 | 2 | 40 | 98.6032 |
| PHercParis4 / old_prefix256 | 1 | 38 | 97.5987 |
| PHercParis4 / old_prefix512 | 2 | 35 | 95.6623 |
| PHercParis4 / prefix256_history0 | 1 | 38 | 97.5987 |
| PHercParis4 / prefix256_history1 | 1 | 37 | 97.6644 |
| PHercParis4 / prefix256_history4 | 1 | 37 | 97.6153 |
| PHercParis4 / prefix256_history8 | 1 | 37 | 97.6153 |
| PHercParis4 / prefix128_history8 | 2 | 39 | 98.6537 |
| PHercParis4 / prefix512_history8 | 2 | 34 | 95.8180 |

## Run

Set control action `reviewed_reverse_history`, then:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Workloads, output and timing: /ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/reverse_history_40500_v1. Recorded wall times include startup, cache filling and resource waits; these are not isolated tracer-speed benchmarks.
