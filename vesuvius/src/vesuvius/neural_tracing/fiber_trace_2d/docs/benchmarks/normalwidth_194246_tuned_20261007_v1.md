# 194246 raw and EMA selections — winning tracing policy

194246 raw and EMA checkpoints, no additional tuning. All use postcutoff.4,cos40,noFW,step6trace=12base and original weights. Previous EMA999u12100 is the matched historical control. GPUs0–5; inference settings unchanged.
Best-restarts refers to the training selection protocol. Only one best_reviewed_trace selection exists; no separate EMA best-restarts files are available. Raw selections were frozen in the initial run; the four EMA selections were frozen before this continuation. Identical hashes mean identical selections, not independent checkpoints.

| Selection | Step | Samples | Selection metric | SHA256 | Captured UTC |
|---|---:|---:|---|---|---|
| current raw | 23300 | 2982400 | test/loss_total | f58ea7ce129b8f57a3882350f767e13987ed457ec4e07dfaa93582e35d28cbac | 2026-10-07T11:47:44.303546+00:00 |
| best_loss raw | 20000 | 2560000 | test/loss_total | b7da239165f7625181f8eefa11d3fe1f2af54e7f97b926a8445caee3a36c8315 | 2026-10-07T11:47:46.758187+00:00 |
| best_restarts raw | 17600 | 2252800 | reviewed_trace/restarts_combined | f834bfc47f0e76e3166f4bc6ceacfb91f2936548e203cc11be007dc660fcfa00 | 2026-10-07T11:47:49.169210+00:00 |
| current_ema99 EMA99 | 23300 | 2982400 | test_ema_0p99/loss_total | 8824a19c24edf74a1fde548f3d4f75d6ba7b5af573fc975e041ebd1ed6f72f91 | 2026-10-07T12:12:02.372603+00:00 |
| best_loss_ema99 EMA99 | 19100 | 2444800 | test_ema_0p99/loss_total | 760d07d8b86db07c959b0805a51dc26d803f86a00606151cf82dfc70f6f27906 | 2026-10-07T12:12:03.055647+00:00 |
| current_ema999 EMA999 | 23300 | 2982400 | test_ema_0p999/loss_total | 9841ecf40c92e366e689721af95939ce2f6e43cab32224e16e1307c41d507064 | 2026-10-07T12:12:03.691296+00:00 |
| best_loss_ema999 EMA999 | 19300 | 2470400 | test_ema_0p999/loss_total | b3035127f248352569c19721a36ed71bc59bfaf41c3b0047bc292e8edc32e369 | 2026-10-07T12:12:04.347904+00:00 |

Each named model is frozen independently; identical reviewed91fibers/182directions. No inference precision or resource changes. Beam8/lookahead2 and FW protocol fixed; changed parameters are explicit below. Defaults unchanged. The tracing policy was tuned on this benchmark, not independently validated; single-error differences may not generalize.

Step is in trace voxels; trace-to-base scale is2. Baseline4 means8base voxels. Step2/5/6/8 means4/10/12/16base voxels, nominal two-step lookahead8/20/24/32base voxels (baseline16). Changing step also changes the physical smoothing scale of cumulative history (history parameter4, normalized recursive mean, not a hard four-step window) and spacing of arithmetic-width-mean samples; these are not isolated weight-only experiments. Reverse check interval remains512base voxels, reached on committed steps.

Min512 is output-only filtering of actual traced-polyline lengths after rollback. Existing GT failures split pieces: this is GT-assisted retained-output analysis, not a GT-free error detector. Min0 online results remain available. Mean=retainedGTmm/(surviving individual stops+1); piece mean=retainedGTmm/surviving pieces. Coverage is always reported; fewer pieces alone is not a winner.

Non-dominated over min512 errors, surviving pieces and retained coverage: Previous EMA999 u12100

Complete cutoff/control comparison: [34 configurations, before/after512 filter](reviewed_ema999_short512_20261006_v1.md). Original online cutoff results: [post-modulation sweep](reviewed_postmod_presence_ema999_20261006_v1.md).

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous EMA999 u12100 | {"step-voxels": 6} | 0 | 0 | 586 | 89.864 | 2.980 | 2.126 | 0.665 | 0.000 | 7.770 |
| Previous EMA999 u12100 | {"step-voxels": 6} | 512 | 0 | 233 | 82.086 | 13.079 | 4.884 | 3.814 | 1.537 | 11.939 |
| current raw u23300 | {} | 0 | 4 | 836 | 88.868 | 1.820 | 1.474 | 0.544 | 0.000 | 5.759 |
| current raw u23300 | {} | 512 | 4 | 291 | 77.107 | 6.362 | 3.673 | 2.647 | 1.338 | 8.133 |
| best_loss raw u20000 | {} | 0 | 4 | 745 | 89.018 | 2.127 | 1.656 | 0.516 | 0.000 | 6.780 |
| best_loss raw u20000 | {} | 512 | 3 | 270 | 78.989 | 8.051 | 4.055 | 2.710 | 1.438 | 10.949 |

## PHerc0139

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous EMA999 u12100 | {"step-voxels": 6} | 0 | 0 | 273 | 91.032 | 1.105 | 0.951 | 0.350 | 0.000 | 3.358 |
| Previous EMA999 u12100 | {"step-voxels": 6} | 512 | 0 | 60 | 66.395 | 5.739 | 3.156 | 2.066 | 1.386 | 8.883 |
| current raw u23300 | {} | 0 | 0 | 272 | 85.755 | 1.028 | 0.899 | 0.256 | 0.000 | 3.814 |
| current raw u23300 | {} | 512 | 0 | 62 | 65.531 | 5.497 | 3.015 | 2.396 | 1.334 | 8.403 |
| best_loss raw u20000 | {} | 0 | 1 | 295 | 90.444 | 1.004 | 0.874 | 0.315 | 0.000 | 2.788 |
| best_loss raw u20000 | {} | 512 | 0 | 70 | 67.028 | 4.446 | 2.731 | 1.939 | 1.324 | 6.602 |

## PHerc0332

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous EMA999 u12100 | {"step-voxels": 6} | 0 | 0 | 8 | 31.915 | 0.444 | 0.444 | 0.333 | 0.021 | 1.038 |
| Previous EMA999 u12100 | {"step-voxels": 6} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| current raw u23300 | {} | 0 | 0 | 7 | 83.200 | 1.543 | 1.323 | 0.103 | 0.009 | 4.489 |
| current raw u23300 | {} | 512 | 0 | 2 | 80.234 | 4.464 | 4.464 | 4.464 | 4.409 | 4.520 |
| best_loss raw u20000 | {} | 0 | 0 | 9 | 81.698 | 1.136 | 1.010 | 0.206 | 0.147 | 2.735 |
| best_loss raw u20000 | {} | 512 | 0 | 4 | 73.919 | 2.057 | 2.057 | 2.092 | 1.322 | 2.741 |

## PHerc1299

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous EMA999 u12100 | {"step-voxels": 6} | 0 | 0 | 24 | 99.807 | 49.048 | 6.131 | 5.033 | 2.434 | 16.375 |
| Previous EMA999 u12100 | {"step-voxels": 6} | 512 | 0 | 23 | 99.807 | 73.572 | 6.398 | 5.033 | 3.516 | 16.793 |
| current raw u23300 | {} | 0 | 1 | 36 | 99.006 | 9.731 | 4.054 | 3.247 | 0.196 | 9.508 |
| current raw u23300 | {} | 512 | 1 | 32 | 98.638 | 12.118 | 4.544 | 3.816 | 1.707 | 11.674 |
| best_loss raw u20000 | {} | 0 | 2 | 26 | 99.608 | 29.370 | 5.648 | 5.257 | 0.816 | 11.271 |
| best_loss raw u20000 | {} | 512 | 2 | 24 | 99.274 | 36.589 | 6.098 | 5.481 | 2.564 | 11.538 |

## PHerc1451

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous EMA999 u12100 | {"step-voxels": 6} | 0 | 0 | 5 | 95.320 | 2.501 | 2.001 | 1.271 | 0.590 | 4.175 |
| Previous EMA999 u12100 | {"step-voxels": 6} | 512 | 0 | 3 | 83.980 | 2.938 | 2.938 | 3.099 | 1.454 | 4.310 |
| current raw u23300 | {} | 0 | 0 | 4 | 97.255 | 3.403 | 2.552 | 2.552 | 1.156 | 3.948 |
| current raw u23300 | {} | 512 | 0 | 2 | 75.201 | 3.946 | 3.946 | 3.946 | 3.944 | 3.948 |
| best_loss raw u20000 | {} | 0 | 0 | 3 | 99.452 | 5.219 | 3.479 | 3.691 | 1.719 | 5.092 |
| best_loss raw u20000 | {} | 512 | 0 | 3 | 99.452 | 5.219 | 3.479 | 3.691 | 1.719 | 5.092 |

## PHercParis4

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous EMA999 u12100 | {"step-voxels": 6} | 0 | 0 | 276 | 88.564 | 4.798 | 2.990 | 1.618 | 0.000 | 9.977 |
| Previous EMA999 u12100 | {"step-voxels": 6} | 512 | 0 | 147 | 85.044 | 15.241 | 5.391 | 4.737 | 1.604 | 12.103 |
| current raw u23300 | {} | 0 | 3 | 517 | 88.190 | 1.961 | 1.590 | 0.604 | 0.000 | 6.363 |
| current raw u23300 | {} | 512 | 3 | 193 | 77.228 | 5.899 | 3.729 | 2.558 | 1.314 | 7.707 |
| best_loss raw u20000 | {} | 0 | 1 | 412 | 86.875 | 2.595 | 1.965 | 0.621 | 0.000 | 7.830 |
| best_loss raw u20000 | {} | 512 | 1 | 169 | 79.270 | 8.491 | 4.371 | 3.057 | 1.532 | 11.289 |

## Reproduce

Set control action `reviewed_194246`. Frozen snapshot selections are retained on resume.

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Immutable workloads, native effective config, timing, outputs and per-piece decisions: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/normalwidth_194246_tuned_20261007_v1`. JSON includes source/control paths, checkpoint and inspected native executable hash; the latter describes the binary present at analysis, not independent proof of historical cached-run provenance. Timings include startup and any on-demand cache fill, not isolated tracing speeds.
