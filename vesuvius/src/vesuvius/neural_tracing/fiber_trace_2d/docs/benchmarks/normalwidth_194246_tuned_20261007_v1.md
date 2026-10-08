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
| best_restarts raw u17600 | {} | 0 | 4 | 647 | 85.654 | 2.448 | 1.835 | 0.690 | 0.000 | 6.801 |
| best_restarts raw u17600 | {} | 512 | 4 | 264 | 76.844 | 8.322 | 4.035 | 3.083 | 1.354 | 10.128 |
| current_ema99 EMA99 u23300 | {} | 0 | 3 | 676 | 86.795 | 2.332 | 1.780 | 0.709 | 0.000 | 6.600 |
| current_ema99 EMA99 u23300 | {} | 512 | 3 | 263 | 77.076 | 8.094 | 4.062 | 3.351 | 1.311 | 9.011 |
| best_loss_ema99 EMA99 u19100 | {} | 0 | 3 | 600 | 87.590 | 2.766 | 2.024 | 0.739 | 0.000 | 7.130 |
| best_loss_ema99 EMA99 u19100 | {} | 512 | 3 | 258 | 80.681 | 9.725 | 4.335 | 3.376 | 1.380 | 9.721 |
| current_ema999 EMA999 u23300 | {} | 0 | 5 | 645 | 85.776 | 2.452 | 1.843 | 0.776 | 0.000 | 6.581 |
| current_ema999 EMA999 u23300 | {} | 512 | 5 | 270 | 77.765 | 7.926 | 3.992 | 3.124 | 1.306 | 8.925 |
| best_loss_ema999 EMA999 u19300 | {} | 0 | 4 | 580 | 87.123 | 2.882 | 2.082 | 0.836 | 0.000 | 7.132 |
| best_loss_ema999 EMA999 u19300 | {} | 512 | 4 | 251 | 79.249 | 10.078 | 4.377 | 3.282 | 1.476 | 9.697 |

## PHerc0139

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous EMA999 u12100 | {"step-voxels": 6} | 0 | 0 | 273 | 91.032 | 1.105 | 0.951 | 0.350 | 0.000 | 3.358 |
| Previous EMA999 u12100 | {"step-voxels": 6} | 512 | 0 | 60 | 66.395 | 5.739 | 3.156 | 2.066 | 1.386 | 8.883 |
| current raw u23300 | {} | 0 | 0 | 272 | 85.755 | 1.028 | 0.899 | 0.256 | 0.000 | 3.814 |
| current raw u23300 | {} | 512 | 0 | 62 | 65.531 | 5.497 | 3.015 | 2.396 | 1.334 | 8.403 |
| best_loss raw u20000 | {} | 0 | 1 | 295 | 90.444 | 1.004 | 0.874 | 0.315 | 0.000 | 2.788 |
| best_loss raw u20000 | {} | 512 | 0 | 70 | 67.028 | 4.446 | 2.731 | 1.939 | 1.324 | 6.602 |
| best_restarts raw u17600 | {} | 0 | 0 | 271 | 89.849 | 1.095 | 0.946 | 0.352 | 0.000 | 3.350 |
| best_restarts raw u17600 | {} | 512 | 0 | 69 | 68.968 | 4.575 | 2.851 | 1.824 | 1.326 | 7.597 |
| current_ema99 EMA99 u23300 | {} | 0 | 0 | 249 | 88.261 | 1.176 | 1.011 | 0.343 | 0.000 | 4.282 |
| current_ema99 EMA99 u23300 | {} | 512 | 0 | 69 | 70.063 | 4.874 | 2.896 | 2.110 | 1.270 | 7.991 |
| best_loss_ema99 EMA99 u19100 | {} | 0 | 0 | 238 | 90.368 | 1.282 | 1.083 | 0.461 | 0.000 | 4.134 |
| best_loss_ema99 EMA99 u19100 | {} | 512 | 0 | 70 | 71.140 | 4.831 | 2.899 | 2.000 | 1.310 | 8.056 |
| current_ema999 EMA999 u23300 | {} | 0 | 0 | 248 | 87.011 | 1.171 | 1.001 | 0.345 | 0.000 | 3.293 |
| current_ema999 EMA999 u23300 | {} | 512 | 0 | 73 | 69.842 | 4.331 | 2.729 | 2.031 | 1.280 | 7.475 |
| best_loss_ema999 EMA999 u19300 | {} | 0 | 0 | 243 | 89.958 | 1.246 | 1.056 | 0.518 | 0.000 | 3.366 |
| best_loss_ema999 EMA999 u19300 | {} | 512 | 0 | 70 | 68.419 | 4.646 | 2.788 | 1.896 | 1.304 | 8.081 |

## PHerc0332

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous EMA999 u12100 | {"step-voxels": 6} | 0 | 0 | 8 | 31.915 | 0.444 | 0.444 | 0.333 | 0.021 | 1.038 |
| Previous EMA999 u12100 | {"step-voxels": 6} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| current raw u23300 | {} | 0 | 0 | 7 | 83.200 | 1.543 | 1.323 | 0.103 | 0.009 | 4.489 |
| current raw u23300 | {} | 512 | 0 | 2 | 80.234 | 4.464 | 4.464 | 4.464 | 4.409 | 4.520 |
| best_loss raw u20000 | {} | 0 | 0 | 9 | 81.698 | 1.136 | 1.010 | 0.206 | 0.147 | 2.735 |
| best_loss raw u20000 | {} | 512 | 0 | 4 | 73.919 | 2.057 | 2.057 | 2.092 | 1.322 | 2.741 |
| best_restarts raw u17600 | {} | 0 | 0 | 19 | 55.225 | 0.341 | 0.323 | 0.116 | 0.000 | 1.650 |
| best_restarts raw u17600 | {} | 512 | 0 | 2 | 29.845 | 1.107 | 1.661 | 1.661 | 1.648 | 1.673 |
| current_ema99 EMA99 u23300 | {} | 0 | 0 | 17 | 52.610 | 0.344 | 0.344 | 0.174 | 0.021 | 1.304 |
| current_ema99 EMA99 u23300 | {} | 512 | 0 | 2 | 23.458 | 0.870 | 1.305 | 1.305 | 1.304 | 1.307 |
| best_loss_ema99 EMA99 u19100 | {} | 0 | 0 | 8 | 34.362 | 0.478 | 0.478 | 0.012 | 0.000 | 1.886 |
| best_loss_ema99 EMA99 u19100 | {} | 512 | 0 | 2 | 33.886 | 1.257 | 1.886 | 1.886 | 1.885 | 1.886 |
| current_ema999 EMA999 u23300 | {} | 0 | 0 | 10 | 39.852 | 0.403 | 0.443 | 0.188 | 0.000 | 1.586 |
| current_ema999 EMA999 u23300 | {} | 512 | 0 | 2 | 28.470 | 1.056 | 1.584 | 1.584 | 1.571 | 1.597 |
| best_loss_ema999 EMA999 u19300 | {} | 0 | 0 | 5 | 33.058 | 0.613 | 0.736 | 0.201 | 0.000 | 1.809 |
| best_loss_ema999 EMA999 u19300 | {} | 512 | 0 | 2 | 31.255 | 1.159 | 1.739 | 1.739 | 1.634 | 1.844 |

## PHerc1299

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous EMA999 u12100 | {"step-voxels": 6} | 0 | 0 | 24 | 99.807 | 49.048 | 6.131 | 5.033 | 2.434 | 16.375 |
| Previous EMA999 u12100 | {"step-voxels": 6} | 512 | 0 | 23 | 99.807 | 73.572 | 6.398 | 5.033 | 3.516 | 16.793 |
| current raw u23300 | {} | 0 | 1 | 36 | 99.006 | 9.731 | 4.054 | 3.247 | 0.196 | 9.508 |
| current raw u23300 | {} | 512 | 1 | 32 | 98.638 | 12.118 | 4.544 | 3.816 | 1.707 | 11.674 |
| best_loss raw u20000 | {} | 0 | 2 | 26 | 99.608 | 29.370 | 5.648 | 5.257 | 0.816 | 11.271 |
| best_loss raw u20000 | {} | 512 | 2 | 24 | 99.274 | 36.589 | 6.098 | 5.481 | 2.564 | 11.538 |
| best_restarts raw u17600 | {} | 0 | 2 | 24 | 100.000 | 49.142 | 6.143 | 5.481 | 4.130 | 10.622 |
| best_restarts raw u17600 | {} | 512 | 2 | 23 | 99.676 | 48.983 | 6.389 | 5.481 | 4.240 | 10.702 |
| current_ema99 EMA99 u23300 | {} | 0 | 1 | 26 | 99.808 | 29.429 | 5.659 | 4.811 | 0.508 | 15.539 |
| current_ema99 EMA99 u23300 | {} | 512 | 1 | 24 | 99.808 | 49.048 | 6.131 | 5.033 | 2.090 | 16.375 |
| best_loss_ema99 EMA99 u19100 | {} | 0 | 1 | 26 | 99.784 | 29.422 | 5.658 | 5.033 | 0.901 | 11.172 |
| best_loss_ema99 EMA99 u19100 | {} | 512 | 1 | 24 | 99.473 | 36.662 | 6.110 | 5.257 | 3.574 | 11.427 |
| current_ema999 EMA999 u23300 | {} | 0 | 0 | 26 | 99.727 | 29.405 | 5.655 | 5.033 | 0.631 | 11.275 |
| current_ema999 EMA999 u23300 | {} | 512 | 0 | 24 | 99.667 | 48.979 | 6.122 | 5.257 | 3.421 | 11.543 |
| best_loss_ema999 EMA999 u19300 | {} | 0 | 0 | 27 | 99.711 | 24.500 | 5.445 | 5.033 | 0.358 | 11.065 |
| best_loss_ema999 EMA999 u19300 | {} | 512 | 0 | 24 | 99.009 | 48.656 | 6.082 | 5.257 | 2.574 | 11.451 |

## PHerc1451

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous EMA999 u12100 | {"step-voxels": 6} | 0 | 0 | 5 | 95.320 | 2.501 | 2.001 | 1.271 | 0.590 | 4.175 |
| Previous EMA999 u12100 | {"step-voxels": 6} | 512 | 0 | 3 | 83.980 | 2.938 | 2.938 | 3.099 | 1.454 | 4.310 |
| current raw u23300 | {} | 0 | 0 | 4 | 97.255 | 3.403 | 2.552 | 2.552 | 1.156 | 3.948 |
| current raw u23300 | {} | 512 | 0 | 2 | 75.201 | 3.946 | 3.946 | 3.946 | 3.944 | 3.948 |
| best_loss raw u20000 | {} | 0 | 0 | 3 | 99.452 | 5.219 | 3.479 | 3.691 | 1.719 | 5.092 |
| best_loss raw u20000 | {} | 512 | 0 | 3 | 99.452 | 5.219 | 3.479 | 3.691 | 1.719 | 5.092 |
| best_restarts raw u17600 | {} | 0 | 0 | 5 | 87.699 | 2.301 | 1.841 | 1.861 | 0.468 | 3.760 |
| best_restarts raw u17600 | {} | 512 | 0 | 3 | 78.519 | 2.747 | 2.747 | 2.241 | 1.899 | 3.950 |
| current_ema99 EMA99 u23300 | {} | 0 | 0 | 9 | 93.164 | 1.222 | 1.086 | 0.695 | 0.127 | 2.701 |
| current_ema99 EMA99 u23300 | {} | 512 | 0 | 2 | 49.930 | 1.747 | 2.620 | 2.620 | 2.257 | 2.984 |
| best_loss_ema99 EMA99 u19100 | {} | 0 | 0 | 5 | 93.282 | 1.958 | 1.958 | 0.458 | 0.086 | 4.464 |
| best_loss_ema99 EMA99 u19100 | {} | 512 | 0 | 2 | 84.817 | 4.451 | 4.451 | 4.451 | 4.432 | 4.470 |
| current_ema999 EMA999 u23300 | {} | 0 | 0 | 7 | 94.819 | 1.659 | 1.422 | 0.632 | 0.115 | 3.884 |
| current_ema999 EMA999 u23300 | {} | 512 | 0 | 3 | 80.765 | 2.826 | 2.826 | 2.997 | 1.394 | 4.137 |
| best_loss_ema999 EMA999 u19300 | {} | 0 | 0 | 3 | 99.456 | 5.219 | 3.480 | 4.517 | 1.058 | 5.175 |
| best_loss_ema999 EMA999 u19300 | {} | 512 | 0 | 2 | 93.032 | 4.882 | 4.882 | 4.882 | 4.553 | 5.211 |

## PHercParis4

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous EMA999 u12100 | {"step-voxels": 6} | 0 | 0 | 276 | 88.564 | 4.798 | 2.990 | 1.618 | 0.000 | 9.977 |
| Previous EMA999 u12100 | {"step-voxels": 6} | 512 | 0 | 147 | 85.044 | 15.241 | 5.391 | 4.737 | 1.604 | 12.103 |
| current raw u23300 | {} | 0 | 3 | 517 | 88.190 | 1.961 | 1.590 | 0.604 | 0.000 | 6.363 |
| current raw u23300 | {} | 512 | 3 | 193 | 77.228 | 5.899 | 3.729 | 2.558 | 1.314 | 7.707 |
| best_loss raw u20000 | {} | 0 | 1 | 412 | 86.875 | 2.595 | 1.965 | 0.621 | 0.000 | 7.830 |
| best_loss raw u20000 | {} | 512 | 1 | 169 | 79.270 | 8.491 | 4.371 | 3.057 | 1.532 | 11.289 |
| best_restarts raw u17600 | {} | 0 | 2 | 328 | 82.442 | 3.340 | 2.342 | 1.271 | 0.000 | 7.932 |
| best_restarts raw u17600 | {} | 512 | 2 | 167 | 76.185 | 8.875 | 4.251 | 3.372 | 1.395 | 10.188 |
| current_ema99 EMA99 u23300 | {} | 0 | 2 | 375 | 84.623 | 2.857 | 2.103 | 0.922 | 0.000 | 6.754 |
| current_ema99 EMA99 u23300 | {} | 512 | 2 | 166 | 76.573 | 8.297 | 4.299 | 3.579 | 1.377 | 8.910 |
| best_loss_ema99 EMA99 u19100 | {} | 0 | 2 | 323 | 85.382 | 3.552 | 2.463 | 1.155 | 0.000 | 8.120 |
| best_loss_ema99 EMA99 u19100 | {} | 512 | 2 | 160 | 81.140 | 11.120 | 4.726 | 3.597 | 1.501 | 11.242 |
| current_ema999 EMA999 u23300 | {} | 0 | 5 | 354 | 83.638 | 3.057 | 2.202 | 1.022 | 0.000 | 6.730 |
| current_ema999 EMA999 u23300 | {} | 512 | 5 | 168 | 77.280 | 8.473 | 4.287 | 3.595 | 1.351 | 8.927 |
| best_loss_ema999 EMA999 u19300 | {} | 0 | 4 | 302 | 84.770 | 3.892 | 2.616 | 1.324 | 0.000 | 8.120 |
| best_loss_ema999 EMA999 u19300 | {} | 512 | 4 | 153 | 79.855 | 11.812 | 4.864 | 3.596 | 1.532 | 11.378 |

## Reproduce

Set control action `reviewed_194246`. Frozen snapshot selections are retained on resume.

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Immutable workloads, native effective config, timing, outputs and per-piece decisions: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/normalwidth_194246_tuned_20261007_v1`. JSON includes source/control paths, checkpoint and inspected native executable hash; the latter describes the binary present at analysis, not independent proof of historical cached-run provenance. Timings include startup and any on-demand cache fill, not isolated tracing speeds.
