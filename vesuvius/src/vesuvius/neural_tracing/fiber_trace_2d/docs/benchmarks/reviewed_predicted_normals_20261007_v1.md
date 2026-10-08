# Predicted versus Lasagna tracing normals

Only tracing normal source changes. GT acceptance and restart search keep Lasagna normals. Same frozen checkpoints, widths, cutoff and step6; FW/BW disabled in both winning controls.
Fresh auxiliary-normal caches required: elapsed times include inference and are not isolated tracing speed comparisons. Historical controls retained; matched-cache Lasagna controls reuse identical presence/direction/width bytes. GPU0–5, unchanged worker/batch settings.

Each named model is frozen independently; identical reviewed91fibers/182directions. No inference precision or resource changes. Beam8/lookahead2 and FW protocol fixed; changed parameters are explicit below. Defaults unchanged. The tracing policy was tuned on this benchmark, not independently validated; single-error differences may not generalize.

Step is in trace voxels; trace-to-base scale is2. Baseline4 means8base voxels. Step2/5/6/8 means4/10/12/16base voxels, nominal two-step lookahead8/20/24/32base voxels (baseline16). Changing step also changes the physical smoothing scale of cumulative history (history parameter4, normalized recursive mean, not a hard four-step window) and spacing of arithmetic-width-mean samples; these are not isolated weight-only experiments. Reverse-check settings are recorded in trace_args; disabled checks remain disabled.

Min512 is output-only filtering of actual traced-polyline lengths after rollback. Existing GT failures split pieces: this is GT-assisted retained-output analysis, not a GT-free error detector. Min0 online results remain available. Mean=retainedGTmm/(surviving individual stops+1); piece mean=retainedGTmm/surviving pieces. Coverage is always reported; fewer pieces alone is not a winner.

Non-dominated over min512 errors, surviving pieces and retained coverage: EMA999 u12100 Lasagna, EMA999 u12100 fiber normals

Complete cutoff/control comparison: [34 configurations, before/after512 filter](reviewed_ema999_short512_20261006_v1.md). Original online cutoff results: [post-modulation sweep](reviewed_postmod_presence_ema999_20261006_v1.md).

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA99 u19100 Lasagna | {} | 0 | 3 | 600 | 87.590 | 2.766 | 2.024 | 0.739 | 0.000 | 7.130 |
| EMA99 u19100 Lasagna | {} | 512 | 3 | 258 | 80.681 | 9.725 | 4.335 | 3.376 | 1.380 | 9.721 |
| EMA99 u19100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 0 | 3 | 597 | 87.594 | 2.798 | 2.034 | 0.760 | 0.000 | 7.491 |
| EMA99 u19100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 512 | 3 | 257 | 80.773 | 9.822 | 4.357 | 3.363 | 1.380 | 9.730 |
| EMA99 u19100 fiber normals | {"tracing_normals": "fiber"} | 0 | 4 | 589 | 87.649 | 2.852 | 2.063 | 0.803 | 0.000 | 7.341 |
| EMA99 u19100 fiber normals | {"tracing_normals": "fiber"} | 512 | 4 | 258 | 80.821 | 9.827 | 4.342 | 3.351 | 1.386 | 9.721 |
| EMA999 u12100 Lasagna | {"step-voxels": 6} | 0 | 0 | 586 | 89.864 | 2.980 | 2.126 | 0.665 | 0.000 | 7.770 |
| EMA999 u12100 Lasagna | {"step-voxels": 6} | 512 | 0 | 233 | 82.086 | 13.079 | 4.884 | 3.814 | 1.537 | 11.939 |
| EMA999 u12100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 0 | 1 | 588 | 89.806 | 2.950 | 2.117 | 0.678 | 0.000 | 7.747 |
| EMA999 u12100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 512 | 1 | 234 | 82.031 | 12.776 | 4.859 | 3.830 | 1.533 | 11.912 |
| EMA999 u12100 fiber normals | {"tracing_normals": "fiber"} | 0 | 3 | 581 | 89.018 | 2.959 | 2.124 | 0.697 | 0.000 | 8.120 |
| EMA999 u12100 fiber normals | {"tracing_normals": "fiber"} | 512 | 3 | 232 | 81.070 | 12.486 | 4.844 | 3.706 | 1.538 | 11.667 |

## PHerc0139

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA99 u19100 Lasagna | {} | 0 | 0 | 238 | 90.368 | 1.282 | 1.083 | 0.461 | 0.000 | 4.134 |
| EMA99 u19100 Lasagna | {} | 512 | 0 | 70 | 71.140 | 4.831 | 2.899 | 2.000 | 1.310 | 8.056 |
| EMA99 u19100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 0 | 0 | 236 | 90.384 | 1.302 | 1.092 | 0.466 | 0.000 | 4.180 |
| EMA99 u19100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 512 | 0 | 70 | 71.396 | 4.848 | 2.909 | 2.040 | 1.324 | 8.055 |
| EMA99 u19100 fiber normals | {"tracing_normals": "fiber"} | 0 | 0 | 231 | 90.414 | 1.336 | 1.116 | 0.518 | 0.000 | 4.219 |
| EMA99 u19100 fiber normals | {"tracing_normals": "fiber"} | 512 | 0 | 72 | 72.013 | 4.777 | 2.853 | 2.005 | 1.327 | 7.882 |
| EMA999 u12100 Lasagna | {"step-voxels": 6} | 0 | 0 | 273 | 91.032 | 1.105 | 0.951 | 0.350 | 0.000 | 3.358 |
| EMA999 u12100 Lasagna | {"step-voxels": 6} | 512 | 0 | 60 | 66.395 | 5.739 | 3.156 | 2.066 | 1.386 | 8.883 |
| EMA999 u12100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 0 | 0 | 268 | 91.111 | 1.130 | 0.970 | 0.397 | 0.000 | 3.363 |
| EMA999 u12100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 512 | 0 | 61 | 66.921 | 5.784 | 3.129 | 2.061 | 1.360 | 8.883 |
| EMA999 u12100 fiber normals | {"tracing_normals": "fiber"} | 0 | 2 | 256 | 91.385 | 1.190 | 1.018 | 0.405 | 0.000 | 3.377 |
| EMA999 u12100 fiber normals | {"tracing_normals": "fiber"} | 512 | 2 | 62 | 68.326 | 5.732 | 3.143 | 2.075 | 1.366 | 8.734 |

## PHerc0332

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA99 u19100 Lasagna | {} | 0 | 0 | 8 | 34.362 | 0.478 | 0.478 | 0.012 | 0.000 | 1.886 |
| EMA99 u19100 Lasagna | {} | 512 | 0 | 2 | 33.886 | 1.257 | 1.886 | 1.886 | 1.885 | 1.886 |
| EMA99 u19100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 0 | 0 | 8 | 34.615 | 0.482 | 0.482 | 0.027 | 0.000 | 1.886 |
| EMA99 u19100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 512 | 0 | 2 | 33.879 | 1.257 | 1.885 | 1.885 | 1.884 | 1.886 |
| EMA99 u19100 fiber normals | {"tracing_normals": "fiber"} | 0 | 0 | 8 | 34.093 | 0.474 | 0.474 | 0.000 | 0.000 | 1.886 |
| EMA99 u19100 fiber normals | {"tracing_normals": "fiber"} | 512 | 0 | 2 | 33.885 | 1.257 | 1.886 | 1.886 | 1.885 | 1.886 |
| EMA999 u12100 Lasagna | {"step-voxels": 6} | 0 | 0 | 8 | 31.915 | 0.444 | 0.444 | 0.333 | 0.021 | 1.038 |
| EMA999 u12100 Lasagna | {"step-voxels": 6} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| EMA999 u12100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 0 | 0 | 8 | 31.925 | 0.444 | 0.444 | 0.333 | 0.021 | 1.038 |
| EMA999 u12100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| EMA999 u12100 fiber normals | {"tracing_normals": "fiber"} | 0 | 0 | 7 | 32.448 | 0.516 | 0.516 | 0.666 | 0.026 | 1.039 |
| EMA999 u12100 fiber normals | {"tracing_normals": "fiber"} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |

## PHerc1299

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA99 u19100 Lasagna | {} | 0 | 1 | 26 | 99.784 | 29.422 | 5.658 | 5.033 | 0.901 | 11.172 |
| EMA99 u19100 Lasagna | {} | 512 | 1 | 24 | 99.473 | 36.662 | 6.110 | 5.257 | 3.574 | 11.427 |
| EMA99 u19100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 0 | 1 | 26 | 99.784 | 29.422 | 5.658 | 5.033 | 0.901 | 11.172 |
| EMA99 u19100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 512 | 1 | 24 | 99.472 | 36.662 | 6.110 | 5.257 | 3.574 | 11.427 |
| EMA99 u19100 fiber normals | {"tracing_normals": "fiber"} | 0 | 0 | 24 | 99.823 | 49.055 | 6.132 | 5.481 | 4.143 | 11.425 |
| EMA99 u19100 fiber normals | {"tracing_normals": "fiber"} | 512 | 0 | 23 | 99.823 | 73.583 | 6.399 | 5.481 | 4.240 | 11.552 |
| EMA999 u12100 Lasagna | {"step-voxels": 6} | 0 | 0 | 24 | 99.807 | 49.048 | 6.131 | 5.033 | 2.434 | 16.375 |
| EMA999 u12100 Lasagna | {"step-voxels": 6} | 512 | 0 | 23 | 99.807 | 73.572 | 6.398 | 5.033 | 3.516 | 16.793 |
| EMA999 u12100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 0 | 0 | 24 | 99.807 | 49.047 | 6.131 | 5.033 | 2.434 | 16.375 |
| EMA999 u12100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 512 | 0 | 23 | 99.807 | 73.571 | 6.397 | 5.033 | 3.516 | 16.793 |
| EMA999 u12100 fiber normals | {"tracing_normals": "fiber"} | 0 | 0 | 25 | 99.768 | 36.771 | 5.883 | 5.033 | 0.933 | 15.957 |
| EMA999 u12100 fiber normals | {"tracing_normals": "fiber"} | 512 | 0 | 23 | 99.363 | 48.829 | 6.369 | 5.033 | 3.495 | 16.793 |

## PHerc1451

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA99 u19100 Lasagna | {} | 0 | 0 | 5 | 93.282 | 1.958 | 1.958 | 0.458 | 0.086 | 4.464 |
| EMA99 u19100 Lasagna | {} | 512 | 0 | 2 | 84.817 | 4.451 | 4.451 | 4.451 | 4.432 | 4.470 |
| EMA99 u19100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 0 | 0 | 5 | 93.284 | 1.958 | 1.958 | 0.458 | 0.086 | 4.464 |
| EMA99 u19100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 512 | 0 | 2 | 84.819 | 4.451 | 4.451 | 4.451 | 4.432 | 4.470 |
| EMA99 u19100 fiber normals | {"tracing_normals": "fiber"} | 0 | 0 | 5 | 93.577 | 1.964 | 1.964 | 0.459 | 0.086 | 4.470 |
| EMA99 u19100 fiber normals | {"tracing_normals": "fiber"} | 512 | 0 | 2 | 85.107 | 4.466 | 4.466 | 4.466 | 4.461 | 4.472 |
| EMA999 u12100 Lasagna | {"step-voxels": 6} | 0 | 0 | 5 | 95.320 | 2.501 | 2.001 | 1.271 | 0.590 | 4.175 |
| EMA999 u12100 Lasagna | {"step-voxels": 6} | 512 | 0 | 3 | 83.980 | 2.938 | 2.938 | 3.099 | 1.454 | 4.310 |
| EMA999 u12100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 0 | 0 | 5 | 95.324 | 2.501 | 2.001 | 1.271 | 0.591 | 4.175 |
| EMA999 u12100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 512 | 0 | 3 | 83.973 | 2.938 | 2.938 | 3.099 | 1.454 | 4.309 |
| EMA999 u12100 fiber normals | {"tracing_normals": "fiber"} | 0 | 0 | 5 | 95.329 | 2.501 | 2.001 | 1.270 | 0.592 | 4.175 |
| EMA999 u12100 fiber normals | {"tracing_normals": "fiber"} | 512 | 0 | 3 | 83.976 | 2.938 | 2.938 | 3.099 | 1.453 | 4.310 |

## PHercParis4

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA99 u19100 Lasagna | {} | 0 | 2 | 323 | 85.382 | 3.552 | 2.463 | 1.155 | 0.000 | 8.120 |
| EMA99 u19100 Lasagna | {} | 512 | 2 | 160 | 81.140 | 11.120 | 4.726 | 3.597 | 1.501 | 11.242 |
| EMA99 u19100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 0 | 2 | 322 | 85.381 | 3.584 | 2.471 | 1.152 | 0.000 | 8.120 |
| EMA99 u19100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 512 | 2 | 159 | 81.199 | 11.294 | 4.759 | 3.597 | 1.493 | 11.297 |
| EMA99 u19100 fiber normals | {"tracing_normals": "fiber"} | 0 | 4 | 321 | 85.449 | 3.603 | 2.481 | 1.156 | 0.000 | 8.117 |
| EMA99 u19100 fiber normals | {"tracing_normals": "fiber"} | 512 | 4 | 159 | 81.023 | 11.104 | 4.749 | 3.597 | 1.488 | 11.296 |
| EMA999 u12100 Lasagna | {"step-voxels": 6} | 0 | 0 | 276 | 88.564 | 4.798 | 2.990 | 1.618 | 0.000 | 9.977 |
| EMA999 u12100 Lasagna | {"step-voxels": 6} | 512 | 0 | 147 | 85.044 | 15.241 | 5.391 | 4.737 | 1.604 | 12.103 |
| EMA999 u12100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 0 | 1 | 283 | 88.454 | 4.554 | 2.913 | 1.585 | 0.000 | 9.806 |
| EMA999 u12100 Lasagna matched cache | {"tracing_normals": "lasagna"} | 512 | 1 | 147 | 84.801 | 14.634 | 5.376 | 4.780 | 1.604 | 12.103 |
| EMA999 u12100 fiber normals | {"tracing_normals": "fiber"} | 0 | 1 | 288 | 87.197 | 4.345 | 2.822 | 1.257 | 0.000 | 9.806 |
| EMA999 u12100 fiber normals | {"tracing_normals": "fiber"} | 512 | 1 | 144 | 83.012 | 14.596 | 5.372 | 4.342 | 1.645 | 11.706 |

## Reproduce

Set control action `reviewed_predicted_normals`. Frozen snapshot selections are retained on resume.

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Immutable workloads, native effective config, timing, outputs and per-piece decisions: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/reviewed_predicted_normals_20261007_v1`. JSON includes source/control paths, checkpoint and inspected native executable hash; the latter describes the binary present at analysis, not independent proof of historical cached-run provenance. Timings include startup and any on-demand cache fill, not isolated tracing speeds.
