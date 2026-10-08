# Targeted tracing tuning — EMA999 u12100

Same frozen model, reviewed91fibers/182directions and inference cache. No inference precision or resource changes. Beam8/lookahead2 and FW protocol fixed; changed parameters are explicit below. Defaults unchanged. These settings are tuned on this benchmark, not independently validated; single-error differences may not generalize.

Step is in trace voxels; trace-to-base scale is2. Baseline4 means8base voxels. Step2/5/6/8 means4/10/12/16base voxels, nominal two-step lookahead8/20/24/32base voxels (baseline16). Changing step also changes the physical smoothing scale of cumulative history (history parameter4, normalized recursive mean, not a hard four-step window) and spacing of arithmetic-width-mean samples; these are not isolated weight-only experiments. Reverse check interval remains512base voxels, reached on committed steps.

Min512 is output-only filtering of actual traced-polyline lengths after rollback. Existing GT failures split pieces: this is GT-assisted retained-output analysis, not a GT-free error detector. Min0 online results remain available. Mean=retainedGTmm/(surviving individual stops+1); piece mean=retainedGTmm/surviving pieces. Coverage is always reported; fewer pieces alone is not a winner.

Non-dominated over min512 errors, surviving pieces and retained coverage: p40_cos40_norev_step6

Complete cutoff/control comparison: [34 configurations, before/after512 filter](reviewed_ema999_short512_20261006_v1.md). Original online cutoff results: [post-modulation sweep](reviewed_postmod_presence_ema999_20261006_v1.md).

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p40_cos40_norev_baseline | {} | 0 | 3 | 634 | 89.847 | 2.673 | 1.964 | 0.576 | 0.000 | 7.012 |
| p40_cos40_norev_baseline | {} | 512 | 3 | 234 | 81.333 | 12.255 | 4.818 | 3.757 | 1.555 | 11.859 |
| p40_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 0 | 4 | 615 | 89.603 | 2.772 | 2.020 | 0.600 | 0.000 | 7.567 |
| p40_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 512 | 4 | 237 | 81.813 | 12.065 | 4.785 | 3.814 | 1.551 | 11.671 |
| p40_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 0 | 5 | 679 | 89.325 | 2.423 | 1.824 | 0.421 | 0.000 | 6.911 |
| p40_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 512 | 5 | 238 | 81.209 | 11.975 | 4.730 | 3.647 | 1.440 | 11.791 |
| p40_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 0 | 3 | 624 | 89.905 | 2.739 | 1.997 | 0.579 | 0.000 | 7.443 |
| p40_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 512 | 3 | 234 | 81.731 | 12.588 | 4.842 | 3.735 | 1.546 | 11.906 |
| p40_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 0 | 4 | 622 | 89.260 | 2.725 | 1.989 | 0.579 | 0.000 | 7.181 |
| p40_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 512 | 4 | 237 | 81.301 | 12.118 | 4.755 | 3.596 | 1.421 | 11.824 |
| p40_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 0 | 4 | 605 | 89.634 | 2.837 | 2.054 | 0.611 | 0.000 | 8.028 |
| p40_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 512 | 4 | 234 | 81.871 | 12.203 | 4.850 | 3.830 | 1.556 | 11.654 |
| p40_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 0 | 4 | 645 | 88.972 | 2.575 | 1.912 | 0.578 | 0.000 | 7.173 |
| p40_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 512 | 4 | 233 | 79.973 | 12.050 | 4.758 | 3.597 | 1.515 | 11.922 |
| p40_cos40_norev_step2 | {"step-voxels": 2} | 0 | 7 | 938 | 88.686 | 1.588 | 1.311 | 0.352 | 0.000 | 5.759 |
| p40_cos40_norev_step2 | {"step-voxels": 2} | 512 | 5 | 264 | 75.270 | 7.400 | 3.952 | 3.127 | 1.376 | 10.383 |
| p40_cos40_norev_step6 | {"step-voxels": 6} | 0 | 0 | 586 | 89.864 | 2.980 | 2.126 | 0.665 | 0.000 | 7.770 |
| p40_cos40_norev_step6 | {"step-voxels": 6} | 512 | 0 | 233 | 82.086 | 13.079 | 4.884 | 3.814 | 1.537 | 11.939 |
| p40_cos40_norev_step5 | {"step-voxels": 5} | 0 | 5 | 609 | 89.031 | 2.792 | 2.026 | 0.649 | 0.000 | 7.337 |
| p40_cos40_norev_step5 | {"step-voxels": 5} | 512 | 5 | 240 | 81.403 | 11.754 | 4.702 | 3.596 | 1.468 | 11.728 |
| p40_cos40_norev_step8 | {"step-voxels": 8} | 0 | 3 | 580 | 89.032 | 2.981 | 2.128 | 0.694 | 0.000 | 7.662 |
| p40_cos40_norev_step8 | {"step-voxels": 8} | 512 | 3 | 239 | 81.963 | 12.349 | 4.754 | 3.596 | 1.500 | 11.771 |
| p40_cos40_norev_step6_normal005 | {"smoothness-normal-weight": 0.05, "step-voxels": 6} | 0 | 2 | 588 | 89.875 | 2.959 | 2.119 | 0.697 | 0.000 | 7.756 |
| p40_cos40_norev_step6_normal005 | {"smoothness-normal-weight": 0.05, "step-voxels": 6} | 512 | 2 | 235 | 82.024 | 12.226 | 4.838 | 3.626 | 1.460 | 11.883 |
| p60_cos40_norev_baseline | {} | 0 | 1 | 876 | 85.818 | 1.659 | 1.358 | 0.278 | 0.000 | 6.178 |
| p60_cos40_norev_baseline | {} | 512 | 1 | 258 | 75.793 | 8.144 | 4.072 | 2.791 | 1.481 | 10.457 |
| p60_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 0 | 3 | 856 | 86.163 | 1.711 | 1.395 | 0.284 | 0.000 | 6.188 |
| p60_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 512 | 2 | 257 | 76.444 | 8.410 | 4.123 | 3.135 | 1.517 | 10.456 |
| p60_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 0 | 4 | 933 | 85.660 | 1.532 | 1.273 | 0.228 | 0.000 | 5.774 |
| p60_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 512 | 3 | 260 | 75.365 | 7.855 | 4.018 | 2.661 | 1.450 | 10.455 |
| p60_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 0 | 1 | 879 | 85.745 | 1.651 | 1.352 | 0.260 | 0.000 | 6.033 |
| p60_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 512 | 0 | 262 | 76.155 | 8.058 | 4.029 | 2.816 | 1.486 | 10.203 |
| p60_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 0 | 1 | 884 | 85.766 | 1.642 | 1.345 | 0.254 | 0.000 | 5.997 |
| p60_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 512 | 1 | 261 | 75.919 | 8.158 | 4.032 | 2.686 | 1.467 | 10.452 |
| p60_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 0 | 3 | 861 | 85.950 | 1.700 | 1.384 | 0.285 | 0.000 | 5.883 |
| p60_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 512 | 2 | 259 | 76.127 | 8.375 | 4.074 | 2.996 | 1.499 | 10.194 |
| p60_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 0 | 1 | 912 | 85.472 | 1.571 | 1.299 | 0.233 | 0.000 | 5.833 |
| p60_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 512 | 1 | 256 | 75.229 | 8.211 | 4.073 | 3.046 | 1.532 | 10.258 |
| p60_cos40_norev_step2 | {"step-voxels": 2} | 0 | 2 | 1536 | 85.566 | 0.861 | 0.772 | 0.249 | 0.000 | 3.355 |
| p60_cos40_norev_step2 | {"step-voxels": 2} | 512 | 1 | 297 | 61.894 | 4.047 | 2.889 | 2.145 | 1.291 | 6.557 |
| p60_cos40_norev_step6 | {"step-voxels": 6} | 0 | 1 | 840 | 85.144 | 1.733 | 1.405 | 0.288 | 0.000 | 6.081 |
| p60_cos40_norev_step6 | {"step-voxels": 6} | 512 | 0 | 261 | 75.957 | 8.037 | 4.034 | 2.679 | 1.452 | 10.159 |

## PHerc0139

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p40_cos40_norev_baseline | {} | 0 | 0 | 284 | 92.280 | 1.070 | 0.927 | 0.326 | 0.002 | 3.351 |
| p40_cos40_norev_baseline | {} | 512 | 0 | 63 | 67.354 | 5.489 | 3.049 | 2.023 | 1.347 | 8.585 |
| p40_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 0 | 1 | 276 | 92.271 | 1.106 | 0.954 | 0.335 | 0.000 | 3.361 |
| p40_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 512 | 1 | 63 | 68.342 | 5.569 | 3.094 | 2.071 | 1.328 | 8.585 |
| p40_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 0 | 2 | 302 | 92.070 | 0.995 | 0.870 | 0.269 | 0.000 | 3.319 |
| p40_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 512 | 2 | 63 | 67.955 | 5.538 | 3.077 | 2.072 | 1.362 | 8.585 |
| p40_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 0 | 1 | 280 | 92.320 | 1.088 | 0.940 | 0.336 | 0.000 | 3.352 |
| p40_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 512 | 1 | 63 | 68.324 | 5.568 | 3.093 | 2.069 | 1.372 | 8.584 |
| p40_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 0 | 1 | 283 | 92.360 | 1.080 | 0.931 | 0.328 | 0.000 | 3.351 |
| p40_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 512 | 1 | 65 | 68.653 | 5.292 | 3.012 | 2.022 | 1.311 | 8.287 |
| p40_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 0 | 1 | 269 | 92.340 | 1.140 | 0.979 | 0.342 | 0.017 | 3.367 |
| p40_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 512 | 1 | 63 | 68.649 | 5.439 | 3.108 | 2.072 | 1.348 | 8.585 |
| p40_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 0 | 1 | 286 | 92.180 | 1.060 | 0.919 | 0.307 | 0.000 | 3.351 |
| p40_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 512 | 1 | 60 | 66.634 | 5.939 | 3.168 | 2.079 | 1.351 | 8.883 |
| p40_cos40_norev_step2 | {"step-voxels": 2} | 0 | 1 | 377 | 93.010 | 0.783 | 0.704 | 0.241 | 0.000 | 2.812 |
| p40_cos40_norev_step2 | {"step-voxels": 2} | 512 | 1 | 62 | 62.868 | 4.846 | 2.892 | 2.215 | 1.364 | 5.893 |
| p40_cos40_norev_step6 | {"step-voxels": 6} | 0 | 0 | 273 | 91.032 | 1.105 | 0.951 | 0.350 | 0.000 | 3.358 |
| p40_cos40_norev_step6 | {"step-voxels": 6} | 512 | 0 | 60 | 66.395 | 5.739 | 3.156 | 2.066 | 1.386 | 8.883 |
| p40_cos40_norev_step5 | {"step-voxels": 5} | 0 | 1 | 259 | 91.866 | 1.186 | 1.012 | 0.408 | 0.000 | 3.385 |
| p40_cos40_norev_step5 | {"step-voxels": 5} | 512 | 1 | 67 | 70.225 | 5.136 | 2.989 | 2.028 | 1.321 | 7.988 |
| p40_cos40_norev_step8 | {"step-voxels": 8} | 0 | 1 | 255 | 90.191 | 1.180 | 1.009 | 0.387 | 0.000 | 3.411 |
| p40_cos40_norev_step8 | {"step-voxels": 8} | 512 | 1 | 65 | 69.507 | 5.358 | 3.050 | 1.987 | 1.321 | 8.284 |
| p40_cos40_norev_step6_normal005 | {"smoothness-normal-weight": 0.05, "step-voxels": 6} | 0 | 1 | 270 | 91.053 | 1.119 | 0.962 | 0.377 | 0.000 | 3.361 |
| p40_cos40_norev_step6_normal005 | {"smoothness-normal-weight": 0.05, "step-voxels": 6} | 512 | 1 | 61 | 66.959 | 5.617 | 3.131 | 2.071 | 1.363 | 8.883 |
| p60_cos40_norev_baseline | {} | 0 | 0 | 417 | 86.765 | 0.648 | 0.593 | 0.136 | 0.000 | 2.484 |
| p60_cos40_norev_baseline | {} | 512 | 0 | 59 | 61.268 | 5.295 | 2.962 | 2.054 | 1.350 | 8.883 |
| p60_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 0 | 0 | 405 | 86.979 | 0.670 | 0.613 | 0.152 | 0.000 | 2.553 |
| p60_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 512 | 0 | 57 | 62.000 | 5.704 | 3.102 | 2.054 | 1.384 | 8.883 |
| p60_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 0 | 0 | 453 | 86.318 | 0.589 | 0.543 | 0.114 | 0.000 | 2.460 |
| p60_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 512 | 0 | 58 | 60.349 | 5.216 | 2.968 | 2.088 | 1.405 | 8.883 |
| p60_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 0 | 0 | 424 | 86.568 | 0.636 | 0.582 | 0.131 | 0.000 | 2.561 |
| p60_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 512 | 0 | 60 | 62.208 | 5.219 | 2.957 | 2.084 | 1.350 | 8.883 |
| p60_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 0 | 0 | 426 | 86.722 | 0.634 | 0.581 | 0.117 | 0.000 | 2.460 |
| p60_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 512 | 0 | 58 | 61.717 | 5.501 | 3.035 | 2.046 | 1.326 | 8.883 |
| p60_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 0 | 0 | 412 | 86.985 | 0.660 | 0.602 | 0.135 | 0.000 | 2.619 |
| p60_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 512 | 0 | 59 | 62.605 | 5.411 | 3.026 | 2.052 | 1.361 | 8.883 |
| p60_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 0 | 0 | 441 | 86.448 | 0.607 | 0.559 | 0.118 | 0.000 | 2.460 |
| p60_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 512 | 0 | 59 | 61.612 | 5.325 | 2.979 | 2.057 | 1.347 | 8.883 |
| p60_cos40_norev_step2 | {"step-voxels": 2} | 0 | 0 | 616 | 87.695 | 0.431 | 0.406 | 0.114 | 0.000 | 1.812 |
| p60_cos40_norev_step2 | {"step-voxels": 2} | 512 | 0 | 55 | 48.447 | 3.636 | 2.512 | 2.028 | 1.310 | 5.186 |
| p60_cos40_norev_step6 | {"step-voxels": 6} | 0 | 0 | 387 | 85.102 | 0.692 | 0.627 | 0.145 | 0.000 | 2.582 |
| p60_cos40_norev_step6 | {"step-voxels": 6} | 512 | 0 | 59 | 62.657 | 5.416 | 3.029 | 2.032 | 1.358 | 8.883 |

## PHerc0332

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p40_cos40_norev_baseline | {} | 0 | 0 | 7 | 33.296 | 0.529 | 0.529 | 0.694 | 0.023 | 1.069 |
| p40_cos40_norev_baseline | {} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p40_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 0 | 0 | 7 | 33.305 | 0.529 | 0.529 | 0.695 | 0.023 | 1.069 |
| p40_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p40_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 0 | 0 | 8 | 33.095 | 0.460 | 0.460 | 0.383 | 0.000 | 1.066 |
| p40_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p40_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 0 | 0 | 7 | 33.295 | 0.529 | 0.529 | 0.694 | 0.023 | 1.070 |
| p40_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p40_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 0 | 0 | 7 | 33.307 | 0.530 | 0.530 | 0.694 | 0.023 | 1.070 |
| p40_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p40_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 0 | 0 | 8 | 32.949 | 0.458 | 0.458 | 0.383 | 0.000 | 1.068 |
| p40_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p40_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 0 | 0 | 7 | 33.125 | 0.527 | 0.527 | 0.694 | 0.023 | 1.058 |
| p40_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p40_cos40_norev_step2 | {"step-voxels": 2} | 0 | 0 | 13 | 33.063 | 0.283 | 0.283 | 0.084 | 0.000 | 0.829 |
| p40_cos40_norev_step2 | {"step-voxels": 2} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p40_cos40_norev_step6 | {"step-voxels": 6} | 0 | 0 | 8 | 31.915 | 0.444 | 0.444 | 0.333 | 0.021 | 1.038 |
| p40_cos40_norev_step6 | {"step-voxels": 6} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p40_cos40_norev_step5 | {"step-voxels": 5} | 0 | 0 | 7 | 32.993 | 0.525 | 0.525 | 0.675 | 0.021 | 1.056 |
| p40_cos40_norev_step5 | {"step-voxels": 5} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p40_cos40_norev_step8 | {"step-voxels": 8} | 0 | 0 | 8 | 32.085 | 0.446 | 0.446 | 0.346 | 0.000 | 1.063 |
| p40_cos40_norev_step8 | {"step-voxels": 8} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p40_cos40_norev_step6_normal005 | {"smoothness-normal-weight": 0.05, "step-voxels": 6} | 0 | 0 | 7 | 32.444 | 0.516 | 0.516 | 0.666 | 0.026 | 1.037 |
| p40_cos40_norev_step6_normal005 | {"smoothness-normal-weight": 0.05, "step-voxels": 6} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p60_cos40_norev_baseline | {} | 0 | 0 | 9 | 29.300 | 0.362 | 0.362 | 0.054 | 0.000 | 0.996 |
| p60_cos40_norev_baseline | {} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p60_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 0 | 0 | 10 | 28.785 | 0.320 | 0.320 | 0.037 | 0.000 | 0.992 |
| p60_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p60_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 0 | 0 | 9 | 29.315 | 0.362 | 0.362 | 0.053 | 0.000 | 0.990 |
| p60_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p60_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 0 | 0 | 10 | 28.792 | 0.320 | 0.320 | 0.037 | 0.000 | 0.993 |
| p60_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p60_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 0 | 0 | 9 | 29.317 | 0.363 | 0.363 | 0.051 | 0.000 | 0.997 |
| p60_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p60_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 0 | 0 | 10 | 28.787 | 0.320 | 0.320 | 0.037 | 0.000 | 0.992 |
| p60_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p60_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 0 | 0 | 10 | 28.463 | 0.317 | 0.317 | 0.037 | 0.000 | 0.966 |
| p60_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p60_cos40_norev_step2 | {"step-voxels": 2} | 0 | 0 | 14 | 29.137 | 0.232 | 0.232 | 0.029 | 0.000 | 0.956 |
| p60_cos40_norev_step2 | {"step-voxels": 2} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |
| p60_cos40_norev_step6 | {"step-voxels": 6} | 0 | 0 | 9 | 27.774 | 0.343 | 0.343 | 0.032 | 0.000 | 0.925 |
| p60_cos40_norev_step6 | {"step-voxels": 6} | 512 | 0 | 0 | 0.000 | — | — | — | — | — |

## PHerc1299

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p40_cos40_norev_baseline | {} | 0 | 1 | 26 | 99.794 | 29.425 | 5.659 | 4.811 | 0.483 | 15.539 |
| p40_cos40_norev_baseline | {} | 512 | 1 | 23 | 99.076 | 36.516 | 6.351 | 5.033 | 3.517 | 16.793 |
| p40_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 0 | 1 | 26 | 99.793 | 29.424 | 5.659 | 4.811 | 0.541 | 15.539 |
| p40_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 512 | 1 | 23 | 99.030 | 36.499 | 6.348 | 5.033 | 3.510 | 16.793 |
| p40_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 0 | 1 | 27 | 99.768 | 24.514 | 5.448 | 4.589 | 0.428 | 15.120 |
| p40_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 512 | 1 | 23 | 98.741 | 36.393 | 6.329 | 5.033 | 3.512 | 16.793 |
| p40_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 0 | 0 | 25 | 99.793 | 36.780 | 5.885 | 5.033 | 0.938 | 15.957 |
| p40_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 512 | 0 | 23 | 99.383 | 48.839 | 6.370 | 5.033 | 3.516 | 16.793 |
| p40_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 0 | 1 | 26 | 99.792 | 29.424 | 5.658 | 4.811 | 0.964 | 15.539 |
| p40_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 512 | 1 | 24 | 99.380 | 36.628 | 6.105 | 5.033 | 2.094 | 16.375 |
| p40_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 0 | 1 | 26 | 99.793 | 29.424 | 5.659 | 4.811 | 0.502 | 15.539 |
| p40_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 512 | 1 | 23 | 99.065 | 36.512 | 6.350 | 5.033 | 3.517 | 16.793 |
| p40_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 0 | 2 | 28 | 99.766 | 21.012 | 5.253 | 4.589 | 0.871 | 12.951 |
| p40_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 512 | 2 | 26 | 99.356 | 24.413 | 5.634 | 4.811 | 2.040 | 13.518 |
| p40_cos40_norev_step2 | {"step-voxels": 2} | 0 | 4 | 34 | 99.750 | 10.504 | 4.325 | 4.283 | 0.593 | 9.929 |
| p40_cos40_norev_step2 | {"step-voxels": 2} | 512 | 4 | 27 | 96.732 | 14.261 | 5.282 | 4.432 | 2.158 | 11.238 |
| p40_cos40_norev_step6 | {"step-voxels": 6} | 0 | 0 | 24 | 99.807 | 49.048 | 6.131 | 5.033 | 2.434 | 16.375 |
| p40_cos40_norev_step6 | {"step-voxels": 6} | 512 | 0 | 23 | 99.807 | 73.572 | 6.398 | 5.033 | 3.516 | 16.793 |
| p40_cos40_norev_step5 | {"step-voxels": 5} | 0 | 2 | 27 | 99.775 | 24.516 | 5.448 | 4.589 | 0.481 | 15.120 |
| p40_cos40_norev_step5 | {"step-voxels": 5} | 512 | 2 | 24 | 99.075 | 29.213 | 6.086 | 5.033 | 2.104 | 16.375 |
| p40_cos40_norev_step8 | {"step-voxels": 8} | 0 | 0 | 23 | 99.846 | 73.600 | 6.400 | 5.481 | 4.112 | 16.793 |
| p40_cos40_norev_step8 | {"step-voxels": 8} | 512 | 0 | 22 | 99.846 | 147.200 | 6.691 | 5.481 | 4.240 | 17.211 |
| p40_cos40_norev_step6_normal005 | {"smoothness-normal-weight": 0.05, "step-voxels": 6} | 0 | 0 | 25 | 99.770 | 36.772 | 5.884 | 5.033 | 0.932 | 15.957 |
| p40_cos40_norev_step6_normal005 | {"smoothness-normal-weight": 0.05, "step-voxels": 6} | 512 | 0 | 23 | 99.362 | 48.829 | 6.369 | 5.033 | 3.516 | 16.793 |
| p60_cos40_norev_baseline | {} | 0 | 1 | 36 | 99.482 | 9.778 | 4.074 | 4.283 | 0.144 | 8.985 |
| p60_cos40_norev_baseline | {} | 512 | 1 | 26 | 96.242 | 15.765 | 5.457 | 4.773 | 2.366 | 9.789 |
| p60_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 0 | 2 | 36 | 99.520 | 9.781 | 4.076 | 4.283 | 0.144 | 8.403 |
| p60_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 512 | 2 | 27 | 96.879 | 15.870 | 5.290 | 4.796 | 2.373 | 8.511 |
| p60_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 0 | 2 | 37 | 99.494 | 9.168 | 3.964 | 4.240 | 0.139 | 8.613 |
| p60_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 512 | 2 | 27 | 96.637 | 14.247 | 5.277 | 4.790 | 2.318 | 8.691 |
| p60_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 0 | 0 | 35 | 99.506 | 10.479 | 4.191 | 4.372 | 0.134 | 9.069 |
| p60_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 512 | 0 | 26 | 96.964 | 20.422 | 5.498 | 5.033 | 2.361 | 9.801 |
| p60_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 0 | 1 | 38 | 99.457 | 8.625 | 3.859 | 4.283 | 0.163 | 8.619 |
| p60_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 512 | 1 | 28 | 96.286 | 14.195 | 5.070 | 4.787 | 2.044 | 8.683 |
| p60_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 0 | 1 | 35 | 99.519 | 10.480 | 4.192 | 4.326 | 0.134 | 8.635 |
| p60_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 512 | 1 | 27 | 97.104 | 17.895 | 5.302 | 4.784 | 2.252 | 8.706 |
| p60_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 0 | 0 | 35 | 99.483 | 10.476 | 4.190 | 4.372 | 0.134 | 9.065 |
| p60_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 512 | 0 | 26 | 96.796 | 20.386 | 5.489 | 4.918 | 2.361 | 9.790 |
| p60_cos40_norev_step2 | {"step-voxels": 2} | 0 | 1 | 89 | 98.981 | 2.146 | 1.640 | 0.957 | 0.096 | 5.302 |
| p60_cos40_norev_step2 | {"step-voxels": 2} | 512 | 1 | 39 | 82.424 | 4.340 | 3.116 | 2.728 | 1.255 | 6.109 |
| p60_cos40_norev_step6 | {"step-voxels": 6} | 0 | 0 | 33 | 99.458 | 12.219 | 4.443 | 4.563 | 0.105 | 9.212 |
| p60_cos40_norev_step6 | {"step-voxels": 6} | 512 | 0 | 26 | 97.340 | 20.501 | 5.519 | 4.896 | 2.356 | 9.764 |

## PHerc1451

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p40_cos40_norev_baseline | {} | 0 | 0 | 7 | 95.082 | 1.663 | 1.426 | 1.115 | 0.184 | 3.241 |
| p40_cos40_norev_baseline | {} | 512 | 0 | 2 | 61.713 | 2.159 | 3.239 | 3.239 | 3.233 | 3.244 |
| p40_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 0 | 0 | 5 | 95.791 | 2.513 | 2.011 | 1.289 | 0.609 | 4.175 |
| p40_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 512 | 0 | 3 | 84.156 | 2.944 | 2.944 | 3.100 | 1.470 | 4.309 |
| p40_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 0 | 0 | 6 | 95.428 | 2.003 | 1.669 | 0.864 | 0.153 | 4.141 |
| p40_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 512 | 0 | 2 | 73.126 | 3.838 | 3.838 | 3.838 | 3.292 | 4.383 |
| p40_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 0 | 0 | 6 | 95.442 | 2.003 | 1.670 | 1.224 | 0.612 | 3.210 |
| p40_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 512 | 0 | 3 | 72.740 | 1.909 | 2.545 | 3.099 | 1.469 | 3.232 |
| p40_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 0 | 0 | 6 | 95.253 | 1.999 | 1.666 | 1.219 | 0.598 | 3.215 |
| p40_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 512 | 0 | 3 | 72.796 | 1.910 | 2.547 | 3.099 | 1.469 | 3.239 |
| p40_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 0 | 0 | 7 | 95.062 | 1.663 | 1.425 | 1.114 | 0.184 | 3.245 |
| p40_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 512 | 0 | 2 | 61.768 | 2.161 | 3.242 | 3.242 | 3.233 | 3.250 |
| p40_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 0 | 0 | 6 | 95.432 | 2.003 | 1.669 | 1.203 | 0.613 | 3.231 |
| p40_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 512 | 0 | 3 | 72.911 | 1.913 | 2.551 | 3.113 | 1.454 | 3.254 |
| p40_cos40_norev_step2 | {"step-voxels": 2} | 0 | 0 | 9 | 95.496 | 1.253 | 1.114 | 0.614 | 0.012 | 3.240 |
| p40_cos40_norev_step2 | {"step-voxels": 2} | 512 | 0 | 2 | 61.704 | 2.159 | 3.238 | 3.238 | 3.229 | 3.248 |
| p40_cos40_norev_step6 | {"step-voxels": 6} | 0 | 0 | 5 | 95.320 | 2.501 | 2.001 | 1.271 | 0.590 | 4.175 |
| p40_cos40_norev_step6 | {"step-voxels": 6} | 512 | 0 | 3 | 83.980 | 2.938 | 2.938 | 3.099 | 1.454 | 4.310 |
| p40_cos40_norev_step5 | {"step-voxels": 5} | 0 | 0 | 7 | 94.779 | 1.658 | 1.421 | 1.106 | 0.262 | 3.011 |
| p40_cos40_norev_step5 | {"step-voxels": 5} | 512 | 0 | 3 | 71.657 | 2.507 | 2.507 | 2.396 | 1.905 | 3.187 |
| p40_cos40_norev_step8 | {"step-voxels": 8} | 0 | 0 | 7 | 93.454 | 1.635 | 1.401 | 1.116 | 0.582 | 2.507 |
| p40_cos40_norev_step8 | {"step-voxels": 8} | 512 | 0 | 3 | 65.020 | 2.275 | 2.275 | 2.454 | 1.902 | 2.522 |
| p40_cos40_norev_step6_normal005 | {"smoothness-normal-weight": 0.05, "step-voxels": 6} | 0 | 0 | 6 | 94.778 | 1.990 | 1.658 | 1.208 | 0.593 | 3.206 |
| p40_cos40_norev_step6_normal005 | {"smoothness-normal-weight": 0.05, "step-voxels": 6} | 512 | 0 | 3 | 72.509 | 1.903 | 2.537 | 3.097 | 1.454 | 3.228 |
| p60_cos40_norev_baseline | {} | 0 | 0 | 8 | 91.093 | 1.366 | 1.195 | 0.855 | 0.580 | 2.488 |
| p60_cos40_norev_baseline | {} | 512 | 0 | 2 | 47.407 | 1.659 | 2.488 | 2.488 | 2.486 | 2.489 |
| p60_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 0 | 0 | 8 | 91.066 | 1.365 | 1.195 | 0.836 | 0.201 | 2.908 |
| p60_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 512 | 0 | 2 | 53.549 | 1.873 | 2.810 | 2.810 | 2.518 | 3.103 |
| p60_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 0 | 0 | 8 | 91.079 | 1.366 | 1.195 | 0.855 | 0.580 | 2.488 |
| p60_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 512 | 0 | 2 | 47.404 | 1.658 | 2.488 | 2.488 | 2.487 | 2.489 |
| p60_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 0 | 0 | 9 | 90.906 | 1.193 | 1.060 | 0.612 | 0.230 | 2.487 |
| p60_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 512 | 0 | 2 | 47.390 | 1.658 | 2.487 | 2.487 | 2.486 | 2.488 |
| p60_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 0 | 0 | 9 | 90.706 | 1.190 | 1.058 | 0.612 | 0.230 | 2.487 |
| p60_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 512 | 0 | 2 | 47.382 | 1.658 | 2.487 | 2.487 | 2.487 | 2.487 |
| p60_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 0 | 0 | 9 | 90.890 | 1.192 | 1.060 | 0.614 | 0.230 | 2.487 |
| p60_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 512 | 0 | 2 | 47.390 | 1.658 | 2.487 | 2.487 | 2.486 | 2.488 |
| p60_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 0 | 0 | 8 | 91.241 | 1.368 | 1.197 | 0.834 | 0.201 | 2.909 |
| p60_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 512 | 0 | 2 | 53.572 | 1.874 | 2.811 | 2.811 | 2.518 | 3.105 |
| p60_cos40_norev_step2 | {"step-voxels": 2} | 0 | 0 | 15 | 90.804 | 0.681 | 0.635 | 0.620 | 0.000 | 1.237 |
| p60_cos40_norev_step2 | {"step-voxels": 2} | 512 | 0 | 1 | 14.409 | 0.756 | 1.512 | 1.512 | 1.512 | 1.512 |
| p60_cos40_norev_step6 | {"step-voxels": 6} | 0 | 0 | 9 | 89.647 | 1.176 | 1.045 | 0.605 | 0.237 | 2.451 |
| p60_cos40_norev_step6 | {"step-voxels": 6} | 512 | 0 | 2 | 46.595 | 1.630 | 2.445 | 2.445 | 2.421 | 2.470 |

## PHercParis4

| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p40_cos40_norev_baseline | {} | 0 | 2 | 310 | 88.146 | 3.988 | 2.650 | 0.977 | 0.000 | 9.203 |
| p40_cos40_norev_baseline | {} | 512 | 2 | 146 | 83.996 | 14.769 | 5.361 | 4.599 | 1.656 | 12.110 |
| p40_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 0 | 2 | 301 | 87.777 | 4.131 | 2.718 | 1.179 | 0.000 | 9.386 |
| p40_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 512 | 2 | 148 | 84.164 | 14.260 | 5.299 | 4.649 | 1.654 | 11.694 |
| p40_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 0 | 2 | 336 | 87.435 | 3.512 | 2.425 | 0.612 | 0.000 | 8.878 |
| p40_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 512 | 2 | 150 | 83.552 | 13.904 | 5.191 | 4.250 | 1.599 | 12.015 |
| p40_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 0 | 2 | 306 | 88.215 | 4.090 | 2.687 | 0.981 | 0.000 | 9.543 |
| p40_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 512 | 2 | 145 | 84.120 | 15.371 | 5.406 | 4.354 | 1.642 | 12.156 |
| p40_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 0 | 2 | 300 | 87.245 | 4.127 | 2.710 | 0.982 | 0.000 | 9.806 |
| p40_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 512 | 2 | 145 | 83.379 | 15.235 | 5.359 | 4.554 | 1.641 | 12.156 |
| p40_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 0 | 2 | 295 | 87.814 | 4.262 | 2.774 | 1.175 | 0.000 | 9.478 |
| p40_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 512 | 2 | 146 | 84.402 | 14.840 | 5.387 | 4.780 | 1.641 | 11.688 |
| p40_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 0 | 1 | 318 | 86.876 | 3.748 | 2.546 | 0.919 | 0.000 | 8.760 |
| p40_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 512 | 1 | 144 | 82.024 | 14.422 | 5.308 | 4.342 | 1.602 | 12.181 |
| p40_cos40_norev_step2 | {"step-voxels": 2} | 0 | 2 | 505 | 86.200 | 1.988 | 1.591 | 0.453 | 0.000 | 6.507 |
| p40_cos40_norev_step2 | {"step-voxels": 2} | 512 | 0 | 173 | 76.722 | 7.606 | 4.133 | 3.144 | 1.368 | 10.783 |
| p40_cos40_norev_step6 | {"step-voxels": 6} | 0 | 0 | 276 | 88.564 | 4.798 | 2.990 | 1.618 | 0.000 | 9.977 |
| p40_cos40_norev_step6 | {"step-voxels": 6} | 512 | 0 | 147 | 85.044 | 15.241 | 5.391 | 4.737 | 1.604 | 12.103 |
| p40_cos40_norev_step5 | {"step-voxels": 5} | 0 | 2 | 309 | 87.068 | 3.939 | 2.626 | 0.981 | 0.000 | 9.306 |
| p40_cos40_norev_step5 | {"step-voxels": 5} | 512 | 2 | 146 | 83.110 | 14.894 | 5.305 | 4.342 | 1.662 | 12.125 |
| p40_cos40_norev_step8 | {"step-voxels": 8} | 0 | 2 | 287 | 87.597 | 4.437 | 2.844 | 1.577 | 0.000 | 9.806 |
| p40_cos40_norev_step8 | {"step-voxels": 8} | 512 | 2 | 149 | 84.116 | 14.516 | 5.261 | 4.331 | 1.588 | 12.046 |
| p40_cos40_norev_step6_normal005 | {"smoothness-normal-weight": 0.05, "step-voxels": 6} | 0 | 1 | 280 | 88.579 | 4.664 | 2.948 | 1.597 | 0.000 | 9.841 |
| p40_cos40_norev_step6_normal005 | {"smoothness-normal-weight": 0.05, "step-voxels": 6} | 512 | 1 | 148 | 84.979 | 14.399 | 5.351 | 4.646 | 1.596 | 12.075 |
| p60_cos40_norev_baseline | {} | 0 | 0 | 406 | 83.982 | 2.541 | 1.928 | 0.591 | 0.000 | 7.327 |
| p60_cos40_norev_baseline | {} | 512 | 0 | 171 | 78.228 | 8.379 | 4.263 | 3.071 | 1.547 | 10.828 |
| p60_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 0 | 1 | 397 | 84.429 | 2.623 | 1.982 | 0.649 | 0.000 | 7.355 |
| p60_cos40_norev_tangent5 | {"smoothness-tangent-weight": 5} | 512 | 0 | 171 | 78.802 | 8.539 | 4.295 | 3.281 | 1.599 | 10.825 |
| p60_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 0 | 2 | 426 | 83.881 | 2.376 | 1.835 | 0.531 | 0.000 | 7.284 |
| p60_cos40_norev_tangent20 | {"smoothness-tangent-weight": 20} | 512 | 1 | 173 | 77.811 | 8.057 | 4.191 | 2.776 | 1.537 | 11.189 |
| p60_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 0 | 1 | 401 | 83.938 | 2.573 | 1.951 | 0.652 | 0.000 | 7.340 |
| p60_cos40_norev_normal005 | {"smoothness-normal-weight": 0.05} | 512 | 0 | 174 | 78.366 | 8.114 | 4.197 | 3.044 | 1.538 | 10.716 |
| p60_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 0 | 0 | 402 | 83.926 | 2.573 | 1.946 | 0.657 | 0.000 | 7.341 |
| p60_cos40_norev_normal02 | {"smoothness-normal-weight": 0.2} | 512 | 0 | 173 | 78.271 | 8.384 | 4.216 | 3.073 | 1.549 | 10.758 |
| p60_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 0 | 2 | 395 | 84.113 | 2.639 | 1.984 | 0.677 | 0.000 | 7.426 |
| p60_cos40_norev_cumulative0 | {"cumulative-smoothness-tangent-weight": 0} | 512 | 1 | 171 | 78.180 | 8.571 | 4.261 | 3.232 | 1.564 | 10.828 |
| p60_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 0 | 1 | 418 | 83.572 | 2.426 | 1.863 | 0.571 | 0.000 | 7.355 |
| p60_cos40_norev_cumulative4 | {"cumulative-smoothness-tangent-weight": 4} | 512 | 1 | 169 | 77.127 | 8.261 | 4.253 | 3.232 | 1.587 | 10.899 |
| p60_cos40_norev_step2 | {"step-voxels": 2} | 0 | 1 | 802 | 83.407 | 1.103 | 0.969 | 0.344 | 0.000 | 3.776 |
| p60_cos40_norev_step2 | {"step-voxels": 2} | 512 | 0 | 202 | 64.035 | 4.059 | 2.954 | 2.135 | 1.302 | 6.880 |
| p60_cos40_norev_step6 | {"step-voxels": 6} | 0 | 1 | 402 | 83.527 | 2.552 | 1.936 | 0.652 | 0.000 | 7.283 |
| p60_cos40_norev_step6 | {"step-voxels": 6} | 512 | 0 | 174 | 77.882 | 7.976 | 4.171 | 2.999 | 1.555 | 10.449 |

## Reproduce

Set control action `reviewed_trace_tuning`. Candidate plan: `scripts/reviewed_trace_tuning_plan.json`.

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Immutable workloads, native effective config, timing, outputs and per-piece decisions: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/reviewed_ema999_trace_tuning_20261006_v1`. JSON includes source/control paths, checkpoint and inspected native executable hash; the latter describes the binary present at analysis, not independent proof of historical cached-run provenance. Timings include startup and any on-demand cache fill, not isolated tracing speeds.
