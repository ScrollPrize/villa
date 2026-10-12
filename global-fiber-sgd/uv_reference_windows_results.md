# Shifted reference-winding windows

## Protocol

Reference tifxyz winding_bounds starts atID10,column_begin1, followed left-to-
right by11,12,etc. User-confirmed positions4–10,5–11,6–12 correspond to inclusive
IDs13–19,14–20,15–21. All use exact base-Z clipping to30000–50000 and existing
whole-source median reference labels, preserving source IDs and surviving
constraint evidence. Earlier sign/HV/direction stages are reused, not rerun.

Actual XYZ TIFF validation (not metadata-only):grid956×56913,47026606valid
finite/non-(-1)vertices; validcolumns1–56911,validrows1–954. Zero validvertices
left ofID10's column_begin1; zero right oflastID135's column_end56912; zero
validvertices outside the union of all126declared winding rectangles. Thus no
undescribed surface strip precedesID10. Read-only full-array check took0.58s.

Each subset is solved fresh and reference-assignment/width initialized. Same
151hypotheses,full-parent radial prior,CCW,compiled squared pair objective,
combined coordinate kernel,all variables trainable immediately,seed0.
Six exclusive GPUs,global batch24576,stateLR0.007348469228,widthLR2.
No changed losses,filtering,or added reference targets. The reference selects
the evaluation dataset and initializes only the reference-started run.

100iterations means sampled dataset-pass equivalents, not100Adamupdates.
Integer budgets and diagnostic intervals:

| Positions | IDs | Fibers | Pieces | Constraints | Links | Updates | Actual passes | Log every updates |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4–10 | 13–19 | 2018 | 2026 | 22186 | 0 | 91 | 100.8030 | 9 |
| 5–11 | 14–20 | 2192 | 2199 | 24771 | 1 | 101 | 100.2049 | 10 |
| 6–12 | 15–21 | 2262 | 2270 | 26429 | 2 | 108 | 100.4279 | 11 |

Extraction15.397/15.242/15.121s respectively (concurrent CPU processes,not summed
walltime). All automatic audits passed: finite arrays,geometryZ/reference range,
endpoint/source ownership,induced degrees,continuations and projection identity.

Reference metrics: equal-source retained-curve median winding, globally offset-
corrected L1/within±2. Widthfactor uses only fully-supported intervals in each
selected reference range, after applying the same offset. Retained populations
may change with rejection. Loss means online objective across preceding update
interval; final short intervals have different sample counts. These overlapping
windows differ in constraints and optimizer update budgets; lower loss alone
does not locate kollesis. No convergence claim based on100passes alone.

## Commands

Use the existing approved runner; no separate entrypoints:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/extract_uv_reference_w13_19.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/extract_uv_reference_w14_20.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/extract_uv_reference_w15_21.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_windows_suite.json
```

Extraction outputs:data/uv_reference_w{13_19,14_20,15_21}_z30k50k_v1.
Solve outputs:data/uv_windows_w{13_19,14_20,15_21}_{fresh,reference}_v1.
Child configs:run_uv_windows_w*_{fresh,reference}.json. Completed solves are
protected from overwrite; a fresh rerun needs new output roots.

## Completed six-run results

All six completed successfully; all final state/width arrays finite. Retained
length100% and all selected sources reference-matched at every checkpoint.

| Positions | Init | Loss | Ref L1 | Within±2 % | Sign violations % | Regional width factor | Solve seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| 4–10 | fresh | 51.8623 | 2.6239 | 55.5996 | 23.1155 | 0.9865 | 2.968 |
| 4–10 | reference | 37.6902 | 0.4876 | 100.0000 | 4.6955 | 1.0092 | 2.906 |
| 5–11 | fresh | 50.9035 | 2.7427 | 53.3759 | 22.3843 | 1.0010 | 3.382 |
| 5–11 | reference | 35.2951 | 0.4831 | 100.0000 | 4.5390 | 1.0055 | 3.309 |
| 6–12 | fresh | 50.5939 | 2.9010 | 50.2210 | 22.3935 | 0.9486 | 3.618 |
| 6–12 | reference | 34.5186 | 0.4881 | 100.0000 | 4.8024 | 1.0032 | 3.546 |

Previous positions1–10 baseline (IDs10–19):fresh loss43.2843,L1=2.3063,
within2=62.1652%,sign22.5004%,widthfactor.9072;reference loss32.3750,L1=.4885,
within2=100%,sign9.8259%,widthfactor1.0083. Previous123updates vs current91/101/108.

Fresh improves L1 in every window but remains far from reference. Reference
runs settle nearL1=.49 and100%within2, with fewer final sign violations4.5–4.8%
than previous9.8%. Initial reference sign rates also decrease from15.7495%
(previous) to11.6591%,10.9807%,10.2723%. This is evidence that the selected graphs
have fewer reference-order conflicts, not proof that a kollesis is the cause.
Loss keeps falling;100pass runs do not show convergence. Reference-initialized
MAP errors grow while objective falls. Width comparisons depend on globaloffset;
fresh13–19 changes0→-1,14–20 remains-1,15–21 remains0; allreference runs offset0.

## Timing

Single trial per initialization, exclusive6×H100,global24576,first5updates
excluded from CUDA per-update mean/p50/p95. CUDA timing is max acrossranks.
Solve wall includes optimization/diagnostics/save but excludes worker loading,
preflight and managerwait. Worker elapsed is config-to-summary timestamps.

| Positions | Init | Solve s | Update mean ms | p50 ms | p95 ms | Peak GiB/GPU | Worker elapsed s |
|---|---|---:|---:|---:|---:|---:|---:|
| 4–10 | fresh | 2.968 | 14.347 | 12.679 | 18.276 | 1.156 | 13.724 |
| 4–10 | reference | 2.906 | 14.576 | 12.646 | 23.644 | 1.156 | 14.054 |
| 5–11 | fresh | 3.382 | 16.979 | 15.562 | 20.938 | 1.160 | 14.088 |
| 5–11 | reference | 3.309 | 16.844 | 15.847 | 22.409 | 1.160 | 14.441 |
| 6–12 | fresh | 3.618 | 16.680 | 16.047 | 19.947 | 1.160 | 14.421 |
| 6–12 | reference | 3.546 | 17.856 | 15.994 | 23.498 | 1.160 | 14.664 |

Whole suite **119.889s**, including managerwait/startups/preflights.
Extraction is separate. SixGPU gradient equivalence and combined coordinate
value/gradient/bitwise-repeat preflight passed in everyrun.20targeted CPU
subset/reference tests pass thisfollowup;50targeted UVtests passed before matrix.
No algorithm/code changes for this shifted-window matrix.

## Positions 4–10, IDs 13–19: fresh

| Passes | Online loss | Ref L1 | Within±2 % | Retained % | Sign violations % | Widthfactor | Globaloffset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 3.0059 | 47.820 | 100.000 | 37.327 | 0.8940 | 0 | 12/23 | 14/19 |
| 9.97 | 105.2110 | 2.9078 | 49.802 | 100.000 | 28.781 | 0.8962 | 0 | 12/23 | 14/19 |
| 19.94 | 94.5224 | 2.8826 | 50.644 | 100.000 | 27.637 | 0.8983 | 0 | 12/23 | 14/19 |
| 29.91 | 85.3012 | 2.8954 | 49.158 | 100.000 | 23.159 | 0.9741 | -1 | 11/22 | 14/19 |
| 39.88 | 78.7882 | 2.8741 | 49.455 | 100.000 | 21.629 | 0.9757 | -1 | 11/22 | 14/19 |
| 49.85 | 73.0910 | 2.8528 | 49.950 | 100.000 | 20.876 | 0.9773 | -1 | 11/21 | 14/19 |
| 59.82 | 68.0976 | 2.8122 | 50.942 | 100.000 | 21.663 | 0.9790 | -1 | 11/21 | 14/19 |
| 69.79 | 64.0226 | 2.7656 | 51.833 | 100.000 | 22.165 | 0.9808 | -1 | 11/21 | 14/19 |
| 79.76 | 60.1187 | 2.7081 | 53.915 | 100.000 | 21.919 | 0.9107 | 0 | 12/22 | 14/19 |
| 89.73 | 56.6054 | 2.6650 | 54.609 | 100.000 | 22.464 | 0.9128 | 0 | 12/22 | 14/19 |
| 99.70 | 53.4871 | 2.6249 | 55.600 | 100.000 | 23.038 | 0.9864 | -1 | 11/21 | 14/19 |
| 100.80 | 51.8623 | 2.6239 | 55.600 | 100.000 | 23.116 | 0.9865 | -1 | 11/21 | 14/19 |

## Positions 4–10, IDs 13–19: reference

| Passes | Online loss | Ref L1 | Within±2 % | Retained % | Sign violations % | Widthfactor | Globaloffset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 0.0010 | 100.000 | 100.000 | 11.659 | 1.0000 | 0 | 14/19 | 14/19 |
| 9.97 | 79.6007 | 0.1378 | 100.000 | 100.000 | 5.390 | 1.0011 | 0 | 14/19 | 14/19 |
| 19.94 | 71.0189 | 0.2711 | 100.000 | 100.000 | 5.352 | 1.0024 | 0 | 13/19 | 14/19 |
| 29.91 | 63.9242 | 0.3508 | 100.000 | 100.000 | 5.053 | 1.0035 | 0 | 13/19 | 14/19 |
| 39.88 | 58.7104 | 0.3657 | 100.000 | 100.000 | 4.474 | 1.0044 | 0 | 13.700000000000017/19 | 14/19 |
| 49.85 | 54.0277 | 0.3925 | 100.000 | 100.000 | 4.005 | 1.0053 | 0 | 13/19 | 14/19 |
| 59.82 | 50.0704 | 0.4217 | 100.000 | 100.000 | 3.827 | 1.0061 | 0 | 13/19 | 14/19 |
| 69.79 | 46.8759 | 0.4554 | 100.000 | 100.000 | 3.624 | 1.0069 | 0 | 13/19 | 14/19 |
| 79.76 | 43.9563 | 0.4742 | 100.000 | 100.000 | 3.750 | 1.0076 | 0 | 13/19 | 14/19 |
| 89.73 | 41.3497 | 0.4871 | 100.000 | 100.000 | 4.300 | 1.0084 | 0 | 13/19 | 14/19 |
| 99.70 | 39.0184 | 0.4911 | 100.000 | 100.000 | 4.604 | 1.0091 | 0 | 13/19 | 14/19 |
| 100.80 | 37.6902 | 0.4876 | 100.000 | 100.000 | 4.695 | 1.0092 | 0 | 13/19 | 14/19 |

## Positions 5–11, IDs 14–20: fresh

| Passes | Online loss | Ref L1 | Within±2 % | Retained % | Sign violations % | Widthfactor | Globaloffset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 3.1957 | 45.119 | 100.000 | 37.662 | 0.9829 | -1 | 11/23 | 15/20 |
| 9.92 | 107.0744 | 3.0839 | 46.943 | 100.000 | 28.593 | 0.9147 | 0 | 12/23 | 15/20 |
| 19.84 | 95.1548 | 3.0652 | 46.989 | 100.000 | 27.523 | 0.9862 | -1 | 11/22.90000000000009 | 15/20 |
| 29.76 | 85.8127 | 3.0625 | 46.989 | 100.000 | 23.182 | 0.9877 | -1 | 11/23 | 15/20 |
| 39.69 | 78.8869 | 3.0301 | 47.400 | 100.000 | 22.056 | 0.9893 | -1 | 11/22 | 15/20 |
| 49.61 | 72.9998 | 2.9936 | 47.993 | 100.000 | 21.517 | 0.9910 | -1 | 12/22 | 15/20 |
| 59.53 | 67.6413 | 2.9402 | 49.270 | 100.000 | 21.228 | 0.9251 | 0 | 13/23 | 15/20 |
| 69.45 | 63.1894 | 2.8837 | 50.593 | 100.000 | 21.694 | 0.9273 | 0 | 13/23 | 15/20 |
| 79.37 | 59.5153 | 2.8248 | 51.551 | 100.000 | 21.741 | 0.9294 | 0 | 13/23 | 15/20 |
| 89.29 | 55.8146 | 2.7851 | 52.646 | 100.000 | 21.586 | 0.9986 | -1 | 12/22 | 15/20 |
| 99.21 | 52.4122 | 2.7477 | 53.193 | 100.000 | 22.324 | 1.0007 | -1 | 12/22 | 15/20 |
| 100.20 | 50.9035 | 2.7427 | 53.376 | 100.000 | 22.384 | 1.0010 | -1 | 12/22 | 15/20 |

## Positions 5–11, IDs 14–20: reference

| Passes | Online loss | Ref L1 | Within±2 % | Retained % | Sign violations % | Widthfactor | Globaloffset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 0.0009 | 100.000 | 100.000 | 10.981 | 1.0000 | 0 | 15/20 | 15/20 |
| 9.92 | 78.8068 | 0.1556 | 100.000 | 100.000 | 5.497 | 1.0011 | 0 | 14/20 | 15/20 |
| 19.84 | 69.5630 | 0.2988 | 100.000 | 100.000 | 5.238 | 1.0022 | 0 | 14/20 | 15/20 |
| 29.76 | 62.2675 | 0.3554 | 100.000 | 100.000 | 4.871 | 1.0030 | 0 | 14/20 | 15/20 |
| 39.69 | 56.6558 | 0.3695 | 100.000 | 100.000 | 4.241 | 1.0035 | 0 | 14/20 | 15/20 |
| 49.61 | 51.8284 | 0.4028 | 100.000 | 100.000 | 3.853 | 1.0039 | 0 | 14/20 | 15/20 |
| 59.53 | 47.8898 | 0.4402 | 100.000 | 100.000 | 3.762 | 1.0043 | 0 | 14/20 | 15/20 |
| 69.45 | 44.6565 | 0.4681 | 100.000 | 100.000 | 3.762 | 1.0046 | 0 | 14/20 | 15/20 |
| 79.37 | 41.6667 | 0.4799 | 100.000 | 100.000 | 3.767 | 1.0049 | 0 | 14/20 | 15/20 |
| 89.29 | 39.0313 | 0.4831 | 100.000 | 100.000 | 3.594 | 1.0052 | 0 | 14/20 | 15/20 |
| 99.21 | 36.5668 | 0.4813 | 100.000 | 100.000 | 4.401 | 1.0055 | 0 | 14/20 | 15/20 |
| 100.20 | 35.2951 | 0.4831 | 100.000 | 100.000 | 4.539 | 1.0055 | 0 | 14/20 | 15/20 |

## Positions 6–12, IDs 15–21: fresh

| Passes | Online loss | Ref L1 | Within±2 % | Retained % | Sign violations % | Widthfactor | Globaloffset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 3.3599 | 43.148 | 100.000 | 37.290 | 0.9288 | 0 | 13/25 | 15/21 |
| 10.23 | 107.9895 | 3.2480 | 45.535 | 100.000 | 28.652 | 0.9306 | 0 | 13/24 | 15/21 |
| 20.46 | 95.1172 | 3.2325 | 44.651 | 100.000 | 27.734 | 0.9977 | -1 | 12/23 | 15/21 |
| 30.69 | 85.2392 | 3.2268 | 45.358 | 100.000 | 23.890 | 0.9340 | 0 | 13/24 | 15/21 |
| 40.92 | 78.1467 | 3.1897 | 37.401 | 100.000 | 22.434 | 0.9697 | -0.5 | 12.5/23.5 | 15/21 |
| 51.14 | 71.9483 | 3.1432 | 46.154 | 100.000 | 21.847 | 0.9379 | 0 | 13/24 | 15/21 |
| 61.37 | 66.4403 | 3.0774 | 47.524 | 100.000 | 21.851 | 0.9401 | 0 | 13/24 | 15/21 |
| 71.60 | 61.9056 | 3.0124 | 48.718 | 100.000 | 22.110 | 0.9423 | 0 | 13/24 | 15/21 |
| 81.83 | 57.6525 | 2.9655 | 49.248 | 100.000 | 21.605 | 0.9445 | 0 | 13/24 | 15/21 |
| 92.06 | 53.6807 | 2.9253 | 49.912 | 100.000 | 21.969 | 0.9467 | 0 | 13/24 | 15/21 |
| 100.43 | 50.5939 | 2.9010 | 50.221 | 100.000 | 22.393 | 0.9486 | 0 | 13/24 | 15/21 |

## Positions 6–12, IDs 15–21: reference

| Passes | Online loss | Ref L1 | Within±2 % | Retained % | Sign violations % | Widthfactor | Globaloffset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 0.0004 | 100.000 | 100.000 | 10.272 | 1.0000 | 0 | 15/21 | 15/21 |
| 10.23 | 77.9469 | 0.1720 | 100.000 | 100.000 | 4.730 | 1.0006 | 0 | 15/21 | 15/21 |
| 20.46 | 67.9904 | 0.3183 | 100.000 | 100.000 | 4.742 | 1.0011 | 0 | 15/21 | 15/21 |
| 30.69 | 60.5807 | 0.3630 | 100.000 | 100.000 | 4.050 | 1.0015 | 0 | 15/21 | 15/21 |
| 40.92 | 54.6609 | 0.3851 | 100.000 | 100.000 | 3.479 | 1.0018 | 0 | 15/21 | 15/21 |
| 51.14 | 49.7658 | 0.4337 | 100.000 | 100.000 | 3.423 | 1.0021 | 0 | 15/21 | 15/21 |
| 61.37 | 45.8470 | 0.4664 | 100.000 | 100.000 | 3.346 | 1.0024 | 0 | 15/21 | 15/21 |
| 71.60 | 42.5197 | 0.4867 | 100.000 | 100.000 | 3.653 | 1.0027 | 0 | 15/21 | 15/21 |
| 81.83 | 39.5335 | 0.4934 | 100.000 | 100.000 | 3.831 | 1.0029 | 0 | 15/21 | 15/21 |
| 92.06 | 36.7493 | 0.4929 | 100.000 | 100.000 | 4.426 | 1.0031 | 0 | 15/21 | 15/21 |
| 100.43 | 34.5186 | 0.4881 | 100.000 | 100.000 | 4.802 | 1.0032 | 0 | 15/21 | 15/21 |
