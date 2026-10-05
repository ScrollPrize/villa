# Isotropic20 gap-boundary and disabling-cost tuning

Frozen raw40500 iso20/full-prefix512/no-history traces. Orthogonal-only clean pieces, minima64/256. Constraint extraction, evidence, prepass and hard active-active continuity unchanged. New opt-in boundary prior charges once per adjacent active/Defect edge: interior disabledrun two boundaries, endpoint run one, wholly disabledtrace zero. No charge inside a contiguous disabledrun. Defect cost remains per weighted incident mass unless explicitly labeled length-based. Post-decode hard-sign projection is unchanged and may add defects without optimizing this prior. Gap counts include wholly disabled source traces; interior gaps are reported separately. Only retained continuation edges participate in the optimized boundary prior: prepass-disconnected pieces and post-decode projection mean geometric boundary counts need not equal optimized penalties.

Overlapping iso20 baseline: non-oracle50%,observedshift83.3333%,oracle80.7692%,constraint−25.2365%. Replay100% inherited for every row:0ordinary failures,6filteredstops,93.7463%retained. Oracle/constraint scores for event mode remain unavailable due to the existing calibration issue; this sweep does not alter or retry that evaluator. Accuracy/26; shift uses the observed-reference count N shown per row (usually24; some settings also lose reference8 and have23). Tied/unbounded readouts use deterministic rank1, not uniquely identified ladders.

| Case | Min | Boundary cost | Defect cost | Unary unit | Messages | Non-oracle % | Shift % (N) | Disabled length % | Gaps | Interior gaps | Whole disabled traces | Boundaries | Active components | Projected defects | Ties/unbounded | Seconds |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m256-b0-d100-i500 | 256 | 0 | 100 | incident mass | 500 | 34.62% | 16.67% (24) | 21.105% | 720 | 27 | 538 | 209 | 29 | 2 | 4/3 | 20.07 |
| m256-b0-d100-i500-rebuilt-control | 256 | 0 | 100 | incident mass | 500 | 34.62% | 16.67% (24) | 21.105% | 720 | 27 | 538 | 209 | 29 | 2 | 4/3 | 21.70 |
| m256-b10-d100-i1000 | 256 | 10 | 100 | incident mass | 1000 | 34.62% | 20.83% (24) | 20.997% | 718 | 25 | 539 | 204 | 21 | 1 | 4/3 | 365.63 |
| m256-b10-d100-i500 | 256 | 10 | 100 | incident mass | 500 | 34.62% | 20.83% (24) | 20.997% | 718 | 25 | 539 | 204 | 21 | 1 | 4/3 | 22.11 |
| m256-b10-d100-i5000 | 256 | 10 | 100 | incident mass | 5000 | 34.62% | 20.83% (24) | 20.997% | 718 | 25 | 539 | 204 | 21 | 1 | 4/3 | 153.53 |
| m256-b10-d1200-i500-length128 | 256 | 10 | 1200 | 128 base vx | 500 | 15.38% | 16.67% (24) | 16.795% | 627 | 52 | 456 | 223 | 189 | 1 | 3/3 | 25.83 |
| m256-b10-d25-i500 | 256 | 10 | 25 | incident mass | 500 | 34.62% | 16.67% (24) | 21.046% | 721 | 27 | 539 | 209 | 36 | 2 | 5/4 | 31.63 |
| m256-b10-d300-i1000-length128 | 256 | 10 | 300 | 128 base vx | 1000 | 19.23% | 20.83% (24) | 16.788% | 629 | 47 | 458 | 218 | 141 | 0 | 3/3 | 34.86 |
| m256-b10-d300-i500-length128 | 256 | 10 | 300 | 128 base vx | 500 | 19.23% | 20.83% (24) | 16.758% | 627 | 45 | 458 | 214 | 143 | 1 | 3/3 | 22.48 |
| m256-b10-d300-i5000-length128 | 256 | 10 | 300 | 128 base vx | 5000 | 19.23% | 20.83% (24) | 16.824% | 628 | 45 | 458 | 215 | 143 | 1 | 3/3 | 119.17 |
| m256-b10-d400-i500 | 256 | 10 | 400 | incident mass | 500 | 34.62% | 20.83% (24) | 21.110% | 719 | 25 | 539 | 205 | 22 | 1 | 4/3 | 26.03 |
| m256-b10-d4800-i500-length128 | 256 | 10 | 4800 | 128 base vx | 500 | 19.23% | 20.83% (24) | 16.833% | 630 | 53 | 456 | 227 | 145 | 0 | 3/3 | 29.84 |
| m256-b100-d100-i500 | 256 | 100 | 100 | incident mass | 500 | 34.62% | 16.67% (24) | 21.510% | 718 | 22 | 548 | 192 | 34 | 1 | 4/3 | 24.82 |
| m256-b1000-d100-i500 | 256 | 1000 | 100 | incident mass | 500 | 26.92% | 41.67% (24) | 18.823% | 618 | 5 | 571 | 52 | 44 | 0 | 4/2 | 22.05 |
| m256-b1000-d1200-i500-length128 | 256 | 1000 | 1200 | 128 base vx | 500 | 15.38% | 17.39% (23) | 17.174% | 603 | 21 | 480 | 144 | 190 | 0 | 3/3 | 23.85 |
| m256-b1000-d300-i500-length128 | 256 | 1000 | 300 | 128 base vx | 500 | 15.38% | 13.04% (23) | 16.945% | 533 | 0 | 511 | 22 | 153 | 0 | 1/1 | 21.61 |
| m256-b1000-d4800-i500-length128 | 256 | 1000 | 4800 | 128 base vx | 500 | 19.23% | 20.83% (24) | 16.747% | 619 | 42 | 462 | 199 | 145 | 2 | 4/4 | 25.08 |
| m256-b10000-d100-i500 | 256 | 10000 | 100 | incident mass | 500 | 30.77% | 50.00% (24) | 25.021% | 662 | 0 | 661 | 1 | 89 | 0 | 2/2 | 33.54 |
| m64-b0-d100-i500 | 64 | 0 | 100 | incident mass | 500 | 15.38% | 12.50% (24) | 12.920% | 1096 | 481 | 394 | 1183 | 23 | 9 | 1/1 | 32.58 |
| m64-b0-d100-i500-rebuilt-control | 64 | 0 | 100 | incident mass | 500 | 15.38% | 12.50% (24) | 12.920% | 1096 | 481 | 394 | 1183 | 23 | 9 | 1/1 | 34.59 |
| m64-b10-d100-i500 | 64 | 10 | 100 | incident mass | 500 | 15.38% | 12.50% (24) | 12.505% | 1054 | 462 | 395 | 1121 | 25 | 5 | 1/1 | 31.93 |
| m64-b10-d1200-i500-length128 | 64 | 10 | 1200 | 128 base vx | 500 | 19.23% | 16.67% (24) | 11.716% | 1004 | 484 | 387 | 1101 | 139 | 4 | 1/1 | 32.63 |
| m64-b10-d300-i500-length128 | 64 | 10 | 300 | 128 base vx | 500 | 19.23% | 20.83% (24) | 11.737% | 978 | 453 | 387 | 1044 | 142 | 5 | 1/1 | 33.78 |
| m64-b10-d4800-i1000-length128 | 64 | 10 | 4800 | 128 base vx | 1000 | 19.23% | 16.67% (24) | 9.273% | 878 | 431 | 387 | 922 | 122 | 10 | 1/1 | 62.81 |
| m64-b10-d4800-i500-length128 | 64 | 10 | 4800 | 128 base vx | 500 | 23.08% | 25.00% (24) | 9.289% | 878 | 430 | 387 | 921 | 121 | 4 | 1/1 | 36.02 |
| m64-b10-d4800-i5000-length128 | 64 | 10 | 4800 | 128 base vx | 5000 | 19.23% | 16.67% (24) | 9.272% | 869 | 423 | 387 | 905 | 124 | 6 | 1/1 | 212.69 |
| m64-b100-d100-i500 | 64 | 100 | 100 | incident mass | 500 | 15.38% | 16.67% (24) | 12.141% | 958 | 376 | 397 | 937 | 13 | 6 | 1/1 | 30.26 |
| m64-b1000-d100-i1000 | 64 | 1000 | 100 | incident mass | 1000 | 26.92% | 50.00% (24) | 11.797% | 613 | 76 | 433 | 256 | 22 | 0 | 2/2 | 65.90 |
| m64-b1000-d100-i500 | 64 | 1000 | 100 | incident mass | 500 | 30.77% | 54.17% (24) | 11.901% | 608 | 56 | 441 | 223 | 25 | 2 | 2/2 | 32.12 |
| m64-b1000-d100-i5000 | 64 | 1000 | 100 | incident mass | 5000 | 26.92% | 50.00% (24) | 11.914% | 621 | 82 | 433 | 270 | 23 | 1 | 2/2 | 442.19 |
| m64-b1000-d1200-i500-length128 | 64 | 1000 | 1200 | 128 base vx | 500 | 19.23% | 20.83% (24) | 12.626% | 665 | 122 | 415 | 372 | 162 | 0 | 3/3 | 33.86 |
| m64-b1000-d25-i500 | 64 | 1000 | 25 | incident mass | 500 | 19.23% | 29.17% (24) | 28.186% | 748 | 19 | 645 | 122 | 61 | 0 | 5/5 | 38.71 |
| m64-b1000-d300-i1000-length128 | 64 | 1000 | 300 | 128 base vx | 1000 | 30.77% | 58.33% (24) | 14.615% | 567 | 14 | 494 | 87 | 113 | 1 | 3/2 | 72.13 |
| m64-b1000-d300-i500-length128 | 64 | 1000 | 300 | 128 base vx | 500 | 42.31% | 62.50% (24) | 15.775% | 568 | 5 | 507 | 66 | 113 | 0 | 4/3 | 32.32 |
| m64-b1000-d300-i5000-length128 | 64 | 1000 | 300 | 128 base vx | 5000 | 30.77% | 58.33% (24) | 14.602% | 565 | 13 | 494 | 84 | 113 | 0 | 3/2 | 236.11 |
| m64-b1000-d400-i500 | 64 | 1000 | 400 | incident mass | 500 | 19.23% | 16.67% (24) | 12.722% | 852 | 241 | 407 | 686 | 11 | 0 | 3/3 | 41.61 |
| m64-b1000-d4800-i500-length128 | 64 | 1000 | 4800 | 128 base vx | 500 | 23.08% | 25.00% (24) | 9.961% | 802 | 324 | 389 | 737 | 131 | 2 | 1/1 | 37.31 |
| m64-b10000-d100-i500 | 64 | 10000 | 100 | incident mass | 500 | 19.23% | 29.17% (24) | 28.789% | 720 | 19 | 678 | 61 | 103 | 2 | 6/6 | 33.23 |

## Matched comparison (500-message tuning budget)

Higher percentages are better except disabled length. Replay is inherited, not rerun. “Unavailable” is the existing local-event oracle calibration failure, not zero. Length winners are selected across the tested boundary settings.

| Method | Replay % | Constraint score % | Oracle accuracy % | Non-oracle accuracy % | Observed shift score % | Disabled length % |
|---|---:|---:|---:|---:|---:|---:|
| Overlapping iso20 baseline | 100.00% | −25.24% | 80.77% | 50.00% | 83.33% | — |
| Minimum64, untuned | 100.00% | Unavailable | Unavailable | 15.38% | 12.50% | 12.920% |
| Minimum64, incident-cost tuned | 100.00% | Unavailable | Unavailable | 30.77% | 54.17% | 11.901% |
| Minimum64, length-cost tuned | 100.00% | Unavailable | Unavailable | 42.31% | 62.50% | 15.775% |
| Minimum256, untuned | 100.00% | Unavailable | Unavailable | 34.62% | 16.67% | 21.105% |
| Minimum256, incident-cost tuned | 100.00% | Unavailable | Unavailable | 34.62% | 20.83% | 20.997% |
| Minimum256, length-cost tuned | 100.00% | Unavailable | Unavailable | 19.23% | 20.83% | 16.747% |

## Stage notes

StageA complete: boundarycost0/10/100/1000/10000, defect100; zero controls exactly match historical initial states.
StageB complete: defect25/100/400 at bestboundary perminimum; selection accuracy then shift then fewergaps.
Higher-budget1000/5000 checks completed for each minimum winner. No production defaults changed.
StageC: physical-length unary via existing native length model; common boundarycost10, disablecost300/1200/4800 per128basevoxels at both minima. Central1200 approximately matches12.14 signed incidences per128voxels at cost100 in the min64 control; not exact evidence-mass matching. Prepass unchanged. Rebuilt zero controls checked against historical states.
StageC and its1000/5000budget checks complete. No production defaults changed.
StageC: physical-length unary via existing native length model; common boundarycost1000, disablecost300/1200/4800 per128basevoxels at both minima. Central1200 approximately matches12.14 signed incidences per128voxels at cost100 in the min64 control; not exact evidence-mass matching. Prepass unchanged. Rebuilt zero controls checked against historical states.
Length stage at boundarycost1000 complete. Primary length sweep has1000/5000budget checks; strong-boundary interaction receives those checks only if it beats the same-minimum incident-cost winner. No production defaults changed.

Artifacts:`/ephemeral/hendrik/fiber_crop_bench/20261002_iso20_gap_tuning_v1`. Approved no-argument runner `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, actions `iso20_gap_tuning` and `iso20_gap_length_tuning` in scripts/training_failure_mining_control.json. CPU128threads,RelWithDebInfo,one repetition; no inference/retracing. Parameters selected on this crop, not a held-out validation. Full ladders/convergence/geometry checks in adjacent JSON.
