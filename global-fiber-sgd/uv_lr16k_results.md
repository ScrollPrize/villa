# Learning-rate scaling at batch16384

All3runs completed exit0 under one exclusive GPU1/2/3 manager lease,24CPU,
32GiB reserved perGPU. Same fresh radial CCW151,seed0,full compiled quadratic,
4032updates,66,060,288sampled constraints (~10passes with replacement).
Only embedding/state optimizer LR differs; winding-width AdamLR stays2.
No warmstart/referenceinitialization or delayed/frozen variables. No code changes.

| LR | Solver seconds | Ref L1(windings) | Within +/-2 % | Retained length % | Sign violations % | Peak allocated GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 397.86 | 11.745175 | 13.1990 | 66.6851 | 3.6633 | 7.430 |
| 0.06 | 414.58 | 11.660019 | 13.2925 | 65.4857 | 3.1654 | 7.430 |
| 0.12 | 435.26 | 11.571921 | 13.2302 | 62.1046 | 3.3124 | 7.430 |

ReferenceL1 is global-offset corrected (+12), evaluated on retained matched
source fibers; changing rejection changes the evaluated population. It is not
proof of improved placement for identical fibers. Final sign audit uses entire
constraint dataset and retained known-sign denominator. Fitq10/90 remains49/122.
Time includes compilation,prior diagnostics and final hardaudit but not final
reference metric/save,managerqueue/preparation. Memory PyTorchallocated,notreserved.

The0.03control exactly reproduces the previous16k reference/retention/sign metrics.
PreviousB4096LR.03:519.61s,L1=11.723346,within2=13.213415%,retained66.016892%,
sign3.330811%. B16kLR.06 takes20.2%less recordedtime, slightlybetterL1/sign,
retains0.5312percentagepointslesslength. Versus16kLR.03,it reducesL1by.085156
and signviolationsby.497914points,with1.199353pointsless retainedlength.

Recommendation:0.06 is a reasonable balanced16k candidate;0.12rejects substantially
more length yet has worse signviolation andwithin2 metrics than0.06. No defaults
changed. Single seed,no significance claim. This is not exact batch-equivalent
optimization; Adam history and update counts differ from4k. WidthLR not scaled.
No LRwarmup,schedule or extra optimizationstage added.

## Command

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_lr16k_matrix.json
```

Configs run_uv_lr16k_{0.03,0.06,0.12}.json. Outputs under
 data/uv_lr16k_{0.03,0.06,0.12}_v1 with history,solution,summary,runnerlog.
Use new output paths for reruns; completed outputs protected. Independent config
review approved exactsettings. No newcode required; existing91-test regression
suite covered runner/solver. Runtimevalidation:all3exit0,finite metrics,controlrepeat.

## Complete progression

Intermediate sign diagnostics use100kfixedsample,finalusesfullgraph. Online loss
is mean since previousdiagnostic; final interval has only2updates and should not
be used alone forconvergence ranking. Passmeans sampled event-equivalent.

### LR 0.03

| Pass | Seconds | Online loss | Ref L1 | Within2 % | Retained % | Sign % | Fit q10/90 | Ref q10/90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 8.11 | — | 12.2423 | 12.9912 | 100.0000 | 40.9860 | 49/122 | 42/124 |
| 0.999 | 45.69 | 89.2786 | 11.9989 | 13.0518 | 93.9523 | 18.5279 | 49/122 | 42/124 |
| 1.999 | 82.60 | 21.8253 | 11.9060 | 13.0962 | 79.9054 | 11.5529 | 49/122 | 42/124 |
| 2.998 | 119.78 | 9.4330 | 11.8607 | 13.1320 | 72.7196 | 8.5846 | 49/122 | 42/124 |
| 3.998 | 157.18 | 6.0234 | 11.8271 | 13.1423 | 69.4813 | 6.8149 | 49/122 | 42/124 |
| 4.997 | 195.00 | 4.6514 | 11.8046 | 13.1571 | 67.9580 | 5.5829 | 49/122 | 42/125 |
| 5.997 | 232.93 | 3.9255 | 11.7859 | 13.1726 | 67.2590 | 5.0111 | 48/122 | 42/125 |
| 6.996 | 271.09 | 3.4901 | 11.7745 | 13.1666 | 66.9217 | 4.5870 | 48/122 | 42/125 |
| 7.996 | 309.31 | 3.2031 | 11.7622 | 13.1836 | 66.7409 | 4.0552 | 48/122 | 42/125 |
| 8.995 | 347.73 | 2.9962 | 11.7527 | 13.1976 | 66.6798 | 3.8479 | 48/122 | 42/125 |
| 9.995 | 386.38 | 2.8356 | 11.7450 | 13.1980 | 66.6843 | 3.6318 | 48/122 | 42/125 |
| 10.000 | 397.86 | 2.7580 | 11.7452 | 13.1990 | 66.6851 | 3.6633 | 48/122 | 42/125 |

### LR 0.06

| Pass | Seconds | Online loss | Ref L1 | Within2 % | Retained % | Sign % | Fit q10/90 | Ref q10/90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 8.08 | — | 12.2423 | 12.9912 | 100.0000 | 40.9860 | 49/122 | 42/124 |
| 0.999 | 46.62 | 79.6046 | 11.9436 | 13.0534 | 82.3299 | 14.4115 | 49/122 | 42/124 |
| 1.999 | 84.99 | 12.1649 | 11.8352 | 13.1537 | 69.4661 | 8.7030 | 49/122 | 42/124 |
| 2.998 | 123.74 | 5.7220 | 11.7804 | 13.1606 | 66.1915 | 6.2699 | 49/122 | 42/125 |
| 3.998 | 162.83 | 4.1489 | 11.7454 | 13.2124 | 65.3634 | 5.0778 | 49/122 | 42/125 |
| 4.997 | 202.26 | 3.5117 | 11.7198 | 13.2388 | 65.1557 | 4.2871 | 49/122 | 42/125 |
| 5.997 | 242.02 | 3.1634 | 11.7056 | 13.2709 | 65.1482 | 3.9501 | 49/122 | 42/125 |
| 6.996 | 281.99 | 2.9396 | 11.6918 | 13.2637 | 65.2369 | 3.6658 | 49/122 | 42/125 |
| 7.996 | 322.18 | 2.8200 | 11.6803 | 13.2480 | 65.3090 | 3.3189 | 49/122 | 42/125 |
| 8.995 | 362.53 | 2.6940 | 11.6693 | 13.2956 | 65.3963 | 3.3500 | 49/122 | 42/125 |
| 9.995 | 403.10 | 2.5916 | 11.6601 | 13.2961 | 65.4851 | 3.1108 | 49/122 | 42/125 |
| 10.000 | 414.58 | 2.5980 | 11.6600 | 13.2925 | 65.4857 | 3.1654 | 49/122 | 42/125 |

### LR 0.12

| Pass | Seconds | Online loss | Ref L1 | Within2 % | Retained % | Sign % | Fit q10/90 | Ref q10/90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 7.92 | — | 12.2423 | 12.9912 | 100.0000 | 40.9860 | 49/122 | 42/124 |
| 0.999 | 47.58 | 110.5579 | 11.8740 | 13.0915 | 70.5224 | 13.6741 | 49/122 | 42/124 |
| 1.999 | 87.93 | 33.6819 | 11.7568 | 13.1179 | 62.8190 | 8.6191 | 49/122 | 42/124 |
| 2.998 | 128.89 | 15.4091 | 11.6995 | 13.1505 | 61.7720 | 6.2430 | 49/122 | 42/124 |
| 3.998 | 170.40 | 8.6814 | 11.6664 | 13.1659 | 61.6289 | 5.1864 | 49/122 | 42/124 |
| 4.997 | 212.12 | 6.2246 | 11.6440 | 13.2181 | 61.6786 | 4.3977 | 49/122 | 42/124 |
| 5.997 | 254.14 | 5.1653 | 11.6250 | 13.2146 | 61.7427 | 3.8135 | 49/122 | 42/124 |
| 6.996 | 296.43 | 4.7244 | 11.6073 | 13.2618 | 61.8420 | 3.6752 | 49/122 | 42/124 |
| 7.996 | 338.86 | 4.4156 | 11.5946 | 13.2662 | 61.9389 | 3.5017 | 49/122 | 42/124 |
| 8.995 | 381.38 | 4.2210 | 11.5820 | 13.2590 | 62.0143 | 3.3549 | 49/122 | 42/124 |
| 9.995 | 424.08 | 4.0575 | 11.5720 | 13.2306 | 62.1036 | 3.1699 | 49/122 | 42/124 |
| 10.000 | 435.26 | 4.6824 | 11.5719 | 13.2302 | 62.1046 | 3.3124 | 49/122 | 42/124 |
