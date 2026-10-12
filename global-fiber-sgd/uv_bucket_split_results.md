# Bucket topology plus weaker positional loss: 10k results

## Method

Opt-in squared objective, bucket distance weight1, existing metric-position
distance weight0.1. Sign hinge and fixed selected/expected violation charge use
only bucket separation plus the H/V half-layer ordering convention and a fixed
integer correction from recorded angular geometry. No optimized phase, offset,
or width enters bucket topology. Phase/offset/V/continuation regularization,
class balancing10:1 perpendicular:parallel, fixed charge32 and all joint
updates unchanged. No reference prior is added in reference initialization.

Same positions5–11 (referenceIDs14–20), Z30000–50000;2199pieces,24771events,
151hypotheses,CCW,seed0,FP32,compiled pair/combined coordinate backends. State
LR0.07348469228349533,widthLR2,globalbatch24576,6exclusiveH100 GPUs0–5.
10080 sampled minibatch updates =10000.649 pass equivalents, not exhaustive
epochs. Diagnostics every1008updates. Previous normal/HV/direction stages reused.

## Reproduce

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_bucket_split_suite.json
```

Completed output roots must not be overwritten; use new output roots for reruns.
Configs: run_uv_bucket_split_{fresh,reference}.json.
Outputs: data/uv_bucket_split_{fresh,reference}_v1/{history.json,summary.json,solution.npz}.
Manager7b42a4cb-46be-4e0e-af0e-a0d4a98d13e8 finished and released all resources.

## Metrics and caveats

Loss is online mean over preceding1k sampled passes; initial row has no loss.
RefL1/within2 use global median offset correction on retained fibers; changes
in rejection change the cohort. Widthfactor = mean fitted/reference width after
offset alignment. Bucket sign errors are NOT the old coordinate-based sign
errors. Old/new losses use different objectives and cannot be ranked directly.
The fixed integer correction uses immutable angle observations; large angular
fold distortions remain a model limitation. Shared phase is still unbounded,
so bucket and metric geometry may disagree; existing weak priors remain as
requested. Zero bucket violations does not prove correct physical reconstruction.

## Fresh

| Pass equivalents | Loss | Ref L1 | Within±2 % | Retained length % | Perp bucket signs % | Parallel bucket signs % | Width factor | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 3.1957 | 45.119 | 100.000 | 37.7750 | 23.8614 | 0.9829 | 11.0–23.0 | 15.0–20.0 |
| 1000.06 | 75.1463 | 2.2034 | 64.689 | 32.938 | 0.0000 | 0.0000 | 0.9202 | 13.0–23.0 | 14.0–20.0 |
| 2000.13 | 5.3558 | 2.2288 | 63.753 | 36.573 | 0.0000 | 0.0000 | 0.9156 | 12.0–23.0 | 14.0–20.0 |
| 3000.19 | 4.8892 | 2.2833 | 62.286 | 38.881 | 0.0000 | 0.0000 | 0.9133 | 12.0–23.0 | 14.0–20.0 |
| 4000.26 | 4.5903 | 2.2663 | 63.665 | 41.669 | 0.0000 | 0.0000 | 0.9126 | 13.0–23.0 | 14.0–20.0 |
| 5000.32 | 4.3424 | 2.2338 | 64.361 | 43.480 | 0.0000 | 0.0000 | 0.9124 | 13.0–23.0 | 14.0–20.0 |
| 6000.39 | 4.1525 | 2.2258 | 64.530 | 45.295 | 0.0000 | 0.0000 | 0.9122 | 13.0–23.0 | 14.0–20.0 |
| 7000.45 | 3.9995 | 2.2019 | 64.408 | 46.758 | 0.0000 | 0.0000 | 0.9122 | 13.0–23.0 | 14.0–20.0 |
| 8000.52 | 3.8797 | 2.1909 | 64.825 | 48.210 | 0.0000 | 0.0000 | 0.9122 | 13.0–23.0 | 14.0–20.0 |
| 9000.58 | 3.7615 | 2.1784 | 64.859 | 49.411 | 0.0000 | 0.0000 | 0.9123 | 13.0–23.0 | 14.0–20.0 |
| 10000.65 | 3.6809 | 2.1762 | 64.905 | 50.411 | 0.0000 | 0.0000 | 0.9123 | 13.0–23.0 | 14.0–20.0 |

## Reference initialized

| Pass equivalents | Loss | Ref L1 | Within±2 % | Retained length % | Perp bucket signs % | Parallel bucket signs % | Width factor | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 0.0009 | 100.000 | 100.000 | 9.2860 | 2.0929 | 1.0000 | 15.0–20.0 | 15.0–20.0 |
| 1000.06 | 49.4695 | 0.5581 | 98.623 | 66.887 | 0.0000 | 0.0000 | 0.9433 | 14.0–20.0 | 14.0–20.0 |
| 2000.13 | 3.3896 | 0.5884 | 98.636 | 67.694 | 0.0000 | 0.0000 | 0.9138 | 14.0–20.0 | 14.0–20.0 |
| 3000.19 | 3.1405 | 0.5961 | 98.535 | 68.268 | 0.0000 | 0.0000 | 0.9079 | 14.0–20.0 | 14.0–20.0 |
| 4000.26 | 3.0184 | 0.5910 | 98.720 | 68.846 | 0.0000 | 0.0000 | 0.9086 | 14.0–20.0 | 14.0–20.0 |
| 5000.32 | 2.9248 | 0.6041 | 98.504 | 69.751 | 0.0000 | 0.0000 | 0.9085 | 14.0–20.0 | 14.0–20.0 |
| 6000.39 | 2.8165 | 0.6045 | 98.630 | 70.384 | 0.0000 | 0.0000 | 0.9085 | 14.0–20.0 | 14.0–20.0 |
| 7000.45 | 2.7297 | 0.6026 | 98.700 | 71.041 | 0.0000 | 0.0000 | 0.9092 | 14.0–20.0 | 14.0–20.0 |
| 8000.52 | 2.6622 | 0.6164 | 98.542 | 71.730 | 0.0000 | 0.0000 | 0.9086 | 14.0–20.0 | 14.0–20.0 |
| 9000.58 | 2.5976 | 0.6069 | 98.608 | 72.337 | 0.0000 | 0.0000 | 0.9093 | 14.0–20.0 | 14.0–20.0 |
| 10000.65 | 2.5355 | 0.6235 | 98.618 | 73.071 | 0.0000 | 0.0000 | 0.9088 | 14.0–20.0 | 14.0–20.0 |

## Final comparison with previous high-LR formulation

| Initialization | Objective | Loss (different objectives) | Ref L1 | Within±2 % | Retained length % | Width factor |
|---|---|---:|---:|---:|---:|---:|
| fresh | Previous | 2.7149 | 1.2058 | 86.635 | 65.508 | 0.8514 |
| fresh | Split | 3.6809 | 2.1762 | 64.905 | 50.411 | 0.9123 |
| reference | Previous | 2.5073 | 0.6331 | 97.615 | 78.247 | 0.9082 |
| reference | Split | 2.5355 | 0.6235 | 98.618 | 73.071 | 0.9088 |

Fresh quality is worse (higherL1,lowerretention). ReferenceL1 is slightly
better and within2 higher, but retention is lower. Not a demonstrated overall
improvement; preserve as opt-in experiment, do not replace existing loss default.

## Speed and validation

| Initialization | Previous solve s | Split solve s | Split update mean/p50/p95 ms | Split peak GiB/GPU |
|---|---:|---:|---|---:|
| fresh | 185.607 | 166.282 | 16.062/16.033/16.350 | 1.159 |
| reference | 185.917 | 169.701 | 16.644/16.616/16.818 | 1.161 |

Single trial per initialization,10080updates,first5excluded from update timing,
max rank timings; wall excludes manager wait/process startup.89 CPUtests passed,
including seam rounding,0.1position gradient, topology coordinate independence,
fixed selected sign and matching audit. Both production sixGPU compiled/eager
gradient checks and coordinate value/gradient/repeat checks passed. Final states
and widths finite; suite exit0. Independent plan and final implementation review.
