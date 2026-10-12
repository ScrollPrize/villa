# Ten-reference-winding regional solve

## Dataset and protocol

Select original AFV fibers by their existing geometry-only, whole-fiber median
reference assignment to IDs10–19 inclusive. These are the first ten recorded
reference winding IDs, not solver buckets0–9. Intersect actual fiber curves with
base Z=[30000,50000]. Keep disconnected slab reentries as separate pieces.
Remove constraints whose endpoints do not both survive; preserve signs and
measured winding distances of retained constraints. Earlier normal/HV/direction
assignments are reused unchanged.

Both solves use151 winding hypotheses, CCW, the compiled squared objective and
combined coordinate kernel, all variables trainable from the beginning. Fresh
initialization inherits full-corpus radial centers and widths; it does not
stretch this small radial range over151 buckets. The reference run additionally
initializes assignment and supported widths from reference, but does not add a
reference fitting loss or replace the radial width prior.

100 iterations means100 sampled constraint-dataset equivalents with random
minibatch Adam updates (the established current protocol), not100 optimizer
updates. Report the actual equivalent count after integer update rounding.
Six exclusive GPUs, FP32, seed0, global batch24576 (4096/GPU), stateLR0.0073484692
(sqrt scaling of the full6GPU lowLR run), widthLR2 unchanged. Same settings in
both runs. User explicitly requested retaining multi-GPU execution for speed.

Reference L1 and within-two metrics have a global L1-optimal winding offset.
They weight original source fibers equally and use the median predicted winding
over the retained clipped curve. Reference labels remain the original whole-
source matched labels. The retained population can change with rejection;
these metrics therefore are not a fixed-cohort accuracy comparison.
Loss is the mean online objective over the reporting interval, not a frozen-
state whole-graph objective. Regional constraints are an induced subgraph;
missing outside neighbors are intentional, so this is not proof of kollesis.

## Reproduction

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/extract_uv_reference_subset.json
```

Extraction outputs:
`data/uv_reference_w10_19_z30k50k_v1/`.
## Completed results (2026-10-12)

Extraction took15.240s. Selected7218sources by reference winding before Z clipping;
2651sources remain as2662pieces (1077H,1585V), totaling4027780.258base voxels
of arc. There are30048constraints (23039parallel,7009perpendicular), one original
continuation, and134zero-degree pieces retained rather than silently filtered.
All chosen reference IDs are10–19; exact output Z bounds30000/50000.

Boundary accounting:88635edges join winding-selected sources before Z cropping;
30971edges join the2651surviving sources before endpoint clipping;30048survive
endpoint clipping.40017parent edges touch those sources, of which9969are removed
because an endpoint/source is outside the subset. No extra continuation connects
slab reentries. Production audit checks91191clipped curve samples.

| Initialization | Actual passes | Online loss | Ref L1 | Within ±2 % | Retained length % | Sign violations % | Regional width factor | Solve seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh | 100.601 | 43.2843 | 2.3063 | 62.1652 | 100.0000 | 22.5004 | 0.9072 | 5.821 |
| reference | 100.601 | 32.3750 | 0.4885 | 100.0000 | 100.0000 | 9.8259 | 1.0083 | 3.792 |

Both global reference offsets remain0. Widthfactor averages the nine fully
supported reference intervals11–19 (10is not fully covered by landmark mapping),
not all123global intervals. Regional mean absolute width errors: fresh9.2756%,
reference2.1691%. No missing-reference fibers in either result.

123optimizer updates at24576events/update correspond to100.600639sampled
passes. This small corpus therefore receives far fewer updates than the full
volume's3360updates for100passes. Neither run demonstrates convergence; low
rejection at this budget must not be interpreted as evidence of consistency.
The objective is still large and reference-initialized geometry drifts away
from reference. This experiment alone does not identify kollesis.

Final online loss covers the last3updates; preceding rows cover12updates each.
Initial state has no online objective evaluation; it is shown as —, not zero.

## Timing

Exclusive6×H100, FP32, global24576events/update,4096/GPU,123updates each,
first5updates discarded from CUDA update statistics. Peak allocated1.161GiB/GPU.

| Initialization | Solve wall s | Update mean ms | p50 ms | p95 ms | Worker launch-through-summary s |
|---|---:|---:|---:|---:|---:|
| fresh | 5.821 | 17.388 | 16.192 | 20.862 | 16.710 |
| reference | 3.792 | 17.727 | 15.997 | 21.043 | 15.002 |

Both-run suite elapsed **49.889s** from frozen suite
config to completed summary (includes manager wait/startup/preflight). Individual
worker intervals use config/summary file timestamps; solve timer includes
optimization,diagnostics,save but excludes worker loading/preflight. One trial
per initialization; warm caches may explain timing differences. No single-GPU
speedup claim. Manager task c01de29e-1070-4e4c-921d-10eb305820f6 finished and
released all resources.

## fresh: full progression

| Passes | Loss | Ref L1 | Within ±2 % | Retained length % | Sign violations % | Width factor | Fit q10/q90 | Reference q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 2.7722 | 52.094 | 100.000 | 37.692 | 0.8802 | 9/22 | 10/19 |
| 9.81 | 103.3216 | 2.6601 | 54.696 | 100.000 | 29.938 | 0.8829 | 10/22 | 10/19 |
| 19.63 | 89.7082 | 2.6575 | 54.960 | 100.000 | 28.492 | 0.8854 | 10/22 | 10/19 |
| 29.44 | 80.1087 | 2.6292 | 55.300 | 100.000 | 23.684 | 0.8879 | 10/22 | 10/19 |
| 39.26 | 72.5305 | 2.5941 | 55.903 | 100.000 | 23.018 | 0.8907 | 10/22 | 10/19 |
| 49.07 | 66.2787 | 2.5338 | 57.111 | 100.000 | 23.925 | 0.8935 | 10/22 | 10/19 |
| 58.89 | 60.9912 | 2.4700 | 58.506 | 100.000 | 23.867 | 0.8962 | 10/22 | 10/19 |
| 68.70 | 56.4131 | 2.4259 | 59.147 | 100.000 | 24.004 | 0.8990 | 10/21 | 10/19 |
| 78.52 | 52.2712 | 2.3840 | 60.468 | 100.000 | 23.389 | 0.9017 | 10/21 | 10/19 |
| 88.33 | 48.5114 | 2.3485 | 60.920 | 100.000 | 23.108 | 0.9042 | 10/21 | 10/19 |
| 98.15 | 45.2865 | 2.3138 | 61.939 | 100.000 | 22.321 | 0.9067 | 10/21 | 10/19 |
| 100.60 | 43.2843 | 2.3063 | 62.165 | 100.000 | 22.500 | 0.9072 | 10/21 | 10/19 |

## reference: full progression

| Passes | Loss | Ref L1 | Within ±2 % | Retained length % | Sign violations % | Width factor | Fit q10/q90 | Reference q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 0.0011 | 100.000 | 100.000 | 15.750 | 1.0000 | 10/19 | 10/19 |
| 9.81 | 79.7454 | 0.1845 | 100.000 | 100.000 | 8.139 | 1.0020 | 10/19 | 10/19 |
| 19.63 | 68.8951 | 0.3282 | 100.000 | 100.000 | 7.794 | 1.0039 | 10/19 | 10/19 |
| 29.44 | 61.1301 | 0.3576 | 100.000 | 100.000 | 6.305 | 1.0053 | 10/19 | 10/19 |
| 39.26 | 54.8654 | 0.3957 | 100.000 | 100.000 | 5.773 | 1.0065 | 10/19 | 10/19 |
| 49.07 | 49.9065 | 0.4425 | 100.000 | 100.000 | 5.737 | 1.0074 | 10/19 | 10/19 |
| 58.89 | 45.9044 | 0.4749 | 100.000 | 100.000 | 5.747 | 1.0080 | 10/19 | 10/19 |
| 68.70 | 42.3850 | 0.4874 | 100.000 | 100.000 | 7.103 | 1.0083 | 10/19 | 10/19 |
| 78.52 | 39.2787 | 0.4866 | 100.000 | 100.000 | 7.333 | 1.0085 | 10/19 | 10/19 |
| 88.33 | 36.3704 | 0.4874 | 100.000 | 100.000 | 9.063 | 1.0085 | 10/19 | 10/19 |
| 98.15 | 33.8824 | 0.4870 | 100.000 | 100.000 | 9.596 | 1.0084 | 10/19 | 10/19 |
| 100.60 | 32.3750 | 0.4885 | 100.000 | 100.000 | 9.826 | 1.0083 | 10/19 | 10/19 |

## Solve command and artifacts

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_subset100_suite.json
```

Completed outputs are protected; use new output roots for a fresh rerun.
Individual configurations:run_uv_subset100_fresh.json and
run_uv_subset100_reference.json. Outputs:data/uv_subset100_{fresh,reference}_v1
contain config,history,summary,solution; suite:data/uv_subset100_suite_v1.

Extraction:manifest.json records signatures; tensors.npz is the induced graph;
clipped_curves.npz stores XYZ/original arcs/offsets/source IDs for inspection.
projection.npz is a diagnostic/initialization cache, not a new surface-UV
projection for rendering. No visualization requested or generated.

Validation: independent plan/code/config reviews; synthetic clipping/reentry,
negative-direction endpoint mapping, boundary-link roundoff, physical reanchor,
parent prior preservation, diagnostic interval filtering, and production audit.
CUDA combined coordinate values/gradients and bitwise repeat, six-rank
concatenated-gradient preflight passed for each solve. All final state arrays
finite. See task_log for complete targeted test result.
