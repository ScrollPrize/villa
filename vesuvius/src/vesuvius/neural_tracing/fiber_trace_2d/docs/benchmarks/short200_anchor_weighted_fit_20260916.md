# Single-anchor iterative weighted fitting — 2026-09-16

Placement-only experiment on all 200 directed short-benchmark pieces, five
scrolls, 662.089848 mm. Same best EMA999 u22000 model and retained inference as
the existing anchor placement benchmark. Cell side: 32 base voxels; acceptance
ellipse: 40 normal / 80 tangential base voxels. No GT enters anchor fitting.

Output root:
`/ephemeral/hendrik/fiber3d_train2/evaluations/short200_anchor_weighted_fit_20260916_v1`

The experimental extractor fits one unoriented direction and position per owned
cell. It combines presence, sign-invariant directional agreement, a fixed
cell-centered Gaussian and a moving anchor-centered Gaussian. Each iteration
updates the weighted direction tensor and damps the centroid update by 0.5.
Fixed/moving sigmas are 16/12 base voxels. Iteration zero keeps the cell center
but still estimates direction. No cross-cell NMS or adaptive covariance.

This changes multiplicity, support domain and NMS as well as fitting; the legacy
comparison is not an isolated ablation of the centroid update. Accepted anchors
are anchor-to-piece matches, not a count of unique anchors over the volume.
Gap cells are reference arc distance divided by 32, not lattice cells crossed.
Empty bins are fixed one-cell arc bins. Direction-qualified empty bins lack an
accepted anchor whose fitted axis is within 20 degrees of the GT tangent.

## Reproduction and validation

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_anchor_weighted_fit_20260916_v1.json
```

Workspace RelWithDebInfo GCC 13 build. Approved test runner: 125 native tests,
263 Python tests passed, 8 optional skips. A new test fixture initially supplied
null rather than object metadata; fixture corrected and full runner passed.
Results include per-variant native binary/workload/manifest provenance and
per-scroll histograms, endpoint gaps and inspection rankings. Default extraction
and legacy cache identity are unchanged. Experimental construction caches are
separated; inference caches are shared. Timings include setup and cache effects,
not a controlled speed comparison. No downstream fiberlet graph tracing here.

## Results and interpretation

All 21 variants completed; sum of variant wall times 875.06 s (14m35s).
Legacy exactly reproduced the previous 69,141 matches and 0.676528% empty bins.
All variants had zero completely unanchored or single-position pieces.
The center-only gamma2 control gives 31.997% fewer accepted matches, 0.631%
empty positional bins versus 0.677% legacy, but 1.984% direction-qualified empty
bins versus 1.766%. Two iterations/gamma1/q2 give 31.712% fewer matches,
0.688% empty bins and the same 1.984% direction-qualified empty bins.
More iterations did not improve coverage; stronger angular weighting improves
conditional angle agreement modestly. No experimental row beats legacy on
20-degree-qualified coverage. Existing extraction remains default.

One-anchor fitting may reduce downstream construction work, but this is not yet
measured. It is not a demonstrated quality improvement. Center-only is a useful
cheap control for any downstream comparison. Angular P95 near 88 degrees reflects
that positional acceptance can include anchors from differently oriented fibers;
the direction-qualified coverage is more informative than conditional means alone.

## Representative per-scroll comparison

| Scroll | Variant | Accepted matches | Empty bins % | Empty bins requiring ≤20° % |
|---|---|---:|---:|---:|
| PHerc0139 | legacy | 23620 | 0.536 | 2.575 |
| PHerc0332 | legacy | 139 | 0.000 | 0.000 |
| PHerc1299 | legacy | 5129 | 1.479 | 2.564 |
| PHerc1451 | legacy | 97 | 6.897 | 6.897 |
| PHercParis4 | legacy | 40156 | 0.557 | 1.113 |
| PHerc0139 | weighted_i0_p2_q0 | 15789 | 0.823 | 3.577 |
| PHerc0332 | weighted_i0_p2_q0 | 113 | 3.125 | 6.250 |
| PHerc1299 | weighted_i0_p2_q0 | 4271 | 0.888 | 1.874 |
| PHerc1451 | weighted_i0_p2_q0 | 65 | 3.448 | 3.448 |
| PHercParis4 | weighted_i0_p2_q0 | 26780 | 0.433 | 1.052 |
| PHerc0139 | weighted_i2_p1_q2 | 15848 | 0.751 | 3.505 |
| PHerc0332 | weighted_i2_p1_q2 | 105 | 3.125 | 6.250 |
| PHerc1299 | weighted_i2_p1_q2 | 4277 | 0.986 | 2.170 |
| PHerc1451 | weighted_i2_p1_q2 | 67 | 3.448 | 3.448 |
| PHercParis4 | weighted_i2_p1_q2 | 26918 | 0.557 | 1.031 |

PHerc0332 and PHerc1451 each contain only one benchmark piece; their percentages
are coarse and should not be generalized. This is a failure-focused benchmark,
not an unbiased sample of the volume.

## Full matrix

Same short200 GT, model,32-base-voxel cells and40/80ellipse. Legacy is existing multi-component/refined/NMS extraction; experimental rows change fitting, multiplicity and NMS together. Gamma=presence exponent, q=direction exponent.

| Variant | Accepted anchors | Empty arc bins % | Gap P95 cells | Max gap cells | End max cells | Angle median ° | Angle P95 ° | ≤20° % | Empty bins requiring ≤20° % | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy | 69141 | 0.677 | 0.445 | 3.736 | 4.791 | 10.07 | 88.13 | 68.00 | 1.766 | 32.51 |
| weighted_i0_p1_q0 | 47129 | 0.631 | 0.655 | 3.151 | 4.638 | 9.10 | 87.92 | 69.76 | 2.007 | 40.59 |
| weighted_i0_p2_q0 | 47018 | 0.631 | 0.655 | 3.151 | 4.638 | 8.92 | 87.97 | 70.17 | 1.984 | 41.42 |
| weighted_i2_p1_q0 | 47215 | 0.722 | 0.660 | 3.414 | 4.761 | 9.03 | 87.93 | 70.27 | 2.075 | 43.13 |
| weighted_i2_p1_q2 | 47215 | 0.688 | 0.660 | 3.414 | 4.760 | 8.76 | 88.13 | 71.03 | 1.984 | 39.80 |
| weighted_i2_p1_q4 | 47229 | 0.688 | 0.660 | 3.415 | 4.759 | 8.74 | 88.15 | 71.06 | 1.995 | 43.73 |
| weighted_i2_p2_q0 | 47100 | 0.757 | 0.658 | 3.477 | 4.824 | 8.80 | 87.95 | 70.72 | 2.075 | 43.49 |
| weighted_i2_p2_q2 | 47125 | 0.745 | 0.655 | 3.477 | 4.824 | 8.67 | 88.09 | 71.24 | 2.007 | 38.07 |
| weighted_i2_p2_q4 | 47125 | 0.745 | 0.655 | 3.478 | 4.824 | 8.65 | 88.10 | 71.25 | 2.041 | 43.30 |
| weighted_i4_p1_q0 | 47276 | 0.757 | 0.657 | 3.496 | 4.810 | 8.94 | 87.95 | 70.57 | 2.156 | 42.03 |
| weighted_i4_p1_q2 | 47294 | 0.745 | 0.655 | 3.496 | 4.809 | 8.72 | 88.09 | 71.19 | 2.007 | 41.36 |
| weighted_i4_p1_q4 | 47278 | 0.757 | 0.655 | 3.496 | 4.808 | 8.69 | 88.12 | 71.20 | 2.053 | 44.33 |
| weighted_i4_p2_q0 | 47153 | 0.803 | 0.650 | 3.573 | 4.889 | 8.74 | 87.98 | 70.92 | 2.156 | 43.71 |
| weighted_i4_p2_q2 | 47137 | 0.814 | 0.648 | 3.574 | 4.889 | 8.62 | 88.07 | 71.43 | 2.121 | 40.10 |
| weighted_i4_p2_q4 | 47119 | 0.814 | 0.648 | 3.574 | 4.889 | 8.60 | 88.08 | 71.45 | 2.098 | 44.11 |
| weighted_i8_p1_q0 | 47309 | 0.791 | 0.655 | 3.530 | 4.836 | 8.89 | 87.96 | 70.58 | 2.179 | 41.53 |
| weighted_i8_p1_q2 | 47255 | 0.780 | 0.653 | 3.530 | 4.835 | 8.70 | 88.11 | 71.22 | 2.018 | 40.69 |
| weighted_i8_p1_q4 | 47254 | 0.780 | 0.653 | 3.531 | 4.835 | 8.68 | 88.11 | 71.24 | 2.064 | 44.11 |
| weighted_i8_p2_q0 | 47135 | 0.837 | 0.647 | 3.611 | 4.918 | 8.70 | 88.01 | 71.00 | 2.144 | 41.16 |
| weighted_i8_p2_q2 | 47146 | 0.837 | 0.645 | 3.611 | 4.918 | 8.61 | 88.08 | 71.40 | 2.121 | 41.49 |
| weighted_i8_p2_q4 | 47136 | 0.837 | 0.645 | 3.611 | 4.917 | 8.60 | 88.09 | 71.41 | 2.110 | 44.39 |

Timings include setup and cache effects; not controlled throughput comparisons. Full per-scroll statistics and histograms: matrix.json and each variant's report/metrics. No downstream graph tracing in this placement-only sweep.

