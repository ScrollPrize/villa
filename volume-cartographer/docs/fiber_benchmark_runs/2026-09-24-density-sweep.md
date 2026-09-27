# Density-modulated disablement: parameter sweep

Completed 51 parameter settings plus an unchanged control on the reverse-consistency 2048 Paris4 crop. Each statistic receives a 3x3 coarse grid and two four-neighbor refinements. Selected by the existing reference-removal score; no default promotion.

## Mapping and protocol

`m(d) = max(floor, scale / (scale + d))`; disablement cost is the original `100 * incident winding measurements` times `m(d)`. Density is native forward tracer loss per prediction voxel, not presence. Mean and median are arc-length weighted; worst is maximum segment loss density.

Coarse scale: 0.05, 0.5, 5; floor: 0.01, 0.1, 0.5. Refinement tests axial scale/floor neighbors independently per statistic, with scale factors sqrt(10), then 10**0.25 and geometric floor midpoints. Deterministic score ties prefer the larger mean multiplier, then name. This is a finite direct search, not a global optimum guarantee.

Input traces, normals, suppression/reverse consistency, constraint weights and 500-message limits are unchanged from 2026-09-24-defect-density.md. Four concurrent cases, 32 threads each. The 32-thread control exactly reproduces 128-thread factor bytes, initial classes/windings, removal partition and ladder. No new inference or tracing.

## Best results

| Statistic | Scale | Floor | Removal score (%) ↑ | Gain vs control (pp) | Independent correct (%) | Removed pieces (%) |
|---|---:|---:|---:|---:|---:|---:|
| control | 1 | 1 | -110.6766 | 0.0000 | 38.4615 | 15.9647 |
| mean | 0.088913971 | 0.5 | -109.6787 | 0.9979 | 38.4615 | 15.7290 |
| median | 0.088913971 | 0.5 | -109.1046 | 1.5720 | 38.4615 | 15.7143 |
| max | 0.88913971 | 0.63245553 | -108.6632 | 2.0134 | 38.4615 | 15.7143 |

Removal score is `-100 * problematic / retained_fulfilled`, closer to zero is better. It is not the percentage of constraints removed. All conditioned readouts are 100%; independent readout is from frozen initial BP and remains 38.46% for the selected winners.

## Limitations and validation

- 52 successful runs, 49 distinct effective multiplier vectors (some parameter changes are clamped and redundant). The coarse scale5/floors0.01 and0.1 duplicates were not skipped in the already-running orchestrator.
- Initial BP statuses: {'message_limit': 52}; conditioned statuses: {'converged': 14, 'message_limit': 38}. Finite-budget results, not converged-optimum claims.
- All factor CSVs share SHA256 `61155ab3d25ebbb10e18372cc0a04907503178253b93fd0594371aac64411b43`; initial H/V labels are identical; all runs have exactly26 calibrated rank-one reference assignments and no missing references.
- Five focused orchestration tests pass. Native code is unchanged since the preceding 104-test validation.
- Tuned on this crop, not held-out. Global mean penalty also changes; a matched uniform-penalty control is still needed to isolate the value of density targeting. Defaults unchanged.
- Resume checks were tightened during this run to include trace payload hashes and ignore report-only changes. Initial run records predate that stricter input binding and deliberately fail its reuse check; do not silently treat those as fully content-bound resume records.
- Total wall time 76.01 min. Per-case wall times: min/median/max 260.27/318.00/394.26 s (concurrent runs; not a speed comparison).

## Reproduce

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_density_sweep_20260924.json
/home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/neural_tracing/test_density_penalty_sweep.py
```

All generated inputs, configs, logs, CSVs, OBJ removals and run records: `/ephemeral/hendrik/fiber_crop_bench/20260924_density_sweep`. Raw summarized results and complete ladders: `2026-09-24-density-sweep-results.json` beside this report.

## Full parameter table

| Statistic | Scale | Floor | Mean multiplier | Removal score (%) ↑ | Independent (%) | Removed pieces (%) | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 1 | 1 | 1.000000 | -110.6766 | 38.4615 | 15.9647 | 264.15 |
| max | 0.05 | 0.01 | 0.136167 | -120.2605 | 46.1538 | 17.2754 | 334.23 |
| max | 0.05 | 0.1 | 0.162174 | -116.1217 | 38.4615 | 16.8189 | 345.22 |
| max | 0.05 | 0.5 | 0.500165 | -111.3065 | 38.4615 | 16.1267 | 320.07 |
| max | 0.15811388 | 0.5 | 0.519670 | -110.7455 | 30.7692 | 16.1267 | 311.60 |
| max | 0.28117066 | 0.63245553 | 0.650888 | -113.2134 | 34.6154 | 16.7452 | 338.26 |
| max | 0.5 | 0.01 | 0.520668 | -116.8721 | 38.4615 | 16.7158 | 310.66 |
| max | 0.5 | 0.1 | 0.521291 | -116.8721 | 38.4615 | 16.7158 | 265.04 |
| max | 0.5 | 0.2236068 | 0.525165 | -115.8987 | 38.4615 | 16.5538 | 291.93 |
| max | 0.5 | 0.5 | 0.603428 | -111.2733 | 38.4615 | 15.9647 | 309.16 |
| max | 0.5 | 0.56234133 | 0.639274 | -109.0151 | 38.4615 | 15.6848 | 273.06 |
| max | 0.5 | 0.63245553 | 0.682664 | -109.2322 | 34.6154 | 15.6259 | 260.27 |
| max | 0.5 | 0.71131176 | 0.735877 | -110.6023 | 34.6154 | 16.0972 | 305.56 |
| max | 0.88913971 | 0.63245553 | 0.718506 | -108.6632 | 38.4615 | 15.7143 | 276.57 |
| max | 1.5811388 | 0.5 | 0.749837 | -109.5451 | 38.4615 | 15.7143 | 272.60 |
| max | 5 | 0.01 | 0.887598 | -113.5892 | 34.6154 | 16.7747 | 335.88 |
| max | 5 | 0.1 | 0.887598 | -113.5892 | 34.6154 | 16.7747 | 318.55 |
| max | 5 | 0.5 | 0.889432 | -112.9232 | 34.6154 | 16.6274 | 340.87 |
| mean | 0.015811388 | 0.5 | 0.500072 | -111.3065 | 38.4615 | 16.1267 | 323.06 |
| mean | 0.028117066 | 0.5 | 0.501725 | -111.4257 | 38.4615 | 16.1708 | 354.72 |
| mean | 0.05 | 0.01 | 0.250148 | -125.0629 | 38.4615 | 18.3211 | 314.51 |
| mean | 0.05 | 0.1 | 0.262605 | -121.5750 | 34.6154 | 17.7320 | 387.23 |
| mean | 0.05 | 0.2236068 | 0.324122 | -117.0357 | 38.4615 | 16.9514 | 358.35 |
| mean | 0.05 | 0.33437015 | 0.392833 | -112.7854 | 38.4615 | 16.0236 | 367.36 |
| mean | 0.05 | 0.5 | 0.515402 | -110.8944 | 34.6154 | 16.0088 | 296.73 |
| mean | 0.05 | 0.56234133 | 0.569111 | -111.3952 | 34.6154 | 16.2003 | 324.42 |
| mean | 0.05 | 0.63245553 | 0.634254 | -111.4424 | 38.4615 | 16.1708 | 314.70 |
| mean | 0.088913971 | 0.5 | 0.548254 | -109.6787 | 38.4615 | 15.7290 | 293.44 |
| mean | 0.15811388 | 0.5 | 0.591232 | -111.7675 | 38.4615 | 16.0383 | 394.26 |
| mean | 0.5 | 0.01 | 0.653501 | -113.7082 | 38.4615 | 16.2445 | 309.60 |
| mean | 0.5 | 0.1 | 0.653598 | -113.7082 | 38.4615 | 16.2445 | 316.63 |
| mean | 0.5 | 0.5 | 0.688365 | -112.3787 | 38.4615 | 16.3623 | 345.52 |
| mean | 5 | 0.01 | 0.932067 | -112.1523 | 34.6154 | 16.3181 | 303.60 |
| mean | 5 | 0.1 | 0.932067 | -112.1523 | 34.6154 | 16.3181 | 318.80 |
| mean | 5 | 0.5 | 0.932354 | -112.2305 | 34.6154 | 16.3181 | 317.82 |
| median | 0.015811388 | 0.5 | 0.500313 | -111.4202 | 38.4615 | 16.1856 | 360.35 |
| median | 0.028117066 | 0.5 | 0.504130 | -111.9067 | 34.6154 | 16.2592 | 303.46 |
| median | 0.05 | 0.01 | 0.274633 | -124.6612 | 38.4615 | 18.4094 | 334.12 |
| median | 0.05 | 0.1 | 0.286766 | -122.4775 | 38.4615 | 17.8498 | 359.02 |
| median | 0.05 | 0.2236068 | 0.344314 | -116.9364 | 38.4615 | 17.1429 | 359.46 |
| median | 0.05 | 0.33437015 | 0.408647 | -124.6017 | 34.6154 | 17.4374 | 364.71 |
| median | 0.05 | 0.5 | 0.523764 | -111.2096 | 34.6154 | 16.1267 | 301.87 |
| median | 0.05 | 0.56234133 | 0.574296 | -110.8696 | 34.6154 | 16.0825 | 318.17 |
| median | 0.05 | 0.63245553 | 0.636586 | -111.8789 | 34.6154 | 16.3181 | 288.20 |
| median | 0.088913971 | 0.5 | 0.561553 | -109.1046 | 38.4615 | 15.7143 | 299.81 |
| median | 0.15811388 | 0.5 | 0.607415 | -123.2388 | 38.4615 | 17.2607 | 360.18 |
| median | 0.5 | 0.01 | 0.670503 | -113.3489 | 38.4615 | 16.1119 | 315.26 |
| median | 0.5 | 0.1 | 0.670605 | -113.3489 | 38.4615 | 16.1119 | 340.87 |
| median | 0.5 | 0.5 | 0.704408 | -112.8360 | 38.4615 | 16.4065 | 336.48 |
| median | 5 | 0.01 | 0.935754 | -111.9764 | 34.6154 | 16.2003 | 304.18 |
| median | 5 | 0.1 | 0.935754 | -111.9764 | 34.6154 | 16.2003 | 307.32 |
| median | 5 | 0.5 | 0.936072 | -111.9764 | 34.6154 | 16.2003 | 318.83 |
