# Median64 cost sweep

Selected baseline:median64/min256, length150 per128,boundary0,weights0/0/0/0.05/0,hard signs both. Frozen raw40500 iso20 1024 crop;500iterations,128native threads,up to4 independent candidates concurrently. Each candidate gets independent readout AND historical oracle-inliers removal. Oracle protocol fixed. No inference, new geometry, repair, or iteration tuning. One run per variant; timings under concurrent scheduling are not speed comparisons.

| Case | Parameter changes | Removal score ↑ | Problematic % | Problematic / fulfilled counts | Oracle round0 accuracy ↑ | Independent ↑ | Shift ↑ |
|---|---|---:|---:|---|---:|---:|---:|
| p0p05 | winding-weights=0,0,0,0.05,0 | -29.50% | 22.78% | 55515 / 188167 | 76.92% | 61.54% | 87.50% |
| p0 | winding-weights=0,0,0,0,0 | -341.32% | 77.34% | 188465 / 55217 | 53.85% | 3.85% | 25.00% |
| p0p005 | winding-weights=0,0,0,0.005,0 | -29.64% | 22.87% | 55719 / 187963 | 80.77% | 50.00% | 83.33% |
| p0p01 | winding-weights=0,0,0,0.01,0 | -29.43% | 22.74% | 55409 / 188273 | 80.77% | 61.54% | 83.33% |
| p0p025 | winding-weights=0,0,0,0.025,0 | -53.47% | 34.84% | 84903 / 158779 | 80.77% | 61.54% | 87.50% |
| p0p1 | winding-weights=0,0,0,0.1,0 | -29.08% | 22.53% | 54894 / 188788 | 80.77% | 50.00% | 87.50% |
| d0 | winding-defect-cost=0 | -7161.08% | 98.62% | 240326 / 3356 | 61.54% | 23.08% | 25.00% |
| d75 | winding-defect-cost=75 | -29.20% | 22.60% | 55078 / 188604 | 80.77% | 50.00% | 83.33% |
| d300 | winding-defect-cost=300 | -31.13% | 23.74% | 57856 / 185826 | 76.92% | 61.54% | 87.50% |
| b100 | winding-gap-boundary-cost=100 | -34.02% | 25.38% | 61853 / 181829 | 80.77% | 50.00% | 83.33% |
| b500 | winding-gap-boundary-cost=500 | -66.95% | 40.10% | 97724 / 145958 | 80.77% | 57.69% | 75.00% |
| perp-near | winding-weights=0.05,0,0,0.05,0 | -29.11% | 22.54% | 54937 / 188745 | 80.77% | 30.77% | 75.00% |
| perp-far | winding-weights=0,0.05,0,0.05,0 | -17.50% | 14.89% | 36288 / 207394 | 80.77% | 30.77% | 37.50% |
| parallel-same | winding-weights=0,0,0.05,0.05,0 | -25.71% | 20.45% | 49831 / 193851 | 80.77% | 34.62% | 83.33% |
| sign22 | winding-hard-sign-angle=off, winding-hard-signs=both, winding-sign-cost=22 | -57.21% | 36.39% | 88675 / 155007 | 80.77% | 53.85% | 75.00% |
| sign88 | winding-hard-sign-angle=off, winding-hard-signs=both, winding-sign-cost=88 | -40.63% | 28.89% | 70404 / 173278 | 80.77% | 61.54% | 87.50% |
| no-parallel-sign | winding-sign-weights=1,0 | -20.00% | 16.67% | 40617 / 203065 | 80.77% | 53.85% | 79.17% |
| no-perp-sign | winding-sign-weights=0,1 | unavailable | — | — | — | N/A | N/A |
| combo1 | winding-sign-weights=1,0, winding-weights=0,0.05,0,0.05,0 | -13.16% | 11.63% | 28337 / 215345 | 80.77% | 46.15% | 79.17% |
| combo2 | winding-defect-cost=75, winding-weights=0,0.05,0,0.05,0 | -17.51% | 14.90% | 36306 / 207376 | 80.77% | 61.54% | 83.33% |

Removal score=−100×problematic/fulfilled; closer to0 is better. Problematic includes incident-to-removed, retained infringed and retained defect; full breakdown/missing refs/initial-state checks in JSON. Historical old overlap:removal−25.2365%,oracle80.7692%,independent50%,shift83.3333%. Replay100% inherited unchanged. Shift uses observed references only; independent accuracy denominator26.

P2 cannot contribute under the retained parallel cutoff1.5; not swept. Hard continuity stays hard; gap-boundary is its tunable active/disabled cost. Finite sign cases relax both hard sign categories. Zero sign weights remove the selected sign term. Calibration/temperature/confidence/oracle cost fixed. No production defaults changed. Current baseline designation applies to the selected fine-split approach, not global dominance.

Artifacts: `/ephemeral/hendrik/fiber_crop_bench/20261003_iso20_median_costs_v1`. Approved command `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, action `iso20_median_costs`; `report_only:true` refreshes summaries. Config/run records validate resumption. At most2 pairwise combinations of individually removal-improving disjoint cost coordinates follow the matrix.

## Unavailable cases

- no-perp-sign: vc_fiber_trace_chunk: One reference piece crosses incompatible BP gauges. No gauge/metric substitution applied.

## Verdict

- Completed18 single-coordinate cases and2 grouped-coordinate combinations.17+2 removal results valid; no-perp-sign is unavailable because reference pieces cross incompatible gauges. Independent/oracle initial states and factors match for every valid pair. Baseline repeats previous median64 initial states and removal counts exactly.10 Python tests passed; no native changes.
- Lower P1=.01 gives only a small removal improvement (−29.4301% vs−29.5030%) while shift falls87.50%→83.33%. P1=0 collapses independent accuracy to3.85% and removal to−341.3170%. Smaller is not monotonically better.
- Best removal:far-perpendicular magnitude.05 plus no parallel sign,−13.1589%,28337 problematic/215345 fulfilled (11.6287% problematic),oracle80.7692%,independent46.1538%,shift79.1667%.
- Better balance:far-perpendicular magnitude.05 plus disabling75 per128,−17.5073%,36306 problematic/207376 fulfilled (14.8989% problematic),oracle80.7692%,independent61.5385%,shift83.3333%. This preserves baseline exact accuracy and substantially improves removal, but loses one observed shift step.
- Keep selected median64/P1=.05/disable150 baseline recorded; new results are Pareto alternatives, not automatic production promotions. Changed costs affect reference evidence and fulfillment as well as graph optimization; these are end-to-end operational removal scores, not proof the underlying measured constraints became more accurate.
- Five-entry magnitude vector treated as one grouped coordinate in bounded combination selection; not an exhaustive Cartesian grid and internal combinations of P1 with other magnitude entries were not swept. Inactive P2, calibration, temperature, confidence, continuity hardness and oracle protocol unchanged. No extra iterations or rejected repair.
