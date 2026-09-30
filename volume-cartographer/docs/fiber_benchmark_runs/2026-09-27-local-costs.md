# Local-event piece-disable and continuation-cut cost trials

> Historical broken-graph experiment: post-split support filtering discarded certified constraints. These results do not evaluate the intended fine-splitting algorithm. Superseded by the certified-splits correction and rerun; retained for provenance.

Recorded trials: 32; requested batch complete: True.

Same frozen Paris4 2048 beam8 / 40×80 / 45° / reverse-consistent traces and 26 reference fibers as the local-event benchmark. No inference or retracing. Native RelWithDebInfo; one run per setting. Primary matrix: four concurrent × 48 threads; supplemental fixed-H/V matrix: two concurrent × 48 threads, overlapping the primary tail. Timings include contention and are not isolated speed benchmarks. Normal alignment and winding limits both 500.

The primary trials jointly solve H/V and winding; supplementary fixed-H/V trials isolate the new connection choice under the previous orientation policy. Compare finite cuts against the hard row at the same piece cost, orientation mode and parallel setting.

Legacy segmentation has different constraint/continuity counts, so its removal score is context rather than a directly matched cost comparison. JSON records whether rerun fixed-H/V hard controls exactly reproduce the historical local scores and ladders.

Piece cost D is the existing Defect unary per incident evidence mass, not an absolute per-piece penalty. Connection cost C is an absolute cost per active-active continuation cut. Keeping a connection requires equal orientation and integer winding. A cut preserves both pieces and ordinary constraints. Defect endpoints incur no extra cut charge. This is a minimized binary/Potts choice, not a marginalized latent switch. Production defaults unchanged.

Removal score = −100 × problematic / fulfilled constraints (closer to zero is better). Cuts count as problematic, not free successes. Initial accuracy is independent reference readout from the unconditioned solve. Conditioned accuracy is the established removal/oracle metric and is not a deployment result. Defects and cuts below refer to the initial solve; cut percentage uses active-active continuation links as denominator.

| Variant | Piece cost | Cut cost | Removal score | Initial accuracy | Conditioned accuracy | Defect pieces | Cut links | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous legacy (fixed H/V) | 100 | hard | -110.6766% | 38.4615% | 100.0000% | 15.5965% | — | 263.4 |
| Previous local-parallel (fixed H/V) | 100 | hard | -90.5016% | 15.3846% | 57.6923% | 3.8167% | — | 453.7 |
| Previous local-orthogonal (fixed H/V) | 100 | hard | -74.7873% | 11.5385% | 34.6154% | 3.7549% | — | 318.2 |
| Orthogonal only / joint H/V | 1 | 0.1 | -1153.9539% | 3.8462% | 19.2308% | 0.0000% | 26.6081% | 131.6 |
| Orthogonal only / joint H/V | 1 | 1 | -2193.9256% | 0.0000% | 19.2308% | 14.8794% | 31.0566% | 225.2 |
| Orthogonal only / joint H/V | 1 | 10 | -1604.8800% | 11.5385% | 34.6154% | 20.4579% | 24.9898% | 246.9 |
| Orthogonal only / joint H/V | 1 | hard | -1736.1720% | 0.0000% | 34.6154% | 34.3398% | 0.0000% | 254.8 |
| Orthogonal only / joint H/V | 10 | 0.1 | -971.1756% | 3.8462% | 23.0769% | 0.0127% | 42.2378% | 150.2 |
| Orthogonal only / joint H/V | 10 | 1 | -48.8618% | 3.8462% | 30.7692% | 1.3942% | 29.9840% | 584.9 |
| Orthogonal only / joint H/V | 10 | 10 | -20.6638% | 7.6923% | 42.3077% | 0.4015% | 1.1396% | 446.8 |
| Orthogonal only / joint H/V | 10 | hard | -21.8294% | 7.6923% | 42.3077% | 1.5472% | 0.0000% | 351.9 |
| Orthogonal only / joint H/V | 100 | 0.1 | -999.8485% | 0.0000% | 38.4615% | 0.0127% | 36.5105% | 166.4 |
| Orthogonal only / joint H/V | 100 | 1 | -38.3780% | 3.8462% | 26.9231% | 1.0262% | 30.8191% | 602.5 |
| Orthogonal only / joint H/V | 100 | 10 | -24.9398% | 11.5385% | 42.3077% | 0.5450% | 12.6711% | 1403.1 |
| Orthogonal only / joint H/V | 100 | hard | -23.8103% | 11.5385% | 26.9231% | 13.5313% | 0.0000% | 840.2 |
| Parallel on / joint H/V | 1 | 0.1 | -1079.7566% | 7.6923% | 46.1538% | 0.0016% | 31.3673% | 183.3 |
| Parallel on / joint H/V | 1 | 1 | -2215.9874% | 11.5385% | 26.9231% | 15.0377% | 28.9545% | 238.4 |
| Parallel on / joint H/V | 1 | 10 | -1627.7326% | 11.5385% | 34.6154% | 23.3701% | 25.8051% | 283.4 |
| Parallel on / joint H/V | 1 | hard | -1876.9946% | 15.3846% | 15.3846% | 36.7935% | 0.0000% | 236.5 |
| Parallel on / joint H/V | 10 | 0.1 | -1056.9774% | 15.3846% | 19.2308% | 0.0191% | 38.4219% | 175.2 |
| Parallel on / joint H/V | 10 | 1 | -41.8765% | 11.5385% | 34.6154% | 0.0461% | 32.5491% | 597.5 |
| Parallel on / joint H/V | 10 | 10 | -22.7842% | 15.3846% | 30.7692% | 0.3829% | 1.1592% | 386.9 |
| Parallel on / joint H/V | 10 | hard | -20.4176% | 15.3846% | 15.3846% | 1.4396% | 0.0000% | 380.0 |
| Parallel on / joint H/V | 100 | 0.1 | -1106.9767% | 11.5385% | 30.7692% | 0.0191% | 39.0760% | 156.8 |
| Parallel on / joint H/V | 100 | 1 | -48.7945% | 3.8462% | 46.1538% | 0.0254% | 33.3846% | 1014.3 |
| Parallel on / joint H/V | 100 | 10 | -24.0902% | 7.6923% | 50.0000% | 0.1335% | 12.0926% | 1439.4 |
| Parallel on / joint H/V | 100 | hard | -25.9059% | 7.6923% | 46.1538% | 12.8164% | 0.0000% | 462.1 |
| Orthogonal only / fixed H/V | 100 | 0.1 | -525.8985% | 11.5385% | 30.7692% | 2.4489% | 12.2919% | 126.7 |
| Orthogonal only / fixed H/V | 100 | 1 | -69.2946% | 11.5385% | 30.7692% | 2.4489% | 1.8340% | 300.0 |
| Orthogonal only / fixed H/V | 100 | 10 | -54.8084% | 15.3846% | 19.2308% | 2.4569% | 1.7427% | 341.7 |
| Orthogonal only / fixed H/V | 100 | hard | -74.7873% | 11.5385% | 34.6154% | 3.7549% | 0.0000% | 329.0 |
| Parallel on / fixed H/V | 100 | 0.1 | -530.3356% | 11.5385% | 34.6154% | 2.5280% | 12.5162% | 152.7 |
| Parallel on / fixed H/V | 100 | 1 | -60.6726% | 15.3846% | 26.9231% | 2.5328% | 1.8168% | 357.2 |
| Parallel on / fixed H/V | 100 | 10 | -64.7967% | 15.3846% | 50.0000% | 2.5360% | 1.6236% | 469.0 |
| Parallel on / fixed H/V | 100 | hard | -90.5016% | 15.3846% | 57.6923% | 3.8167% | 0.0000% | 517.5 |

## Reproduce

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_local_costs_20260927.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_local_costs_fixed_20260927.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_local_costs_report_20260927.json
```

Artifacts: `/ephemeral/hendrik/fiber_crop_bench/20260927_local_costs_v1`. Every trial has its exact command, input/reference hashes, binary/library hashes, patch, CSVs and log. Resume requires matching provenance. JSON includes ladders when the single-gauge readout is valid, convergence statuses and distinct-event audits; otherwise an explicit metric warning replaces initial accuracy. SVG shows removal score and initial accuracy.

## Audit and limitations

This is loopy sum-product BP followed by marginal-mode decoding and hard-sign projection, not an exact joint MAP optimizer. Cost changes need not produce monotonic changes in the decoded solution. A continuation cut is minimized conditional on those decoded endpoint states; the full piece/cut assignment is not globally optimized. See per-run convergence statuses below.

- orthogonal-d1-c0p1: statuses ['converged']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d1-c1: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d1-c10: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d1-chard: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d10-c0p1: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d10-c1: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d10-c10: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d10-chard: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d100-c0p1: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d100-c1: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d100-c10: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d100-chard: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- parallel-d1-c0p1: statuses ['converged']; event audit {'pieces_with_orthogonal_events': 8044, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- parallel-d1-c1: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8044, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- parallel-d1-c10: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8044, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- parallel-d1-chard: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8044, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- parallel-d10-c0p1: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8044, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- parallel-d10-c1: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8044, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- parallel-d10-c10: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8044, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- parallel-d10-chard: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8044, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- parallel-d100-c0p1: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8044, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- parallel-d100-c1: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8044, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- parallel-d100-c10: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8044, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- parallel-d100-chard: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8044, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d100-c0p1-fixed: statuses ['converged']; event audit {'pieces_with_orthogonal_events': 7873, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 513, 'maximum_orthogonal_factors': 5}.
- orthogonal-d100-c1-fixed: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 7873, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 513, 'maximum_orthogonal_factors': 5}.
- orthogonal-d100-c10-fixed: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 7873, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 513, 'maximum_orthogonal_factors': 5}.
- orthogonal-d100-chard-fixed: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 7873, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 513, 'maximum_orthogonal_factors': 5}.
- parallel-d100-c0p1-fixed: statuses ['converged']; event audit {'pieces_with_orthogonal_events': 7865, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 505, 'maximum_orthogonal_factors': 5}.
- parallel-d100-c1-fixed: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 7865, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 505, 'maximum_orthogonal_factors': 5}.
- parallel-d100-c10-fixed: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 7865, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 505, 'maximum_orthogonal_factors': 5}.
- parallel-d100-chard-fixed: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 7865, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 505, 'maximum_orthogonal_factors': 5}.
