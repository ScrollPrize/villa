# Distance-weighted continuation experiment

> Historical broken-graph experiment: post-split support filtering discarded certified constraints. These results do not evaluate the intended fine-splitting algorithm. Superseded by the certified-splits correction and rerun; retained for provenance.

Joint H/V, parallel constraints off, D=10. Same frozen Paris4 2048 traces and 26 references; 500-iteration budgets. Four concurrent runs × 48 threads, one repetition, RelWithDebInfo. No inference or retracing. All six settings, including controls, use the same new binary.

Distance rows use C*abs(q_b-q_a), where q includes calibrated H/V phase. This is an untruncated soft pair potential, not a removable connection or a flat cut. Ordinary evidence remains unchanged. Defect endpoints pay their usual piece unary and no continuation charge. The existing cut diagnostic counts unequal active endpoints; in distance mode these remain coupled during BP. Reference-conditioned inlier selection still treats unequal continuity as problematic.

Initial accuracy measures independent reference readout of the unconditioned solve; conditioned accuracy uses reference-based removal. Removal is negative problematic/fulfilled percentage, nearer zero better, not winding correctness. Production defaults unchanged.

| Variant | Piece cost | Cost | Removal score | Initial accuracy | Conditioned accuracy | Defect pieces | Unequal links | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous legacy (fixed H/V) | 100 | hard | -110.6766% | 38.4615% | 100.0000% | 15.5965% | — | 263.4 |
| Previous local-parallel (fixed H/V) | 100 | hard | -90.5016% | 15.3846% | 57.6923% | 3.8167% | — | 453.7 |
| Previous local-orthogonal (fixed H/V) | 100 | hard | -74.7873% | 11.5385% | 34.6154% | 3.7549% | — | 318.2 |
| Orthogonal only / joint H/V / linear distance | 10 | 0.1 | -986.1347% | 7.6923% | 34.6154% | 0.0000% | 19.3315% | 128.1 |
| Orthogonal only / joint H/V / linear distance | 10 | 1 | -268.3110% | 11.5385% | 42.3077% | 0.0255% | 6.7404% | 472.5 |
| Orthogonal only / joint H/V | 10 | 10 | -20.6638% | 7.6923% | 42.3077% | 0.4015% | 1.1396% | 395.5 |
| Orthogonal only / joint H/V / linear distance | 10 | 10 | -22.0296% | 11.5385% | 34.6154% | 0.3745% | 1.0797% | 914.7 |
| Orthogonal only / joint H/V / linear distance | 10 | 100 | -20.6727% | 7.6923% | 42.3077% | 0.4095% | 1.2058% | 303.7 |
| Orthogonal only / joint H/V | 10 | hard | -21.8294% | 7.6923% | 42.3077% | 1.5472% | 0.0000% | 327.7 |

## Unconditioned ladder shifts

Count adjacent changes in predicted-minus-reference offset, ignoring a global offset. Shift score = 100*(1-K/N); one suffix shift in 26 references gives 96.15%. Shifts have arbitrary magnitude, not unit-winding cost.

| Trial | Shifts | Shift score |
|---|---:|---:|
| orthogonal-d10-c0p1-distance | 25 | 3.8462% |
| orthogonal-d10-c1-distance | 25 | 3.8462% |
| orthogonal-d10-c10 | 22 | 15.3846% |
| orthogonal-d10-c10-distance | 24 | 7.6923% |
| orthogonal-d10-c100-distance | 23 | 11.5385% |
| orthogonal-d10-chard | 23 | 11.5385% |

## Reproduce

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_local_step_costs_20260927.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_local_step_costs_report_20260927.json
```

Artifacts: `/ephemeral/hendrik/fiber_crop_bench/20260927_local_step_costs_v1`. Every trial has its exact command, input/reference hashes, binary/library hashes, patch, CSVs and log. Resume requires matching provenance. JSON includes ladders when the single-gauge readout is valid, convergence statuses and distinct-event audits; otherwise an explicit metric warning replaces initial accuracy. SVG shows removal score and initial accuracy.

## Audit and limitations

This is loopy sum-product BP followed by marginal-mode decoding and hard-sign projection, not an exact joint MAP optimizer. Cost changes need not produce monotonic changes in the decoded solution. Flat-cut choices are minimized conditional on endpoint states; distance costs retain a soft coupling. Neither full assignment is globally optimized. See per-run convergence statuses below.

- orthogonal-d10-c0p1-distance: statuses ['converged']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d10-c1-distance: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d10-c10: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d10-c10-distance: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d10-c100-distance: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
- orthogonal-d10-chard: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 8042, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 521, 'maximum_orthogonal_factors': 5}.
