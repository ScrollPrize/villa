# Certified constraints preserved across splitting

Correction: certified measurements, signs, support and source-pair-normalized evidence weights survive splitting unchanged. Only owning piece IDs change, plus continuation links and explicitly requested parallel-factor omission. Previous local/cost experiments used an erroneous post-split support filter; they are historical broken-graph diagnostics, not tests of the intended algorithm.

Same frozen Paris4 2048 beam8/40x80/45-degree/reverse-consistent traces, 26 references and 500-message budget. Message-limit termination is accepted completion. Four simultaneous processes x48 threads, RelWithDebInfo, one run per setting; timings are contended. No inference or retracing.

Compare legacy grouped baseline and fixed-H/V local hard controls at D100; then joint-H/V parallel-off D10 hard, flat-C10 and linear-distance10. Initial = unconditioned reference readout; conditioned = reference-assisted benchmark. Removal = negative problematic/fulfilled %, nearer zero better, not correctness.

| Variant | Piece cost | Cost | Removal score | Initial accuracy | Conditioned accuracy | Defect pieces | Unequal links | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous legacy (fixed H/V) | 100 | hard | -110.6766% | 38.4615% | 100.0000% | 15.5965% | — | 263.4 |
| Previous local-parallel (fixed H/V) | 100 | hard | -90.5016% | 15.3846% | 57.6923% | 3.8167% | — | 453.7 |
| Previous local-orthogonal (fixed H/V) | 100 | hard | -74.7873% | 11.5385% | 34.6154% | 3.7549% | — | 318.2 |
| Orthogonal only / joint H/V | 10 | 10 | -49.2508% | 11.5385% | 30.7692% | 5.5963% | 1.1885% | 796.8 |
| Orthogonal only / joint H/V / linear distance | 10 | 10 | -47.7036% | 11.5385% | 34.6154% | 5.4775% | 1.3836% | 713.4 |
| Orthogonal only / joint H/V | 10 | hard | -54.0555% | 11.5385% | 26.9231% | 6.6234% | 0.0000% | 779.6 |
| Orthogonal only / fixed H/V | 100 | hard | -50.3930% | 7.6923% | 30.7692% | 6.6205% | 0.0000% | 664.9 |
| Parallel on / fixed H/V | 100 | hard | -75.8570% | 19.2308% | 30.7692% | 9.9469% | 0.0000% | 1075.4 |
| Parallel on / fixed H/V / legacy grouped | 100 | hard | -110.6766% | 38.4615% | 100.0000% | 15.5965% | 0.0000% | 332.0 |

## Unconditioned ladder shifts

Count adjacent changes in predicted-minus-reference offset, ignoring a global offset. Shift score = 100*(1-K/N); one suffix shift in 26 references gives 96.15%. Shifts have arbitrary magnitude, not unit-winding cost.

| Trial | Shifts | Shift score |
|---|---:|---:|
| orthogonal-d10-c10 | 20 | 23.0769% |
| orthogonal-d10-c10-distance | 22 | 15.3846% |
| orthogonal-d10-chard | 19 | 26.9231% |
| orthogonal-d100-chard-fixed | 14 | 46.1538% |
| parallel-d100-chard-fixed | 14 | 46.1538% |
| parallel-d100-chard-fixed-legacy | 7 | 73.0769% |

## Certified-event retention audit

| Trial | Certified | Retained | Parallel omitted | Support rejected | Reference certified | Reference retained | Missing owners |
|---|---:|---:|---:|---:|---:|---:|---:|
| orthogonal-d10-c10 | 53480 | 35528 | 17952 | 0 | 2913 | 2913 | 0 |
| orthogonal-d10-c10-distance | 53480 | 35528 | 17952 | 0 | 2913 | 2913 | 0 |
| orthogonal-d10-chard | 53480 | 35528 | 17952 | 0 | 2913 | 2913 | 0 |
| orthogonal-d100-chard-fixed | 53480 | 35528 | 17952 | 0 | 2913 | 2913 | 0 |
| parallel-d100-chard-fixed | 53480 | 53480 | 0 | 0 | 5398 | 5398 | 0 |

## Unconditioned reference ladders

| Reference | orthogonal-d10-c10 | orthogonal-d10-c10-distance | orthogonal-d10-chard | orthogonal-d100-chard-fixed | parallel-d100-chard-fixed | parallel-d100-chard-fixed-legacy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 2.5 | 7 | 6 | 8 | -0 |
| 0.5 | 6 | 5.5 | 5 | 4.5 | 0.5 | 0.5 |
| 1 | 3 | 7.5 | 2 | 4 | 4 | 1 |
| 1.5 | 3 | 2.5 | 2 | 4.5 | 8.5 | 1.5 |
| 2 | 3 | 7.5 | 2 | 4 | 4 | 2 |
| 2.5 | 3 | 7.5 | 2 | 2 | 3.5 | 1.5 |
| 3 | 3 | 7.5 | 2 | 1.5 | 3 | 2 |
| 3.5 | 3 | 7 | 2 | 2 | 3.5 | 2.5 |
| 4 | 7 | 2.5 | 6 | 4 | 4 | 4 |
| 4.5 | 5.5 | 6.5 | 4.5 | 4.5 | 4.5 | 4.5 |
| 5 | 6 | 6 | 5 | 4 | 4 | 5 |
| 5.5 | 5.5 | 6.5 | 4.5 | 4.5 | 4.5 | 5.5 |
| 6 | 6 | 6 | 5 | 4 | 5 | 6 |
| 6.5 | 5.5 | 6.5 | 4.5 | 4.5 | 4.5 | 5.5 |
| 7 | 3 | 6 | 2 | 4 | 5 | 6 |
| 7.5 | 5.5 | 7.5 | 4.5 | 4.5 | 4.5 | 6.5 |
| 8 | 6 | 6 | 5 | 5 | 5 | 6 |
| 8.5 | 6.5 | 5.5 | 5.5 | 5.5 | 5.5 | 6.5 |
| 9 | 8 | 2.5 | 7 | 6 | 8 | 7 |
| 9.5 | 6.5 | 5.5 | 5.5 | 5.5 | 12.5 | 6.5 |
| 10 | 7 | 5 | 6 | 6 | 6 | 7 |
| 10.5 | 3 | 7.5 | 2 | 2 | 6.5 | 7.5 |
| 11 | 3 | 7 | 4.5 | 4 | 3 | 8 |
| 11.5 | 6 | 6 | 5 | 5.5 | 3.5 | 9.5 |
| 12 | 5.5 | 6.5 | 4.5 | 4 | 4 | 10 |
| 12.5 | 3 | 6.5 | 2 | 4.5 | 3.5 | 9.5 |

## Reproduce

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_certified_splits_20260927.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_certified_splits_report_20260927.json
```

Artifacts: `/ephemeral/hendrik/fiber_crop_bench/20260927_certified_splits_v3`. Every trial has its exact command, input/reference hashes, binary/library hashes, patch, CSVs and log. Resume requires matching provenance. JSON includes ladders when the single-gauge readout is valid, convergence statuses and distinct-event audits; otherwise an explicit metric warning replaces initial accuracy. SVG shows removal score and initial accuracy.

## Audit and limitations

This is loopy sum-product BP followed by marginal-mode decoding and hard-sign projection, not an exact joint MAP optimizer. Cost changes need not produce monotonic changes in the decoded solution. Flat-cut choices are minimized conditional on endpoint states; distance costs retain a soft coupling. Neither full assignment is globally optimized. See per-run convergence statuses below.

- orthogonal-d10-c10: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 60785, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 6965, 'maximum_orthogonal_factors': 10}; certified extraction [{'pieces': 68851, 'constraints': 101601, 'continuity': 66073, 'consistency_rejected': 521744, 'support_rejected': 0, 'certified': 53480, 'retained_certified': 35528, 'refined': 36377, 'midpoint_cuts': 58700, 'transition_cuts': 7838, 'parallel_omitted': 17952}]; reference source audit [{'certified': 2913, 'retained': 2913, 'missing_owner': 0}].
- orthogonal-d10-c10-distance: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 60785, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 6965, 'maximum_orthogonal_factors': 10}; certified extraction [{'pieces': 68851, 'constraints': 101601, 'continuity': 66073, 'consistency_rejected': 521744, 'support_rejected': 0, 'certified': 53480, 'retained_certified': 35528, 'refined': 36377, 'midpoint_cuts': 58700, 'transition_cuts': 7838, 'parallel_omitted': 17952}]; reference source audit [{'certified': 2913, 'retained': 2913, 'missing_owner': 0}].
- orthogonal-d10-chard: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 60785, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 6965, 'maximum_orthogonal_factors': 10}; certified extraction [{'pieces': 68851, 'constraints': 101601, 'continuity': 66073, 'consistency_rejected': 521744, 'support_rejected': 0, 'certified': 53480, 'retained_certified': 35528, 'refined': 36377, 'midpoint_cuts': 58700, 'transition_cuts': 7838, 'parallel_omitted': 17952}]; reference source audit [{'certified': 2913, 'retained': 2913, 'missing_owner': 0}].
- orthogonal-d100-chard-fixed: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 60785, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 6965, 'maximum_orthogonal_factors': 10}; certified extraction [{'pieces': 68851, 'constraints': 101601, 'continuity': 66073, 'consistency_rejected': 521744, 'support_rejected': 0, 'certified': 53480, 'retained_certified': 35528, 'refined': 36377, 'midpoint_cuts': 58700, 'transition_cuts': 7838, 'parallel_omitted': 17952}]; reference source audit [{'certified': 2913, 'retained': 2913, 'missing_owner': 0}].
- parallel-d100-chard-fixed: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 60756, 'multiple_event_pieces': 0, 'maximum_event_groups': 1, 'multiple_orthogonal_factor_pieces': 6962, 'maximum_orthogonal_factors': 10}; certified extraction [{'pieces': 68851, 'constraints': 119553, 'continuity': 66073, 'consistency_rejected': 521744, 'support_rejected': 0, 'certified': 53480, 'retained_certified': 53480, 'refined': 36377, 'midpoint_cuts': 58700, 'transition_cuts': 7838, 'parallel_omitted': 0}]; reference source audit [{'certified': 5398, 'retained': 5398, 'missing_owner': 0}].
- parallel-d100-chard-fixed-legacy: statuses ['message_limit']; event audit {'pieces_with_orthogonal_events': 0, 'multiple_event_pieces': 0, 'maximum_event_groups': 0, 'multiple_orthogonal_factor_pieces': 0, 'maximum_orthogonal_factors': 0}; certified extraction []; reference source audit [].
