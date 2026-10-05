# Independent BP accuracy and shift — raw40500 1024³

Same frozen source traces, normal/reference hashes and hard-both-sign BP objective. No reference nodes/conditioning/pruning in the solve. Independent reference ranking follows the frozen solve; global gauge alignment uses GT for scoring only. Initial piece states were checked exactly against the previous run. No inference or retracing.

| Variant | Exact /26 | Accuracy | Missing | Observed-only shifts | Observed-only shift score | CPU seconds |
|---|---:|---:|---:|---:|---:|---:|
| unfiltered | 9/26 | 34.6154% | 2 | 7 / 24 | 70.8333% | 13.77 |
| prefix512 | 11/26 | 42.3077% | 2 | 7 / 24 | 70.8333% | 34.13 |

Full26-reference shift score is unavailable if references lack admitted evidence. Observed-only diagnostic sorts the remaining references, skips missing ones, and uses100×(1−K/Nobserved); it must not be compared as the full26score. Accuracy always retains denominator26. Ties/unbounded candidates and full ladders are saved in the adjacentJSON.

## Ladders

| Reference | Unfiltered | Prefix512 |
|---:|---:|---:|
| 0 | 3 | 0 |
| 0.5 | 3.5 | 0.5 |
| 1 | 4 | 1 |
| 1.5 | 4.5 | 1.5 |
| 2 | 5 | 2 |
| 2.5 | 5 | 2.5 |
| 3 | 5 | 3 |
| 3.5 | 5.5 | 3.5 |
| 4 | missing | missing |
| 4.5 | 6.5 | 4.5 |
| 5 | 7 | 5 |
| 5.5 | 6.5 | 4.5 |
| 6 | 7.5 | 5 |
| 6.5 | 7.5 | 5.5 |
| 7 | 8 | 5 |
| 7.5 | 8 | 5.5 |
| 8 | 8 | 8 |
| 8.5 | 8.5 | 6.5 |
| 9 | 9 | 6 |
| 9.5 | 9.5 | 6.5 |
| 10 | 10 | 7 |
| 10.5 | 10.5 | 7.5 |
| 11 | 11 | 9 |
| 11.5 | 11.5 | 9.5 |
| 12 | missing | missing |
| 12.5 | 12.5 | 9.5 |

Approved command: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, action `reviewed_bp_independent`. Exact configs, raw logs, provenance and timings are beside the previous crop outputs.
