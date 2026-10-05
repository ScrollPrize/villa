# Fast historical1024 independent BP metrics

Only u22000 beam8 and Qual. Cached traces; no inference, retracing or oracle conditioning. Historical command/objective flags retained; only pruning/oracle flags removed and independent readout enabled. Current native binary,128CPU threads, RelWithDebInfo, one run per new result. Historical hard-perpendicular independent log reused without a solve. Input/reference hashes checked; trace payload hashes recorded and unchanged during each run. Initial state equality is audited below.

| Variant | Non-oracle accuracy | Correct/26 | Missing | Observed shift score | Shifts / observed refs | Tied / unbounded | Initial states identical | New CPU wall s |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| u22000 original BP | 57.6923% | 15/26 | 2 | 70.8333% | 7 / 24 | 0 / 0 | False | 18.08 |
| u22000 hard perpendicular | 34.6154% | 9/26 | 2 | 70.8333% | 7 / 24 | 0 / 0 | True | 0.00 |
| u22000 hard both signs | 30.7692% | 8/26 | 2 | 75.0000% | 6 / 24 | 0 / 0 | True | 14.50 |
| Qual supplied fibers | 69.2308% | 18/26 | 2 | 66.6667% | 8 / 24 | 0 / 0 | True | 2.83 |

Accuracy uses all26 references. Shift score100*(1-K/Nobserved) omits missing references and bridges their gaps; full26shift is unavailable for incomplete ladders. Global sign/offset alignment uses GT for scoring only. Tied ranks, if any, are diagnostics, not uniquely determined assignments.

## Ladders

Original-BP caveat:4/1204initial piece states differ from the old record:pieces622and1053 shift by+1winding;713and1054 change from activeH to defect. Same recorded objective and cached inputs, but a current native binary, so this is not an exact historical reconstruction. No cause isolated by this task. Other fresh runs reproduce their historical states exactly; hard-perpendicular simply reuses the saved independent log.

| Reference | u22000 original BP | u22000 hard perpendicular | u22000 hard both signs | Qual supplied fibers |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 1 | 0 |
| 0.5 | 0.5 | 0.5 | 1.5 | 0.5 |
| 1 | 1 | 1 | 2 | 1 |
| 1.5 | 1.5 | 1.5 | 2.5 | 1.5 |
| 2 | 2 | 2 | 3 | 2 |
| 2.5 | 2.5 | 2.5 | 2.5 | 2.5 |
| 3 | 3 | 3 | 3 | 3 |
| 3.5 | 3.5 | 2.5 | 3.5 | 3.5 |
| 4 | missing | missing | missing | missing |
| 4.5 | 4.5 | 4.5 | 4.5 | 4.5 |
| 5 | 5 | 5 | 5 | 5 |
| 5.5 | 5.5 | 4.5 | 5.5 | 5.5 |
| 6 | 6 | 5 | 5 | 6 |
| 6.5 | 6.5 | 5.5 | 5.5 | 6.5 |
| 7 | 6 | 5 | 6 | 6 |
| 7.5 | 6.5 | 5.5 | 5.5 | 6.5 |
| 8 | 7.5 | 6 | 6 | 6 |
| 8.5 | 8.5 | 6.5 | 6.5 | 7.5 |
| 9 | 9 | 7 | 7 | 9 |
| 9.5 | 10.5 | 6.5 | 7.5 | 8.5 |
| 10 | 12 | 7 | 9 | 10 |
| 10.5 | 12.5 | 7.5 | 9.5 | 10 |
| 11 | 15 | 9 | 11 | 11 |
| 11.5 | 15.5 | 9.5 | 11.5 | 11.5 |
| 12 | missing | missing | missing | missing |
| 12.5 | 15.5 | 9.5 | 11.5 | 12.5 |

Run: control action `historical_bp_independent`, then `bash /home/hendrik/villa3/scripts/run_training_failure_mining`.
Artifacts: `/ephemeral/hendrik/fiber_crop_bench/20261002_fast_historical_independent_v1`.
