# Stopped finetuning snapshot evaluation — 2026-09-13

Training stopped cleanly at update49,500 /6,336,000 cumulative samples with raw
and both EMA checkpoints complete. It remains stopped. Run:
`finetune_5scroll_s1a_128_sd1_16x8_ema_lr0001_20260912_223743`.
Combined five-scroll JSON+S1a NML, LR0.001; no training/math changes in this task.

Six frozen snapshots were evaluated serially through manager-exclusive admission
with the same50fibers/100directions and source/normal data as prior evaluations.
Every run completed and passed the trace artifact audit. Each snapshot used a
separate prediction cache namespace. No inference results were reused between models.

| Snapshot | Update | Samples | Selection | Dense test loss | Trace score |
|---|---:|---:|---|---:|---:|
| raw | 49500 | 6336000 | latest/current | .468491 | 5.360 |
| EMA99 | 49500 | 6336000 | latest/current | .455648 | 5.665 |
| EMA999 | 49500 | 6336000 | latest/current | .454956 | 5.724 |
| raw | 21600 | 2764800 | best test loss | .378411 | 5.589 |
| EMA99 | 21900 | 2803200 | best test loss | .383839 | 5.324 |
| EMA999 | 23400 | 2995200 | best test loss | .388823 | 5.486 |

Best checkpoints minimize their own raw/EMA dense test loss, not tracing score.
Both latest EMAs trace better than their test-loss-selected best snapshots;
the opposite holds for raw. Dense test loss is therefore not a reliable ranking
of these tracing outcomes. No statistical significance claim:0332 has3fibers,
1451 only2; training budgets also differ across historical comparisons.

## Score

Classic-normalized score(mm-equivalent) = Classic arithmetic mean multiplied
by the geometric mean of five model/Classic per-scroll mean-distance ratios.
Classic's exact scale is5.7226813847mm; compute using unrounded source values.
Each scroll has20% proportional influence. This is not an observed mean length.
Per-scroll trace metrics are unchanged. Earlier Combined EMA999 at1,920,000
samples/u15,000 remains highest among21evaluations at5.790; latest finetune
EMA999 is5.724, essentially matching Classic's5.723 aggregate.

Full metrics, selection reasons, exact metric names, snapshot hashes and source
paths: [21-snapshot comparison](benchmarks/finetune_all_snapshots_20260913.md).

## Reproduction

Approved runner, configured by `vesuvius/scripts/fiber3d_experiment.json`:

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment production-watch
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment trace-series-prepare
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment trace-series-next
```

The watch was bound to the specific finetuning launch and6,336,000sampleboundary.
Preparation freezes configured variants into a new output root; do not rerun it
on existing roots. Next runs one pending evaluation, audits, saves a report and
checks the stopped training root. Repeat next for each configured variant.
`expected_training_state=dead` was explicit for both series; default remains
alive for evaluations of a paused ongoing training. Neither mode restarts training.

Artifact series: `/ephemeral/hendrik/fiber3d_train2/comparisons/finetune_four_20260913`
and `/ephemeral/hendrik/fiber3d_train2/comparisons/finetune_best_emas_20260913`.
Current runner configuration points at the completed second series.

CPU regression validation:

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/neural_tracing/test_experiment_sample_stop.py
```

31passed. Coverage includes stopped/alive invariants, best raw selection,
cache isolation, geometric normalization, invalid baselines, and snapshot labels.
