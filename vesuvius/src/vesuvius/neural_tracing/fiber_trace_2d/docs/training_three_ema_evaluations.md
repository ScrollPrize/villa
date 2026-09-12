# Three EMA evaluations with live training

Requested order: current EMA0.999, best EMA0.999, best EMA0.99 from the active
combined JSON/NML run. All three were frozen before the first evaluation; changing
live best/current files cannot change the selected weights afterward.

| Label | Step | Samples | Test loss | SHA256 prefix |
|---|---:|---:|---:|---|
| Current999 | 11300 | 1446400 | 0.3861950273 | 3e87efbc |
| Best999 | 9800 | 1254400 | 0.3855587457 | 7a4ac502 |
| Best99 | 9800 | 1254400 | 0.3819212223 | ab66f474 |

Source run:
`/ephemeral/hendrik/fiber3d_train2/runs/compare_5scroll_s1a_128_sd1_16x8_ema_500k_20260912_164117`.
Original live training controller PID1975660, launch164113, is not terminated.
Each existing evaluation requests manager-exclusive admission; adaptive training
releases its GPUs and later resumes. No manual pause, stop or relaunch commands.

## Approved runner

Settings live in `vesuvius/scripts/fiber3d_experiment.json`, key `trace_series`.
`run_fiber3d_experiment trace-series-prepare` captures and validates all snapshots
on CPU, requiring the original training root to be alive. It creates a new output
directory and records full hashes/metadata in `series.json`. Never re-prepare an
existing directory or run two series-next commands concurrently.

`run_fiber3d_experiment trace-series-next` executes only the next pending entry,
using the shared prepare/validate-full/full invocation and then the existing
100-direction raw-artifact audit. A failure blocks later entries until inspected;
it does not silently skip/retry an interrupted entry. Original training-root
aliveness is checked before and after evaluation, without sending signals.

## Reports

Output root:
`/ephemeral/hendrik/fiber3d_train2/comparisons/combined_three_ema_20260912`.
Each completed entry writes `comparison_after_<name>.md` with cumulative results.
`current999/`, `best999/`, `best99/` contain logs, workload/selection metadata,
and full trace/audit artifacts. Series state includes complete frozen SHA256s.

The comparison includes Classic and the six prior modern600k raw/EMA results.
Overall score is the arithmetic mean of per-scroll mean lengths, exactly20% per
scroll; pooled overall scores are omitted. Each scroll retains all distribution
statistics and highlights best/ties per metric. These new models are later than
600k; Classic is also longer-trained. This is not a matched-budget comparison.

CPU controller/report suite:13tests passed. All three full GPU evaluations
completed100directions and passed artifact audits, in requested order.

## Results

Equal-scroll means: current999 **6.240 mm**, best999 **6.180 mm**, best99 **6.389 mm**.
Best99 is the largest among the new entries and earlier comparisons, narrowly
above JSON EMA **6.308 mm**. These later combined checkpoints are not budget-matched
to the earlier600k models. Best99 wins0139/Paris4 by mean; JSONEMA wins0332;
Classic remains best on1299/1451.

Full cumulative tables: [final report](benchmarks/combined_three_ema_20260912.md).
Intermediate reports are preserved as comparison_after_current999.md and
comparison_after_best999.md under the output root; both were reported immediately
before launching the next evaluation. Original training root remained alive after
each evaluation and resumed between them through manager grants; no stop/relaunch.
After the third evaluation, the same training task reached running generation7
and update11500 (1472000 samples), LR0.01.
