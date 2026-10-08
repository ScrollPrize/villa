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
Optional `variants` selects a nonempty unique ordered list from `current999`,
`best999`, `best99`; omission retains the original three-entry order. For a new
best pair use `["best99", "best999"]` and a fresh output directory. Both weights
are frozen before evaluation starts, even if training updates its best files.

"Best" is tracked independently for raw weights and each EMA decay, using the
lowest aggregate dense test `loss_total` of that track. It is not per metric or
per scroll; angle, historical metrics, and equal-scroll tracing scores do not
select snapshots. The recorded checkpoint step can precede the current run step.

`run_fiber3d_experiment trace-series-prepare` captures and validates all snapshots
on CPU, requiring the original training root to be alive. It creates a new output
directory and records full hashes/metadata in `series.json`. Never re-prepare an
existing directory or run two series-next commands concurrently.

`run_fiber3d_experiment trace-series-next` executes only the next pending entry,
using the shared prepare/validate-full/full invocation and then the existing
100-direction raw-artifact audit. A failure blocks later entries until inspected;
it does not silently skip/retry an interrupted entry. Original training-root
aliveness is checked before and after evaluation, without sending signals.
After diagnosing a failed attempt, `run_fiber3d_experiment trace-series-retry`
explicitly retries that same frozen selection into a fresh `-retryN` directory.
Previous logs and error are retained. This does not retry a running entry or
refreeze live weights. Prediction-only cache namespaces incorporate the full
output directory identity, so two series may both contain a `best99` evaluation.

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

## Follow-up best pair near1.9M samples

Frozen before either evaluation: best99 at update14700/1881600samples,
testloss0.373057780; best999 at update15000/1920000samples, testloss0.380881539.
Both completed the same50fiber/100direction workload and passed artifact audits.
Equal-scroll means: best99 **6.426mm**, best999 **6.742mm**, compared with the
previous best99 **6.389mm**. Best999's gain is concentrated on0332/1451, with
only3/2testfibers respectively; no statistical significance claim.

Full cumulative comparison and hashes:
[follow-up report](benchmarks/combined_best_pair_20260912_19m.md).
Original artifacts and intermediate report:
`/ephemeral/hendrik/fiber3d_train2/comparisons/combined_best_pair_20260912_19m`.
The first best99 attempt failed before inference because a leaf-only prediction
cache name collided with the previous series. It was retried after fixing the
namespace, retaining the same frozen weights and failed logs. No training
stop/relaunch or cache deletion. Both evaluations used manager-exclusive grants.
Validation:20CPUtests, two completeGPUevaluations and independent report review.
After the pair, the original training task resumed running in generation13.
