# Current snapshots versus dense test loss

Purpose: rising dense test loss need not imply worsening bidirectional tracing.
Freeze current EMA0.999, EMA0.99 and raw together, then evaluate in thatorder using
the locked50fiber/100direction workload. All captured at update20700,
2649600samples from combined JSON/NML run164117. Training staysalive; the existing
manager-exclusive evaluator alone causes pauses/reallocations.

## Reproduction

Use `trace_series` in `vesuvius/scripts/fiber3d_experiment.json`:
`variants=["current999", "current99", "current"]`. The `current` key selects
exactly current.pt rawweights, not best.pt. Omitted variants retain the original
three-EMA order. Select a fresh outputroot for eachseries.

Run approved `vesuvius/scripts/run_fiber3d_experiment trace-series-prepare` once,
then `trace-series-next` three times serially, reporting each result before next.
The shared freezer copies before reading metadata and validates raw vsEMA kind
and testmetric. Raw may contain complete EMA trainingstate. Each selection records
actualstep/samples/metric/hash; do not assume concurrentcurrentfiles share astep.
Once frozen, live snapshot updates cannot change evaluatedweights.

Outputroot:
`/ephemeral/hendrik/fiber3d_train2/comparisons/combined_current_three_20260912_26m`.
Contains series.json, immutableweights, per-variant selections/workloads/logs,
100direction artifacts/audits and cumulative comparison_after_<variant>.md.

## Interpretation

Overall score averages five per-scroll mean tracedlengths with20%weight each.
No pooledoverallscore. Compare each currentEMA with its previously evaluated
bestofthesamedecay, and compare currentraw/EMA at their matchedstep. Classic's
different testloss/trainingbudget is not comparable. Smallscrollsupport is limited
(0332threefibers,1451twofibers); these few checkpoints cannot establish a robust
statistical correlation or that one metric is universally invalid.

Current999: dense loss0.390237 vsbest9990.380882 (+2.46%); tracing6.444vs6.742mm
(-4.42%). Current99: loss0.387144 vsbest990.373058 (+3.78%); tracing6.743vs6.426mm
(+4.93%). Thus rising loss has opposite tracing outcomes in these two
same-decay comparisons; per-scroll regressions/improvements matter.

Current raw: loss 0.403531, tracing 6.496mm. It has worse test loss than both
current EMAs, but a higher overall tracing score than current EMA999. Raw improves
0332 relative to current EMA99 while declining on the other four scrolls. The
20%-weighted arithmetic mean gives equal weight to each scroll's mean, not equal
influence to relative improvements; 0332's much longer segments strongly affect
the overall numerical change. Only three fibers support that scroll's estimate.
Do not choose raw from its aggregate score alone or claim test loss has no value.

## Full reports and validation

- [After current EMA999](benchmarks/combined_current_three_20260912_current999.md)
- [After current EMA99](benchmarks/combined_current_three_20260912_current99.md)
- [After current raw](benchmarks/combined_current_three_20260912_current.md)

All three exited successfully and passed independent 100-direction raw-artifact
audits, in requested order. Full tables printed after each. CPU regression suite:
22 tests passed. No new inference/training numerical settings or manager changes.
The same training root remained alive throughout; no stop/relaunch or deviations.
Final resume verified: same manager task, running generation19, update20900
(2675200 samples), LR0.01.
