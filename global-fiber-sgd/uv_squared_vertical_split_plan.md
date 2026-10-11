# Corrected quadratic fused solve

Run the existing quadratic configuration with corrected V equality, independent
weak V≈Z prior and removed pairwise wrapped phase. No other solver changes.
Fresh non-reference initialization, CCW,151 hypotheses,4096 random minibatches,
16130 updates (~10 event-pass equivalents), all variables active from start.
Reuse cached upstream sign/HV/direction evidence and approved managed runner.
GPU1 shared24GiB reservation,8CPU; no exclusive allocation.

## Validation and report
Use the81 passing tests for committed loss changes. Observe logs and
completion; report initial/per-pass/final reference-offset-corrected L1, within2,
retained length, sign violations, winding quantiles and elapsed time. Compare
historical quadratic run, labeling changed losses and nonisolated timing.
No visualizations or new optimization stages.

## Spec update
No new formulation: execute the corrected loss already approved.

## Docs updates
Archive preceding log; update task/status/log and store reproducible config,
progress table and results. Commit code first and experiment records at finish.
