# Larger-batch learning-rate trial

Run3parallel exclusive GPU lanes, B16384 full compiled quadratic, learning rates
.03(control),.06(sqrt scaling),.12(linear scaling). Keep width_lr=2 explicitly,
all other settings unchanged from completed16k run.4032sampled updates,seed0,
fresh radial initialization,151CCW hypotheses,all parameters optimized from start.
Use existing approved matrix runner with one exclusive lease GPU1/2/3,24CPU,
32GiB perGPU. No loss/code changes and no reference initialization.

## Validation
Independent config review; monitor finite losses and progress. Compare per-pass
and final referenceL1(offset corrected),within+/-2,retention,sign errors,quantile
ranges,objective,elapsed and peakmemory. Same16k data/updatebudget across3runs.
Also compare previous4kLR.03 result; do not equate differently sized updatecounts.
Solver elapsed excludes finalreference report/save and managerqueue. Single seed.

## Spec update
No default or objective changes; experimentalLR only.

## Docs updates
Archive precedingtasklog; update task/plan/status/currentlog, complete progress
and results with exact command/configs, changelog. Persist experiment records.
