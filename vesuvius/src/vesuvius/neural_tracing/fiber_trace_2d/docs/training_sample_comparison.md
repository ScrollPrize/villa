# Short raw/EMA training comparison

Comparison: S1a600k, fresh five-scroll JSON600k, combined JSON+S1a NML600k.
Counts refer to samples, not optimizer updates.
Global128 and test interval100 round the first cap to601600samples/update4700.
The initial combined cap was499200samples/update3900 (user typo); its winning
continuation is additionally evaluated at601600samples/update4700. Initial configs are
train_compare_5scroll_ema_600k.json and train_compare_5scroll_s1a_ema_500k.json.
Both start from scratch at LR.01, warmup100, EMA.99/.999, regular run directory.
Combined Paris4 JSON directories share sampling_group=json, NML usesgroup=nml;
explicit weight1each means10%ofall samples each. Every other scroll gets20%.
Within groups, sampling remains proportional to fiber arclength.

The approved experiment controller accepts `production-watch`. Its control JSON
`production_stop` contains an explicit `launch` directory and positive `samples`
limit. It binds the launch's PID/start/boot identity, not `latest`, and waits for
completed raw/EMA current snapshots and consistent best snapshots before TERM.
The trainer handles bounded worker shutdown. An unverifiable identity or failed
shutdown is an error, never permission to kill an unrelated process. Successful
stop writes `sample_stop_result.json` in that launch directory. This is a
checkpoint-boundary stop, not an exact mid-macrobatch sample cap. Keep the monitor
alive until completion; a live trainer does not reread edited max_steps values.

Select best raw by current test loss, and one best EMA across the configured
decays by its current test loss. Freeze files before inference and retain hashes.
Run the existing locked50fiber/100direction bidirectional evaluator on all six
checkpoints; compare complete-reference mean segment length and per-scroll
failures. Training test losses from different datasets are not directly comparable.
Resume the winning run from its resumable raw checkpoint (including EMA state),
not its inference-only EMA export.

`run_fiber3d_experiment trace-evaluate` freezes and evaluates one configured
best raw/EMA. `run_fiber3d_experiment comparison` executes the remaining serial
campaign from its control JSON: audit the completed initial raw evaluation,
evaluate S1a EMA, run JSON training/evaluations, run combined training/evaluations,
then select the largest complete-reference mean segment length. It aborts on any
failed training or incomplete evaluation, never skips a condition. The campaign
records comparison.json and per-evaluation selection.json, workload.json,
prepare/validate-full/full logs and trace_audit.json under its output root.
Continuation uses the original winning-stage config with2Mupdates and resumes
that run's current raw state, including EMA history. It does not substitute the
best EMA inference weights into an unrelated optimizer state.

## Initial results (2026-09-12)

S1a stopped601600samples. Both selected checkpoints completed the identical
50fiber/100direction evaluation and raw-artifact audit:

| Selection | Checkpoint samples | Mean segment mm | Restarts |
|---|---:|---:|---:|
| S1a best raw |358400|1.819761|4341|
| S1a best EMA(.99) |384000|2.200860|3572|
| JSON best raw |537600|2.750700|2838|
| JSON best EMA(.99) |588800|3.267917|2373|
| Combined500k best raw |499200|2.193691|3584|
| Combined500k best EMA(.99) |499200|3.695271|2087|

All six initial evaluations completed; combined resumed from its raw current
checkpoint, preserving optimizer and EMA state. The initial campaign report is
/ephemeral/hendrik/fiber3d_train2/comparisons/nml_json_combined_20260912/comparison.json.

## Corrected combined600k follow-up

`run_fiber3d_experiment comparison-followup` requires a verified completed
`production-watch` stop at the configured boundary. It selects raw/EMA bests
across both parent and continuation runs, checks checkpoint sample counts against
the cutoff, reuses the managed locked evaluator and raw-artifact audit, then
resumes the continuation's raw current snapshot. Control settings and report
paths are in `fiber3d_experiment.json`; never rerun an existing output directory.
It does not silently skip failures or choose later checkpoints. The additional
report is `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_600k_20260912/report.json`.

## Matched 601600-sample comparison

All conditions trained to601600 samples; checkpoint selection is lowest test loss
available by that cutoff, separately for raw and EMA. All selected EMAs have decay0.99.
Checkpoint selection is not based on tracing performance. Every evaluation completed
50fibers/100directions and passed the raw-artifact audit. Lengths below are mm.
Higher segment lengths and fewer restarts are better. Pooled percentiles use all
segments directly; terminal and zero-length segments are included.

| Model | Step | Checkpoint samples | Selection test loss |
|---|---:|---:|---:|
| NML raw | 2800 | 358400 | 2.388740 |
| NML EMA | 3000 | 384000 | 2.491809 |
| JSON raw | 4200 | 537600 | 0.405784 |
| JSON EMA | 4600 | 588800 | 0.389835 |
| Combined raw | 4600 | 588800 | 0.412379 |
| Combined EMA | 4600 | 588800 | 0.400296 |

NML selection test loss uses its own test set; do not compare its absolute loss
to the five-scroll test losses. Tracing evaluation is identical for all models.

### All scrolls

50 fibers, 100 directions, 8081.558 mm directed reference.

| Model | Restarts | Segments | Mean | Median | P05 | P95 | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| NML raw | 4341 | 4441 | 1.820 | 0.098 | 0.009 | 9.698 | 36 |
| NML EMA | 3572 | 3672 | 2.201 | 0.076 | 0.008 | 11.401 | 39 |
| JSON raw | 2838 | 2938 | 2.751 | 0.053 | 0.011 | 12.693 | 19 |
| JSON EMA | 2373 | 2473 | 3.268 | 0.071 | 0.010 | 15.793 | 25 |
| Combined raw | 2571 | 2671 | 3.026 | 0.090 | 0.011 | 16.656 | 17 |
| Combined EMA | 1927 | 2027 | 3.987 | 0.149 | 0.009 | 23.270 | 24 |

### PHerc0139

10 fibers, 20 directions, 1392.347 mm directed reference.

| Model | Restarts | Segments | Mean | Median | P05 | P95 | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| NML raw | 902 | 922 | 1.510 | 0.078 | 0.007 | 9.693 | 10 |
| NML EMA | 750 | 770 | 1.808 | 0.082 | 0.005 | 9.358 | 8 |
| JSON raw | 485 | 505 | 2.757 | 0.097 | 0.005 | 15.315 | 8 |
| JSON EMA | 437 | 457 | 3.047 | 0.106 | 0.005 | 15.933 | 13 |
| Combined raw | 448 | 468 | 2.975 | 0.133 | 0.007 | 15.984 | 4 |
| Combined EMA | 459 | 479 | 2.907 | 0.094 | 0.006 | 15.596 | 7 |

### PHerc0332

3 fibers, 6 directions, 471.222 mm directed reference.

| Model | Restarts | Segments | Mean | Median | P05 | P95 | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| NML raw | 69 | 75 | 6.283 | 2.909 | 0.025 | 22.058 | 0 |
| NML EMA | 52 | 58 | 8.125 | 0.250 | 0.008 | 35.816 | 0 |
| JSON raw | 26 | 32 | 14.726 | 1.999 | 0.007 | 60.832 | 0 |
| JSON EMA | 17 | 23 | 20.488 | 8.486 | 0.009 | 69.482 | 0 |
| Combined raw | 23 | 29 | 16.249 | 2.528 | 0.006 | 58.601 | 0 |
| Combined EMA | 19 | 25 | 18.849 | 10.735 | 0.004 | 67.689 | 0 |

### PHerc1299

5 fibers, 10 directions, 599.748 mm directed reference.

| Model | Restarts | Segments | Mean | Median | P05 | P95 | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| NML raw | 344 | 354 | 1.694 | 0.260 | 0.007 | 9.602 | 5 |
| NML EMA | 318 | 328 | 1.828 | 0.248 | 0.009 | 9.318 | 6 |
| JSON raw | 225 | 235 | 2.552 | 0.223 | 0.009 | 11.192 | 1 |
| JSON EMA | 233 | 243 | 2.468 | 0.217 | 0.012 | 11.649 | 2 |
| Combined raw | 250 | 260 | 2.307 | 0.184 | 0.009 | 11.731 | 3 |
| Combined EMA | 216 | 226 | 2.654 | 0.257 | 0.012 | 14.332 | 3 |

### PHerc1451

2 fibers, 4 directions, 94.430 mm directed reference.

| Model | Restarts | Segments | Mean | Median | P05 | P95 | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| NML raw | 83 | 87 | 1.085 | 0.079 | 0.009 | 5.855 | 2 |
| NML EMA | 72 | 76 | 1.243 | 0.156 | 0.010 | 6.246 | 0 |
| JSON raw | 43 | 47 | 2.009 | 0.217 | 0.011 | 11.336 | 0 |
| JSON EMA | 37 | 41 | 2.303 | 0.268 | 0.019 | 11.598 | 0 |
| Combined raw | 36 | 40 | 2.361 | 0.436 | 0.045 | 11.234 | 0 |
| Combined EMA | 39 | 43 | 2.196 | 0.262 | 0.029 | 11.056 | 0 |

### PHercParis4

30 fibers, 60 directions, 5523.811 mm directed reference.

| Model | Restarts | Segments | Mean | Median | P05 | P95 | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| NML raw | 2943 | 3003 | 1.839 | 0.092 | 0.010 | 9.102 | 19 |
| NML EMA | 2380 | 2440 | 2.264 | 0.061 | 0.009 | 11.655 | 25 |
| JSON raw | 2059 | 2119 | 2.607 | 0.034 | 0.013 | 10.385 | 10 |
| JSON EMA | 1649 | 1709 | 3.232 | 0.048 | 0.012 | 13.225 | 10 |
| Combined raw | 1814 | 1874 | 2.948 | 0.066 | 0.012 | 14.619 | 10 |
| Combined EMA | 1194 | 1254 | 4.405 | 0.145 | 0.011 | 27.513 | 14 |

Combined EMA improves mean segment length22.0% over JSON EMA and reduces
restarts18.8%. Relative to its earlier500k result, mean improves7.9% and restarts
decrease7.7%. Combined training was resumed from raw current at4700, retaining
optimizer/all EMA state, with LR0.01 and2M-update limit.

The continuation launch is
`/ephemeral/hendrik/fiber3d_train2/runs/.launches/compare_5scroll_s1a_128_sd1_16x8_ema_500k-20260912_164113`.
The legacy500k text in its name does not limit its training budget.

### Frozen evaluation provenance

- NML raw: `/ephemeral/hendrik/fiber3d_train2/evaluations/compare_s1a_600k_best_raw_20260912`; SHA256 `5b359ed861fcccb491d3cf35fd159795aabce762f46ac54237bfdd1dfd6c00c9`.
- NML EMA: `/ephemeral/hendrik/fiber3d_train2/comparisons/nml_json_combined_20260912/s1a_600k_ema`; SHA256 `4f7b875e116332202a5eb762c2a7229280a8b7631326e3340f898a76c523cdf7`.
- JSON raw: `/ephemeral/hendrik/fiber3d_train2/comparisons/nml_json_combined_20260912/json_600k_raw`; SHA256 `8e4ed6d8794c6070335c33572fd55bfc2e657cd2e3dcc1cee4885ddce236cdb4`.
- JSON EMA: `/ephemeral/hendrik/fiber3d_train2/comparisons/nml_json_combined_20260912/json_600k_ema`; SHA256 `c2c5053f91a4c3ccccfe362795035a1c9c0e18e1fdf1b125984d96e71bffe4d8`.
- Combined raw: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_600k_20260912/raw`; SHA256 `5f7bccacb390174da74feafd76a3edfbfd880a59bee97efac573f56633f42314`.
- Combined EMA: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_600k_20260912/ema`; SHA256 `3c9766fc5d9f4e41686cbb6d4d780369c4a865ea7c13b4c6889636e015558124`.

The overall gain is mainly Paris4: combined EMA mean4.405mm versus JSON EMA3.232mm.
Combined EMA is also better on1299, but JSON EMA remains better on0139,0332 and1451.
Paris4 contributes30/50 fibers and68.4% of directed reference length; the pooled
score is not an equal-scroll average. Outside Paris4, pooled mean is3.309mm for
combined EMA versus3.348mm for JSON EMA. Small-scroll results have limited support
(0332:3fibers;1451:2fibers); no statistical significance claim is made.
