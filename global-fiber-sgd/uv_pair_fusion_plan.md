# Fused pair evaluation with unchanged minibatch updates

User requests fused evaluation; keep4096constraint batches,151windings and
minibatch optimizer updates. Preserve current full joint objective, precision,
lookup interpolation, probabilities and all winding hypotheses.

## Implementation and spec update
Start with PyTorch fullgraph compilation of the existing shared pair_objective,
including its backward, instead of duplicating math in a handwritten kernel.
Installed torch2.11/triton3.6 are available. Opt-in pair_backend=compiled;
eager remains portable reference/fallback. No automatic silent fallback when
compiled explicitly selected. Existing eager default retained until measured.
No mixed precision, pruning, freezing, changed losses or batchsize.

Add a GPU-managed benchmark mode through the existing approved UV runner.
Use real4096constraint batches from the151reference-initialized tensors and
perturbed/current states; compare objective and gradients of probabilities,
q and v. Include expected-cost and optional sign-charge paths in small tests.
Warmup separately, then repeated CUDAevent forward/backward timings and peak
allocated memory. Profile eager/compiled kernels (same GPU grant), report
mean,p50,p95 and compile latency. Test shared backend selection, CPU fallback,
invalid configs, and existing objective tests. If compiled is unsuitable,
report evidence before proposing a different implementation.

## Validation scope
FP32 arithmetic preserved but fusion can change rounding; compare absolute and
relative gradient errors and reject material disagreement. GPU benchmark includes
actual K151/B4096 and a small uneven-tail batch. Then run matched short optimizer
trajectories with same data/init and minibatchupdates, report loss/gradient checks.
No full10pass solve unless needed to verify a justified integration issue.

## Docs/results
Independent review before implementation; task log records commands, inputs,
versions,GPU,timings,memory,profiling and limitations. Report measured speedup
without promising orders of magnitude from compilation alone.
