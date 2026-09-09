# Plan: tune Fiber3D training batch size

1. Add a benchmark-only `--batch-size-per-gpu` override to the existing
   reproducible two-GPU profile harness and record it in artifact metadata and
   names. Do not change production defaults.
2. Establish batch 8 as the matched final-code baseline, then sweep larger
   per-GPU batches on GPUs 0 and 1. Keep GroupNorm, output stride 4, BF16,
   post-sampling CUDA augmentation, workers, warmup, and measured steps fixed.
3. Stop increasing after an OOM or after throughput clearly ceases improving.
   For each successful point report global samples/s, samples/s/GPU, iteration
   mean/median/p95, and peak allocated/reserved GiB per GPU.
4. Select the highest-throughput safe point, update the reusable benchmark args,
   and run a confirmation repeat.
5. Run focused harness tests and update task status, task log, and the two-GPU
   profiling document.

## Spec update

No production contract changes. Record that benchmark batch-size comparisons
must use a per-GPU value and report both global and per-GPU throughput.

## Documentation update

Add the batch-size sweep table, exact reusable command, artifacts, selected
point, and memory limitation to the training profiling document.

## Changelog

No entry unless the selected size is applied to a production configuration.

Independent-review refinements: each point runs in a fresh DDP process group
with a five-minute fail-fast timeout. Throughput uses the slowest-rank elapsed
window and latency uses the per-step maximum across ranks. The declared sweep is
8, 12, 16, 24, then 28 samples/GPU; it stops before a projected point would
leave less than 10% reserved-memory headroom. Contenders 24 and 28 receive
confirmation runs. One sample means one 128-cubed CP patch and GiB means 2^30
bytes. The result is benchmark-safe only; changing global optimizer batch size
requires a later convergence evaluation.
