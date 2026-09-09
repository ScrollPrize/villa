# Plan: tune Fiber3D clipping and peak LR

1. Generalize the approved profile launcher to an explicit GPU list and exact
   optimizer-global-batch accounting while retaining the S1a baseline data and
   fresh deterministic initialization. On six ranks, represent global batch
   128 as eight full logical batches of 16. Assign logical ranks cyclically to
   six physical ranks, suppress per-backward DDP synchronization, and manually
   average accumulated gradients once per optimizer update.
2. Add opt-in global pre-clip L2 gradient-norm measurement using
   `torch.nn.utils.clip_grad_norm_`; handle AMP unscale before measurement and
   log LR, norm, and clipping occurrence to console, timing JSONL, and
   TensorBoard. Defaults remain disabled/no clipping.
3. Add an opt-in from-scratch LR-range schedule: linear warmup followed by
   exponential growth from configured start to end LR. The range scan does not
   clip gradients, so instability is observable.
4. Run 100 warmup plus 600 range steps on six H100s with full local batches of
   16 and exact optimizer global batch 128, using
   the old S1a NML reference data, BF16, GroupNorm, stride 4, and the shared
   asynchronous CUDA augmentation pipeline.
5. Analyze smoothed loss versus LR and pre-clip global norms. Choose LR
   candidates below divergence and a clip threshold around stable-region
   p95--p99 (or 2--5x median), intended to clip only occasionally.
6. Automate short fresh-initialization verification experiments derived from
   the range result, plus one matched clipping trial at the best unclipped LR.
   Compare held-out loss at exact global batch 128 and write consolidated JSON
   and Markdown reports while leaving final selection manual.
7. Add focused schedule/clipping/logging tests and update specs, profiling docs,
   changelog, status, and task log.

## Spec update

Document disabled-by-default global L2 clipping, pre-clip norm semantics,
AMP ordering, range schedule, and rank-consistent logging.

## Documentation update

Record exact reusable command, GPU topology, global batch limitation, LR/loss
curve results, norm percentiles, recommended settings, verification results,
and artifacts.

## Changelog

Add an entry for the new opt-in training and tuning controls.
