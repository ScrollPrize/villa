# Task log: tune Fiber3D clipping and peak LR

- Used the historical `training_baseline_s1a_20260729.json` S1a NML data and
  model definition, fresh seed-0 initialization, BF16, GroupNorm, native output
  stride 4, and the shared asynchronous CUDA augmentation pipeline.
- Physical GPUs were `0,1,4,5,6,7`; GPUs 2 and 3 remained untouched. The target
  optimizer batch was exactly eight logical rank-local batches of 16. Physical
  ranks 0 and 1 processed two full logical batches; the other four processed
  one. All backwards used `DDP.no_sync()`, then accumulated gradients were
  manually summed and divided by six once per update. Losses were scaled by
  `6/8`, reproducing the eight-rank DDP mean without partial batches.
- Discarded all earlier tuning results based on balanced partial tail batches.
  Although those runs covered 128 samples, data-dependent loss denominators
  meant they were not exact eight-by-16 emulations.
- Exact-scheduler smoke artifact:
  `profile_6gpu_off_group_os4_b16_sample_pipeline_async_batch_gpu_all_lr_range_20260909_083709_662664`.
  It completed 35 updates without collective mismatch and produced identical
  reduced metrics on all ranks.
- Automated suite artifact: `tuning_suite_20260909_091325_561949`. One reusable
  invocation performed the range, derived fixed candidates, ran them, selected
  only which LR received a matched clipping comparison, and emitted
  `tuning_report.json` plus `tuning_report.md` for manual final selection.
- Exact LR scan artifact inside that suite:
  `profile_6gpu_off_group_os4_b16_sample_pipeline_async_batch_gpu_all_lr_range_20260909_091327_101834`.
  It completed 100 warmup plus 600 exponential updates from LR 0.001 through
  3.0 in 509.2s. Bias-corrected EMA loss was lowest at LR 0.08125 and crossed
  the configured divergence rule at LR 1.558. Pre-clip norms over usable range
  rows were median 0.652, p95 2.002, p99 2.780, max 3.905.
- Fresh held-out verification used 100 warmup updates and 200 total updates:
  LR 0.01/no clip reached test loss 1.431982; automatically derived LRs
  0.027082, 0.081246, and 0.155841 reached 1.659746, 1.683198, and 1.766804.
  LR 0.01 with the rounded p99 clip threshold 2.8 reached 1.432956. The clip
  threshold fired on only 3/200 updates and did not improve the result.
- Recommendation: peak LR 0.01, 100 warmup updates, clipping disabled. The
  range minimum was not a safe fixed peak for early held-out convergence.
- Exact-emulation throughput was about 172--175 samples/s in fixed runs. This
  is intentionally limited by two physical ranks doing twice the work and is
  not the expected eight-GPU production throughput.
