# Task log: CUDA post-sampling augmentation

- Baseline exact production CPU prefix with 128 shared single-threaded workers:
  262.75 samples/s total, 104.8 CPU cores, before H2D. Stage-one loading averaged
  327.79ms/sample and the CPU value prefix averaged 66.64ms/sample. Artifact:
  `/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260908_213630_839413/result.json`.
- Dense coordinate generation remains CPU-side because VC3D immediately consumes
  it for CPU volume sampling. No GPU-to-CPU coordinate round trip will be added.
- Independent review approved that boundary and required per-sample reductions,
  explicit FP32 stream ordering, partition-invariant noise, metadata lifetime,
  and distinct `blur`/`all` configuration. CUDA noise is keyed independently by
  canonical stream index; it is same-backend repeatable but intentionally not
  CPU-bit-identical because PyTorch CPU and CUDA generators differ.
- Implemented `all`: workers now retain native `uint8` sampled volumes and pass
  immutable augmentation parameters to an FP32 CUDA preparation-stream chain
  covering normalization, contrast, brightness, gamma, noise, and both blurs.
- Focused CUDA tests: 4 passed. Focused CPU transport/sampling tests: 9 passed.
- The 128-worker host boundary rose from 262.75 samples/s (104.8 CPU cores) to
  339.56 samples/s production (122.0 CPU cores) with 120 load-affinity workers;
  that is 2.783 samples/s per occupied CPU core. A 124-affinity run exhausted
  native worker resources, so 120 is the measured standalone limit.
- On two H100 GPUs (global batch 16), end-to-end training reached 88.020
  samples/s globally, or 44.010 samples/s/GPU. Mean H2D time fell from 8.06ms
  to 0.855ms/batch after raw uint8 transport; full CUDA value augmentation was
  35.22ms/batch while overlapped with training. Artifact:
  `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_sample_pipeline_async_batch_gpu_all_20260908_220615_721868`.
- Final repeats on the completed code measured 339.80 samples/s production
  (330.04 end-to-end) across 128 CPU workers / 122.0 occupied CPU cores, and
  88.492 samples/s globally across two H100s (44.246/GPU). Rank-0 iteration
  mean/median/p95 was 180.45/180.00/189.57ms.
