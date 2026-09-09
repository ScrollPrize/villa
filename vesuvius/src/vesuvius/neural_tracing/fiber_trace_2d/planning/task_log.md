# Task log: tune Fiber3D training batch size

- Starting matched baseline: per-GPU batch 8, global batch 16, two H100 GPUs,
  88.492 samples/s globally (44.246 samples/s/GPU), rank-0 iteration
  mean/median/p95 180.45/180.00/189.57ms, peak allocated/reserved
  15.39/20.61GiB on rank 0.
- Independent review required slowest-rank throughput, per-step max-rank
  latency, fresh-process timeout containment, explicit memory headroom, and
  confirmation runs. The harness now implements those timing and timeout rules.
- Batch 12: 96.419 samples/s globally (48.209/GPU), critical-path iteration
  mean/median/p95 248.63/248.39/258.45ms, 30.46GiB peak reserved/GPU.
- Batch 16: 100.948 samples/s globally (50.474/GPU), critical-path iteration
  mean/median/p95 316.66/316.02/325.75ms, 40.42GiB peak reserved/GPU.
- Batch 24 repeats: 105.330 and 105.666 samples/s globally, 60.45GiB peak
  reserved/GPU.
- Batch 28 repeats: 107.972 and 108.088 samples/s globally, or 53.986 and
  54.044/GPU. Critical-path iteration means were 518.36 and 517.79ms; peak
  reserved was 70.48GiB/GPU. Selected as the fastest benchmark-safe point.
- Did not run batch 32 because linear reserved-memory projection exceeded the
  required 10% headroom. Selection is not yet a production default: full
  evaluation/visualization memory and convergence at global batch 56 remain to
  be tested.
