# Task log: profile the stride-4 full-pipeline gap

- Native stride-4 GroupNorm implementation committed as `66f3134b7` after 269 Fiber3D/profile tests and two directly affected shared Lasagna tests passed.
- Existing paired result: replay throughput increased from 55.301 to 95.714 samples/s, while real-pipeline throughput increased from 50.231 to 59.154 samples/s. This task will measure rather than assume the cause of that gap.
- Experiments are restricted to physical GPUs 0 and 1 and artifacts below `/ephemeral/hendrik/fiber3d_train2`.
- Independent review found that stride-4-only reruns could locate a bottleneck but not explain the relative speedup. The plan now requires contemporaneous stride-1 controls, uninstrumented versus instrumented controls, and an intermediate raw-batch replay retaining preparation/H2D/target work.
- Fresh stride-4 results were 95.708 samples/s for prepared-GPU replay, 88.224 for augmented-CPU replay, and 60.115 for the legacy real pipeline. Loader instrumentation reduced the legacy result to 58.455 samples/s (2.8% overhead).
- The legacy trainer assigns whole batches to eight workers per rank. Profiled batch service averaged about 2.08s and produced an eight-step starvation wave; one rank waited for input while the other waited at DDP synchronization.
- Extracted the benchmark operations into a shared module and connected the existing ordered shared process pool to training. One global 128-worker/512-slot budget is partitioned across ranks; logical indices map to the canonical DDP sample stream and shared slots remain leased until synchronized H2D completion.
- Integrated throughput was 84.028 and 83.595 samples/s uninstrumented (83.812 mean), while an instrumented control reached 84.853. This recovers 79.5% of the replay-to-real loss and reaches 95.0% of the CPU-replay ceiling.
- The first integration trial completed numerical work at 84.028 samples/s but did not explicitly close its pipeline during trainer teardown. The launcher was interrupted after output; explicit close was added and the next run exited normally.
- Post-extraction standalone shared-pool validation sustained 222.09 production samples/s and 218.41 samples/s including process startup/drain over 4,096 measured samples. Artifact: `/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260908_162725_479470/result.json`.
- Validation: 289 focused Fiber3D, profiling, and ordered-pipeline tests passed.
