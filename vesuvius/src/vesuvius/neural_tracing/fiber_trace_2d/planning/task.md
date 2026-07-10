# Task: Training Benchmark And Profiling Mode

Implement the profiling item from `planning/todo.md`:

- Add `--benchmark` to run training work for the first 100 batches, skip testing,
  and report samples/s where samples are individual CNN image patches.
- Add `--profile` to measure timings by stage: coord generation, coord
  augmentation, Zarr read/sampling, image/value augmentations, forward, and
  backward plus optimizer step.
- Add `--load-only` so benchmark/profile can isolate data loading by running
  coordinate generation, coordinate augmentation, and volume sampling, while
  skipping image/value augmentation and model training work.
- After the 100 batches, print a summary with average stage time per CNN patch.
- During profiling, print table-like per-batch values so stage timing is easy to
  inspect.
