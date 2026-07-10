# Status: Parallel And Batched Loader I/O

- [x] Create current `task.md`.
- [x] Create current `task_plan.md`.
- [x] Update `planning/specs.md` with loader I/O batching requirements.
- [x] Add `loader_workers` config with logical-core default.
- [x] Add sampler-level batch API.
- [x] Implement flattened single-call sampler batch loading.
- [x] Wire `build_sample` to batch sample strip-z stacks.
- [x] Parallelize `load_batch` across CP samples while preserving order/skips.
- [x] Update docs/changelog/task log after implementation.
- [x] Run focused validation.
