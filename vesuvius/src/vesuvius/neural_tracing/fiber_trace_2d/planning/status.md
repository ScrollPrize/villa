# Status: Fix Loader Threading Bottlenecks

- [x] Create current `task.md`.
- [x] Create current `task_plan.md`.
- [x] Identify warm-path threading bottlenecks in `load_batch`.
- [x] Add random-pass order prewarming before worker submission.
- [x] Make random pass lookup use a cached fast path.
- [x] Add a persistent loader-owned CP worker executor.
- [x] Keep `loader_workers=1` as a direct serial path.
- [x] Avoid redundant xyz array reads in supported strip-coordinate cache payloads.
- [x] Preserve deterministic output order and invalid-sample skip behavior.
- [x] Measure the unchanged local `--benchmark --load-only --profile` command.
- [x] Update specs, code docs, changelog, and task log.
- [x] Run focused validation.
