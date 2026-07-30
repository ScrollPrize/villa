# Native Fiber Trace Locality And Scheduling Optimization Log

## Starting Point

- Retained baseline: 1.869s wall / 8.222s CPU, 6,910,839 candidates,
  4,318 generations, and 7 restarts over 87 segments.
- Dominant retained stages: fused pinned-corner gather/decode/score 0.845s,
  frontier construction 0.217s, pruning 0.167s, task construction 0.137s,
  and start sampling 0.132s.
- Prior active task details were intentionally discarded. Durable results remain
  in `planning/changelog.md` and git history.

## Planning

- The new task covers worker granularity, deterministic partial parent
  selection, compact frontiers, spatial chunk/cube ordering, unique-cube corner
  reuse, persistent two-depth pins, bounded envelope prefetch, rolling pins,
  nearby fixed caps, and adaptive escalation.
- The mandatory depth-one/depth-two decision barrier remains: second-depth
  coordinates cannot be generated until first-depth scoring selects parents.
- Representative benchmarks require explicit user approval immediately before
  every invocation. The exact existing command and cache path must be reused.
- No independent-agent tool is available in the current context. Direct review
  found the plan consistent with the current original-index determinism,
  shared-corner-visitor, cap-32, exact-lazy, exhaustive-mode, cache-budget, and
  portability requirements.
- The review confirmed that depth-two work cannot be batched before first-depth
  selection. Locality work must therefore use spatial ordering and a bounded
  session across the required decision barrier, not speculative expansion.
- No implementation or representative benchmark has been run for this task.
