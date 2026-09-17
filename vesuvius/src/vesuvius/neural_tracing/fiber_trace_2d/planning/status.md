# Status

- [x] Read task and applicable workflow instructions.
- [x] Resolve checkpoint SHA and model output scale.
- [x] Write task plan.
- [x] Independently review the plan.
- [x] Freeze the exact checkpoint and validate output stride/metadata.
- [x] Create, prepare, and validate the immutable evaluation config.
- [x] Diagnose first full-resolution launch failure (`EFBIG`, 27 TB weight file).
- [x] Implement/test transparent sparse mmap Z-sharding.
- [x] Replace repeated blend-weight results, temporary CPU output copies, temporary Zarr arrays, and per-chunk fsyncs.
- [x] Validate the optimized path (73 focused tests; native 23/85/17; Python 282 passed, 8 skipped).
- [x] Diagnose ext4 transaction stalls caused by deleting full-volume sparse accumulator shards.
- [x] Specify the bounded hot-slot plus reusable unbounded mmap spill-arena replacement.
- [x] Independently review and correct the hot/spill arena plan.
- [x] Implement shared hot/spill arena allocation, ownership, eviction/promotion, pinning, and telemetry.
- [x] Add exact numerical and lifecycle regression coverage, including spill growth/reuse and crash-like cleanup.
- [x] Run the approved focused and replay test suites.
- [x] Diagnose POSIX-result-slot SIGBUS under host-wide shared-memory pressure.
- [x] Move reusable GPU result slots to one direct file-backed mmap and retain
  only the smaller input ring in POSIX shared memory.
- [x] Resume the approved full benchmark (64-cubed output chunks, 128
  accumulator workers, 192 logical CPUs).
- [x] Validate combined/per-scroll results and all 180 case artifacts.
- [x] Persist benchmark report, task log, status, and changelog.
