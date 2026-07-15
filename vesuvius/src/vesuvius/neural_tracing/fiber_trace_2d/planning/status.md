# Process-Level Startup Fiber Geometry Preload Status

- [x] Replace task statement with process-level startup preload requirement.
- [x] Replace task plan and remove thread-local worker-handle direction from the plan.
- [x] Revert thread-local loader handle experiment from code.
- [x] Add process-level startup geometry jobs and worker function.
- [x] Wire `ProcessPoolExecutor` into startup geometry preload.
- [x] Preserve serial `loader_workers=1` path.
- [x] Update specs/docs/changelog after implementation.
- [x] Update tests by removing thread-local assertions and covering process startup.
- [x] Run focused test suite.
- [x] Run benchmark command and report startup/load-only results.
