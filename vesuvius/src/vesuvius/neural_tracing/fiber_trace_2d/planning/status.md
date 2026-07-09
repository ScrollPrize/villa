# Status: Python-Owned Parallel Prefetch Downloads

- [x] Read local `AGENTS.md` instructions.
- [x] Read `planning/todo.md`.
- [x] Read current high-level plan and specs.
- [x] Replace `planning/task.md` with the active prefetch task.
- [x] Replace `planning/task_plan.md` with a reviewable implementation plan.
- [x] Include the Python-side `.empty` marker handling in the plan.
- [x] Include the dependency-producer/download-consumer parallelization plan.
- [x] User review of `planning/task_plan.md`.
- [x] Implement VC3D dependency-only binding/accessors returning source URL/key, cache path, empty path, extension, and payload format metadata.
- [x] Remove VC3D prefetch/download binding and Python fallback chunk reads.
- [x] Implement Python prefetch request/cache/download pipeline with direct-source uncompressed `.bin` support, unique temp files, atomic rename, retries, and zero-byte `.empty` markers.
- [x] Remove temporary debug prefetch output.
- [x] Update specs and docs.
- [x] Add/update tests.
- [x] Run focused validation commands.
- [x] Update task log and changelog.
