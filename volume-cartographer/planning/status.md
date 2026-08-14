# Status

- [x] Read the active task, render/fetch specification, and broader plan.
- [x] Trace `openSource()`, `Volume::setIOThreads()`, service construction, and
      scheduler replacement callers.
- [x] Reproduce and identify the non-Valgrind CI failure.
- [x] Write and independently review the implementation plan.
- [x] Split service-global and source-local configuration types.
- [x] Replace `openSource()` with source-only `acquireSource()`.
- [x] Implement in-place source-read admission reconfiguration.
- [x] Remove scheduler replacement/migration and `Volume::setIOThreads()`.
- [x] Update all production and isolated-cache callers.
- [x] Replace the duplicate-fetch migration test and add transition coverage.
- [x] Update specification, API documentation, task log, and changelog.
- [x] Run focused tests, the complete core CI shard, VC3D build, and synthetic
      render validation.
