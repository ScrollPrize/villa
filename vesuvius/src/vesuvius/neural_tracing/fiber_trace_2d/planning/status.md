# Status: Fiber Scale-2 Output, Sparse Accumulator Activity, and 64³ Chunks

- [x] Read the local workflow and current shared inference specifications.
- [x] Inspect current Fiber scale semantics and Lasagna/Fiber OME chunk defaults.
- [x] Diagnose unconditional untouched-region mmap clearing and flush-time
  support rescans.
- [x] Draft implementation, spec, docs, tests, measurement, changelog, and
  task-log plan.
- [x] Independent review against task, specs, overarching plan, and code.
- [x] Correct the initial conflation of `--inference-scaledown-power` with
  tracer/model config `scaledown`.
- [x] Incorporate review findings on direct-footprint boundary semantics,
  exact scale validation, dirty-product state, ring reuse, store limitations,
  progress deduplication, and interruption cleanup.
- [x] User approved using the historical Lasagna weighted-pyrdown and border
  behavior unchanged for Fiber inference.
- [x] Obtain approval to begin the complete implementation plan.
- [x] Implement Fiber inference-scaledown-power default and factor conversion.
- [x] Implement lazy support/activity tracking and dirty-only flush/release.
- [x] Change Lasagna/Fiber OME chunk defaults to 64³.
- [x] Add shared progress/flush observability.
- [ ] Run unit, integration, sparse-allocation, resume, crop, and representative
  performance validation.
- [x] Run focused shared-runner and circular reuse regressions.
- [x] Update specs, docs, changelog, status, and task log after implementation.
