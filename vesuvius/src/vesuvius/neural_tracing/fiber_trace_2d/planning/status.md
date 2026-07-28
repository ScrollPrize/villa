# Status: Native 3D Trace2CP GPU Sparse Sampling

- [x] Read project AGENTS workflow.
- [x] Read current specs and high-level plan for relevant constraints.
- [x] Replace `task.md` with the native 3D Trace2CP sparse GPU sampling task.
- [x] Replace `task_plan.md` with the detailed implementation plan.
- [x] Review task plan against specs before implementation.
- [x] Implement sparse Lasagna normal sampler with tensor interpolation and Lasagna closed-form axis reconstruction.
- [ ] Implement shared sparse inferred-field sampling path.
- [x] Remove per-lookup CPU block-output copies from tracer candidate lookup.
- [x] Keep point-choice helper tensor-native for caches that support torch lookup.
- [x] Add/update tests.
- [x] Update specs/docs/changelog/task_log after implementation.
- [x] Run focused validation.
- [ ] Run full before/after metric profiling on the user dataset.
