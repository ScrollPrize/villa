# 3D Augmentation Stream And 2D Spec Parity Status

- [x] Read local AGENTS.md workflow.
- [x] Replace `planning/task.md` with the current task.
- [x] Replace `planning/task_plan.md`.
- [x] Review relevant shared 2D specs.
- [x] Inspect current 3D augmentation, test-loader, and sample-index paths.
- [ ] Implement explicit raw/data/augmentation sample-index split in 3D.
- [ ] Add 3D `training.max_sample_index` behavior.
- [ ] Disable dense 3D test-loader augmentations by default with explicit opt-in.
- [ ] Make 3D `training.max_steps: 0` repeat indefinitely like 2D.
- [ ] Make 3D dense `test_control_points: 0` evaluate all held-out CPs in flat order.
- [ ] Match 3D prefetch step-count resolution to 2D semantics.
- [ ] Add 3D CLI `--resume` training support.
- [ ] Update shared console-progress spec to first 100 training steps.
- [ ] Ensure 3D samples carry full transformed fiber-line context while losses obey target mode.
- [ ] Add/update focused 3D tests.
- [ ] Update specs/docs/changelog/task_log.
- [ ] Run validation commands.
