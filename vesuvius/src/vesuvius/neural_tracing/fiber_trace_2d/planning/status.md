# Status: Cached Fused Augmentation Maps For Line And CP Warp

- [x] Read local requirements and current spec section.
- [x] Identify remaining gap: transform object still permits per-call smooth
  setup and line/CP are not batched/reused enough.
- [x] Create current `task.md`.
- [x] Create current `task_plan.md`.
- [x] Update `planning/specs.md` with fused-map performance requirements.
- [x] Implement cached fused transform internals.
- [x] Batch line+CP source-to-output mapping.
- [x] Reuse transformed line/CP across shared strip-z offsets where applicable.
- [x] Update docs/changelog/task log after implementation.
- [x] Run focused validation.
