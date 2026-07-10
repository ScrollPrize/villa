# Status: Cached Fused Augmentation Maps For Line And CP Warp

- [x] Read local requirements and current spec section.
- [x] Identify remaining gap: transform object still permits per-call smooth
  setup and line/CP are not batched/reused enough.
- [x] Create current `task.md`.
- [x] Create current `task_plan.md`.
- [x] Update `planning/specs.md` with fused-map performance requirements.
- [ ] Implement cached fused transform internals.
- [ ] Batch line+CP source-to-output mapping.
- [ ] Reuse transformed line/CP across shared strip-z offsets where applicable.
- [ ] Update docs/changelog/task log after implementation.
- [ ] Run focused validation.
