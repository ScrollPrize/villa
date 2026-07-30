# VC3D Persistent Fiber-Traced Segments

## Planning

- [x] Read the user request and repository instructions.
- [x] Trace current fiber trace completion, auto-save, finalization, CP edit,
  JSON persistence, optimizer, and Ctrl-right-click paths.
- [x] Replace `planning/task.md` with the current task.
- [x] Replace `planning/task_plan.md` with data model, implementation, tests,
  spec, docs, changelog, and risks.
- [x] Perform a direct consistency review against current code and specs.
- [ ] Obtain independent agent review (not permitted by the active
  no-delegation runtime policy unless explicitly requested by the user).
- [ ] Obtain user approval to begin implementation.

## Implementation

- [ ] Add VC3D CP state with optional `segmentToNext` metadata and mutation
  helpers.
- [ ] Add version-2 object-valued CP serialization and version-1 loading.
- [ ] Update every C++, Python, CLI/probe, import/export, and sync/merge reader
  to parse and validate CP-owned segment metadata.
- [ ] Make accepted trace completion finalized and atomically attach its
  record.
- [ ] Protect valid traced ranges in existing-line and full-reinitialization
  Lasagna paths.
- [ ] Apply scoped invalidation/remapping to every CP mutation path.
- [ ] Add transactional Ctrl-right-click reversion to Lasagna optimization.

## Validation And Documentation

- [ ] Add segment-state, version-1 compatibility, version-2 round-trip, and
  strict cross-reader schema tests.
- [ ] Add existing-line and full-reinitialization protection tests.
- [ ] Add trace/save/edit/reload/revert lifecycle and menu tests.
- [ ] Build focused tests and VC3D with `-j32`.
- [ ] Run focused test binaries and relevant broader regressions.
- [ ] Perform the manual VC3D trace/edit/reload/revert smoke test.
- [ ] Update `planning/specs.md`.
- [ ] Update code-structure and VC3D fiber JSON documentation.
- [ ] Update `planning/changelog.md`.
- [x] Replace `planning/task_log.md` for this task.
