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
- [x] Obtain user approval to begin implementation.

## Implementation

- [x] Add VC3D CP state with optional `segmentToNext` metadata and mutation
  helpers.
- [x] Add version-2 object-valued CP serialization and version-1 loading.
- [x] Update every C++, Python, CLI/probe, import/export, and sync/merge reader
  to parse and validate CP-owned segment metadata.
- [x] Make accepted trace completion finalized and atomically attach its
  record.
- [x] Protect valid traced ranges in existing-line and full-reinitialization
  Lasagna paths.
- [x] Apply scoped invalidation/remapping to every CP mutation path.
- [x] Add transactional Ctrl-right-click reversion to Lasagna optimization.

## Validation And Documentation

- [x] Add segment-state, version-1 compatibility, version-2 round-trip, and
  strict cross-reader schema tests.
- [x] Add existing-line and full-reinitialization protection tests.
- [x] Add pure state tests for CP mutation and persistence; compile the live
  trace/save/reload/revert and menu paths into VC3D.
- [x] Build focused tests and VC3D with `-j32`.
- [x] Run focused test binaries and relevant broader regressions.
- [ ] Perform the manual VC3D trace/edit/reload/revert smoke test.
- [x] Update `planning/specs.md`.
- [x] Update code-structure and VC3D fiber JSON documentation.
- [x] Update `planning/changelog.md`.
- [x] Replace `planning/task_log.md` for this task.
