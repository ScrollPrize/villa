# VC3D Fiber-Global Tracing Mode Status

## Planning

- [x] Read the todo request and repository instructions.
- [x] Trace current native segment, Lasagna reinitialization, tail generation,
  persistence, and dialog control paths.
- [x] Replace the active task, plan, status, and task log.
- [x] Review the plan directly against current specs and implementation.
- [ ] Obtain independent-agent review (not permitted unless explicitly requested).

## Implementation

- [x] Add persisted fiber-global mode.
- [x] Add dialog mode and extrapolation controls.
- [x] Add shared native endpoint extrapolation.
- [x] Add mixed whole-fiber native/Lasagna task.
- [x] Route mode changes, full rebuilds, and CP edits by mode.

## Validation And Documentation

- [x] Add focused C++ and persistence regressions.
- [x] Build affected targets and VC3D with `-j32`.
- [x] Run focused and broader relevant tests.
- [x] Update specs, docs, changelog, and task log.
- [x] Perform final diff and consistency review.
