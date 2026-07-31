# Per-Segment Interpolation Goals And Cubic-Spline Fallback Status

## Planning

- [x] Capture the requested schema, fallback, grouping, GUI, and threshold behavior.
- [x] Inspect current segment ownership, native fallback, Lasagna failure, and menu paths.
- [x] Write the implementation and validation plan.
- [x] Review the plan against the current fiber schema and solver APIs.
- [x] Confirm distance, per-span Lasagna fallback, and global-mode scope.
- [x] Add persisted metric/message and visible-span label requirements.

## Implementation

- [ ] Generalize persisted segment metadata to goal and actual interpolation modes.
- [ ] Add the shared joint cubic-spline interpolator.
- [ ] Add the grouped interpolation/fallback coordinator.
- [ ] Integrate CP edits, global changes, persistence, and auxiliary readers.
- [ ] Replace segment trace/revert actions with the goal submenu.
- [ ] Persist and render per-span metrics/messages with viewport-aware packing.

## Validation And Documentation

- [ ] Add core spline, coordinator, persistence, and GUI regressions.
- [ ] Run focused C++ and Python tests and build VC3D with `-j32`.
- [ ] Update specs, docs, changelog, and task log.
- [ ] Review and stage all task changes.
