# Per-Segment Interpolation Goals And Cubic-Spline Fallback Status

## Planning

- [x] Capture the requested schema, fallback, grouping, GUI, and threshold behavior.
- [x] Inspect current segment ownership, native fallback, Lasagna failure, and menu paths.
- [x] Write the implementation and validation plan.
- [x] Review the plan against the current fiber schema and solver APIs.
- [x] Confirm distance, per-span Lasagna fallback, and global-mode scope.
- [x] Add persisted metric/message and visible-span label requirements.
- [x] Add the compact per-span actual-mode marker requirement.

## Implementation

- [x] Generalize persisted segment metadata to goal and actual interpolation modes.
- [x] Add the shared joint cubic-spline interpolator.
- [x] Add the grouped interpolation/fallback coordinator.
- [x] Integrate CP edits, global changes, persistence, and auxiliary readers.
- [x] Replace segment trace/revert actions with the goal submenu.
- [x] Persist and render per-span metrics/messages with viewport-aware packing.

## Validation And Documentation

- [x] Add core spline, coordinator, persistence, and generated-view regressions.
- [x] Run focused C++ and Python tests and build VC3D with `-j32`.
- [x] Update specs, docs, changelog, and task log.
- [x] Review and stage all task changes.
