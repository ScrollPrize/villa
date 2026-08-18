# Status: rolling live OME-Zarr input cache for shared 3D inference

- [x] Capture the approved live-cache behavior in `task.md`.
- [x] Inspect current bulk download, automatic download, shared tile scheduling,
  TensorStore read-ahead, manager launch, and source-support behavior.
- [x] Write implementation, testing, spec, docs, and changelog plan.
- [x] Obtain and incorporate independent review against task/spec/plan.
- [x] Extract/reuse shared selected-level download/cache primitives.
- [x] Implement bounded live tile materialization and authoritative source support.
- [x] Implement conservative whole-Z-plane eviction and accounting.
- [x] Integrate Fiber, Lasagna, manager, progress, and provenance.
- [x] Add regression tests and update specs/docs/changelog/task log.
- [x] Run focused validation and report limitations/performance evidence.
