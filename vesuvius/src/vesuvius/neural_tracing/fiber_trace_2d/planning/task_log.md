# Strict VC3D Fiber V3 Parsing And Consumer Completion

## Discovery

- The canonical VC3D writer emits v3, but the GUI loader currently fills a
  missing non-final descriptor and defaults a missing mode for every version.
- The shared C++ helper validates descriptors only when present.
- The Vesuvius Python reader and sync validator likewise allow missing v3
  descriptors; sync also does not enforce every positive/config bound.
- Lasagna atlas, Atlas constraints export, Atlas inspection, and Spiral still
  contain numeric-only control-point readers.
- `vc_lasagna_line_probe` replaces `line_points` and increments generation
  while retaining stale v3 segment descriptors.
- The required `control_point_segments` field was added to `Vc3dFiber`, but the
  NML constructor and multiple test factories were not updated. The focused
  existing NML test fails with the missing-constructor-argument error.
- Stored endpoint acceptance thresholds are historical base-voxel values and
  require no scaling change.

## Plan Review

- Reviewed the plan against the current v3 persistence, per-span atomicity,
  provenance, legacy-v1, sync, Atlas, and NML requirements in `specs.md`.
- The plan keeps missing-mode defaulting only for v1 and makes v3 complete and
  non-repairing in every consumer.
- It explicitly keeps schema errors separate from the existing branch-link
  repair workflow.
- It avoids a second probe serializer by planning a shared format DTO/helper.
- Cross-subproject Python packaging may prevent every runtime from importing
  one implementation without adding an undesirable heavyweight dependency.
  The plan therefore requires format-only extraction where feasible plus one
  shared conformance corpus for any unavoidable thin package adapter. Any
  actual duplication or packaging deviation must be recorded here before
  implementation.

## Process Deviation

- Independent subagent review was not used because the active collaboration
  policy prohibits delegation unless the user explicitly requests subagents.
  A local review was performed instead.

## Implementation

- Added the shared C++ `Vc3dFiberJson` reader in `FiberJson.hpp`. Version 3 now
  requires `optimization_mode`, one complete descriptor on every non-final CP,
  no descriptor on the final CP, exact descriptor/config fields, valid enums,
  finite values, positive scales, and mode-consistent diagnostics.
- VC3D validates with that reader before constructing a session. Descriptor
  materialization now runs only for legacy v1; invalid v3 files are never
  tagged or repaired.
- Migrated the native metric loader, Atlas source loader, Atlas constraints
  exporter, Atlas inspector, and Lasagna probe to the shared C++ reader.
- Extracted dependency-light Python validation into the installable
  `vc3d_fiber_format` package. The NumPy wrapper, Lasagna atlas, and Spiral use
  it. Lasagna and Spiral have thin monorepo path adapters so their existing
  standalone commands and subprocess launches can locate the package without
  importing the training stack.
- Added one legacy-v1 span factory and used it for JSON/NML construction. Kept
  `Vc3dFiber.control_point_segments` required and updated every direct test
  constructor explicitly.
- Tightened sync's existing non-throwing boundary validator to require complete
  v3 mode/descriptors and all current positive/non-negative config constraints.
  Invalid base/local/remote documents continue into the existing manual
  conflict result without normalization or mutation.
- Reworked optimizing Lasagna probe output to preserve exact CPs and goals,
  emit actual Lasagna descriptors, record the input manifest and per-span
  maximum normal-alignment metric, clear trace diagnostics, increment geometry
  and metadata together, validate the result, and rename atomically. Plain
  `--output` copies the validated input without changing producer metadata.
- Extracted normal-alignment degree conversion into the shared
  `vc/lasagna/NormalAlignment.hpp` helper used by VC3D and the probe.

## Plan Deviations And Limitations

- The planned single on-disk C++/Python conformance corpus was not introduced.
  Equivalent malformed/valid cases are covered in the existing C++ and Python
  test suites instead; sharing fixture files across their independent build and
  packaging roots would have expanded this task beyond the format change.
- Probe serialization has helper-level strict reload coverage and all affected
  targets compile, but no end-to-end probe optimization fixture was run because
  the test suite has no small committed Lasagna dataset for that executable.
- Sync retains its existing local validator instead of importing Vesuvius. It
  is a CLI boundary that must remain runnable from `volume-cartographer/scripts`
  and return `False` rather than throw; its rules and regression cases were
  updated to match the canonical format readers.

## Validation

- `cmake --build volume-cartographer/build -j32 --target VC3D
  vc_fiber_trace_metric vc_lasagna_line_probe vc_atlas_inspect
  vc_atlas_constraints_export`: passed.
- C++ `test_line_annotation_generated_views`: 56 cases passed after rebuilding
  with the final strict-reader and serializer assertions.
- Python format/Lasagna/Spiral helper/sync focused batch: 153 tests and 5
  subtests passed.
- Python NML selection: 4 tests passed.
- Python native 3D tracing suite: 179 passed, 2 skipped.
- Spiral service suite with loopback permission: 95 tests and 17 subtests
  passed.
- Sync and merge suites: 146 tests passed.
- Standalone `PYTHONPATH=lasagna` atlas import and direct Spiral service
  `--help` launch passed.
- The C++ Atlas suite has three unrelated existing failures: its pred-snap
  fixtures reference a non-3D `nx` channel. It also reports that the external
  Atlas 21 fixture has obsolete metadata version 4.
