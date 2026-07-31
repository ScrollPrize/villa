# Plan: Strict VC3D Fiber V3 Parsing And Consumer Completion

## Contract And Failure Semantics

1. Define one explicit version matrix and use it in every reader:
   - version 1 has numeric control points, does not require segment descriptors,
     and defaults a missing `optimization_mode` to `lasagna`;
   - version 3 has object control points, requires a valid
     `optimization_mode`, requires one complete descriptor on each non-final
     control point, and forbids a descriptor on the final control point;
   - top-level version 2 and all obsolete descriptor schemas remain errors.
2. Keep the existing strict v3 descriptor rules: exact descriptor/config keys,
   current `(metadata_version, tracer_version) == (3, 2)`, finite values,
   positive scales, valid enums, and mode-consistent metrics/diagnostics.
3. Treat missing v3 mode or descriptors as format errors. Readers must not
   create defaults, tag for reoptimization, mutate the source JSON, or offer a
   repair action. A multi-fiber loader may skip the invalid file and report its
   precise error while leaving every file unchanged.
4. Keep branch-link repair separate. This task removes schema repair for v3;
   it does not change the existing explicit workflow for repairing otherwise
   valid fibers with broken reciprocal branch links.
5. Preserve consumer-specific geometry requirements after common schema
   validation. For example, the metric runner may still require at least two
   CPs while the GUI may represent a one-seed session.

## Shared Format Validation

1. Extend `vc/fiber_tracer/FiberJson.hpp` from a point-array helper into the
   common C++ format validator/reader used by Atlas, the native metric runner,
   the Lasagna probe, Atlas constraint export, and Atlas inspection.
2. Return validated version, optimization mode, line positions, control-point
   positions, and the raw/typed CP-owned descriptors needed by callers. Keep
   GUI-independent JSON-format types in core; adapt them to VC3D's runtime
   interpolation structures at the app boundary.
3. Make `LineAnnotationController::loadFiberJson` invoke the common strict v3
   validation before building a session. Remove its loop that synthesizes
   missing v3 descriptors. Retain explicit legacy-v1 materialization only.
4. Keep the Vesuvius Python fiber-format parser as the authoritative Python
   implementation and split format-only validation/extraction from NumPy and
   training-specific wrappers so geometry-only callers do not import the model
   or dataset stack.
5. Use a shared conformance corpus of valid v1/v3 documents and malformed v3
   cases across C++ and Python tests. The Lasagna and Spiral packages have
   independent runtime/package boundaries, so any unavoidable thin adapter
   must consume the same corpus and expose the same errors rather than grow a
   separate permissive interpretation.

## Complete Remaining V3 Consumers

1. Replace the numeric-only control-point loops in `lasagna/atlas.py` with the
   strict format-only Python reader. Feed only the validated CP positions into
   the existing atlas target construction.
2. Replace the duplicate numeric readers in `AtlasConstraints.cpp` and
   `vc_atlas_inspect.cpp` with the common C++ reader already used by the main
   Atlas path.
3. Change Spiral fiber loading to validate the complete document first and
   then extract v1 numeric or v3 object CP positions. Preserve its existing
   scaling, decimation, ordering, color, and point-collection output.
4. Apply the same strict validator at the Spiral service upload boundary so an
   invalid v3 upload fails immediately instead of being accepted and failing
   later when a fitting session starts.
5. Audit every remaining `control_points` reader after these changes. Any
   geometry-only v3 consumer must still validate the full v3 schema before
   discarding metadata it does not use.

## Correct Lasagna Probe Output

1. Move/reuse the v3 descriptor JSON DTO, validation, and serialization logic
   needed by both VC3D and `vc_lasagna_line_probe`; do not add another raw JSON
   descriptor implementation in the probe.
2. Parse the input strictly before optimization. Invalid v3 input aborts and
   no output file is created or replaced.
3. When `--reopt` or `--reinit-reopt` changes the dense line, build a new
   canonical v3 document atomically:
   - preserve exact control-point positions and each existing v3
     `interp_goal`; legacy-v1 spans receive goal `global`;
   - set every successfully regenerated span's actual `interp_mode` to
     `lasagna`;
   - record the input Lasagna manifest identity in `normal_manifest` and clear
     `fiber_manifest`, because the probe did not consult fiber predictions;
   - compute the same per-span maximum normal-alignment error in degrees used
     by VC3D and store it as `metric`;
   - write an accurate compact `msg`, clear trace meeting/failure fields, and
     record Lasagna failure detail only for a real failed span;
   - update dense geometry, descriptors, and generation as one result.
4. A failed reinitialization must keep the existing no-output behavior. A
   copy/output operation that does not change geometry must not invent a new
   producer or stale diagnostic state.
5. Validate the generated document with the same strict reader before the
   atomic rename.

## Repair Python Fiber Construction

1. Keep `Vc3dFiber.control_point_segments` required so omitted integration work
   continues to fail loudly.
2. Extract one legacy-v1 segment tuple factory and use it from JSON and NML
   loading. NML paths receive explicit global/Lasagna span state and a final
   `None`, matching legacy JSON semantics.
3. Update every direct `Vc3dFiber(...)` constructor in production code and
   tests to provide an explicit tuple. Do not add a dataclass default that
   could hide future format extensions.
4. Verify NML ordering, connected-component splitting, affine transform, 2D
   loading, and 3D whole-fiber trace tests after the repair.

## Sync Strictness

1. Tighten `fiber_merge.is_fiber_doc` for version 3:
   - require `optimization_mode` and validate its enum;
   - require a descriptor on every non-final CP and forbid one on the final CP;
   - enforce positive `trace_to_base_scale`, positive/non-negative config
     bounds, and the same mode-specific consistency rules as the canonical
     readers.
2. Keep version-1 defaults and merge behavior unchanged.
3. Route malformed/incomplete v3 input to the existing manual-conflict path.
   Do not normalize it, construct placeholder geometry, or choose one side by
   generation.
4. Add tests proving local, remote, and base invalidity are all non-destructive
   conflicts and that valid v3 atomic span merging remains unchanged.

## Tests And Validation

1. Add table-driven conformance cases covering:
   - valid v1 without mode;
   - valid complete v3;
   - missing v3 `optimization_mode`;
   - missing first/interior descriptor;
   - descriptor on the final CP;
   - obsolete versions, unknown enums/keys, non-finite values, non-positive
     scales, invalid config bounds, and inconsistent mode diagnostics.
2. Run the C++ cases through the GUI loader adapter, shared core reader, Atlas
   source reader, Atlas constraints exporter, Atlas inspector helper, metric
   loader, and Lasagna probe serializer where practical.
3. Add Lasagna atlas and Spiral tests with complete mixed-mode v3 fixtures and
   malformed v3 fixtures. Confirm extracted positions and existing downstream
   ordering/scaling are unchanged.
4. Add probe output tests that split the optimized line at exact CPs and verify
   goal preservation, actual Lasagna mode, metric units, manifest provenance,
   cleared trace fields, generation, and strict reload.
5. Run:
   - Vesuvius fiber JSON, NML loader, 2D loader, and 3D tracer tests;
   - Lasagna atlas tests;
   - Spiral helper/service tests;
   - `fiber_merge.py` and `vc_sync.py` tests;
   - C++ Atlas, line-annotation, fiber-tracer, and Lasagna optimizer tests.
6. Build affected production targets with all 32 threads:
   `VC3D`, `vc_fiber_trace_metric`, `vc_lasagna_line_probe`,
   `vc_atlas_inspect`, and `vc_atlas_constraints_export`.
7. Run `git diff --check`, search again for direct `control_points` parsing,
   and verify no invalid-v3 test modifies its input file.

## Spec Update

- Change the v3 contract from an optional descriptor field at the JSON object
  level to a required descriptor on every non-final CP, with the final CP as
  the sole allowed omission.
- State that `optimization_mode` is required for v3 but remains optional with
  a Lasagna default for v1.
- State that every parser, including geometry-only tools and sync, validates
  the complete v3 contract and performs no v3 schema repair.
- Document canonical probe output as a new coherent v3 Lasagna result rather
  than an input document with replaced geometry.

## Docs Updates

- Update `volume-cartographer/docs/line_annotation_fibers.md` with the strict
  version matrix, failure behavior, and probe output semantics.
- Update `docs/code_structure.md` with the shared C++/Python format validation
  paths and the completed Atlas, Spiral, NML, and sync consumers.
- Document the Python format-only helper's package boundary if extraction
  changes how Lasagna or Spiral imports it.

## Changelog

- Record strict, non-repairing v3 parsing across VC3D, CLI, Python, Atlas,
  Spiral, and sync; coherent Lasagna probe v3 output; and restored NML loading.

## Out Of Scope

- Do not change trace thresholds or coordinate scaling. Persisted trace config
  remains historical data expressed in base voxels.
- Do not change v1 merge behavior, branch-link repair, interpolation
  algorithms, tracing, Lasagna optimization, or GUI layout.
