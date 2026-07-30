# VC3D Persistent Fiber-Traced Segments

## User request

Fix VC3D line annotation so a successfully native-fiber-traced CP-to-CP
segment is not silently replaced by a full Lasagna optimization during
auto-save and is not changed by edits to unrelated neighboring control points.

Store native fiber-trace information on the CP that starts the traced segment
in line order. Each CP may carry an optional `segment_to_next` value describing
the CP-to-next-CP span. The information persists in the VC3D fiber JSON and
remains valid while that adjacency is unchanged.

Add a Ctrl-right-click context-menu action on a native-fiber-traced segment to
revert that segment explicitly to ordinary Lasagna optimization.

## Required behavior

1. A successful native fiber trace is immediately saveable as the final
   geometry. Auto-save and close must not run an implicit full Lasagna pass
   over it.
2. A CP's optional `segment_to_next` persists optimizer kind, tracer/config
   version, effective trace configuration, selected data sources, trace scale,
   and accepted endpoint error in base voxels. It does not duplicate endpoint
   coordinates.
3. Ordinary local and full Lasagna optimization preserves every valid traced
   segment bit-exactly.
4. Moving either endpoint clears the starting CP's `segment_to_next`. Deleting
   either endpoint removes or clears it naturally with the affected CP
   adjacency.
5. Inserting a CP inside a traced span invalidates that span because it changes
   the segment definition. Inserting, moving, or deleting a CP outside the
   span preserves the traced segment and only remaps runtime indices.
6. Ctrl-right-click on a traced CP-to-CP span offers `Revert segment to
   Lasagna optimization`. Revert removes protection only if the local Lasagna
   replacement succeeds; failure leaves both geometry and metadata unchanged.
7. Fibers without CP segment metadata continue to load and behave as ordinary
   Lasagna-optimized fibers.
8. Add automated coverage for trace/save, persistence/reload, unrelated CP
   edits, endpoint invalidation, optimizer protection, and explicit revert.
9. Pack CP geometry and `segment_to_next` into one control-point JSON object.
   Bump the fiber schema version and update every in-repository reader so
   malformed or unsupported segment information fails loudly rather than
   being silently ignored.

## Scope

- VC3D line-annotation session and stored-fiber state.
- VC3D fiber JSON read/write/import/export handling.
- Shared Python fiber parsing, native C++ fiber CLI/probe parsing, and VC sync
  and merge handling of the new CP object schema.
- Shared Lasagna optimizer support needed to preserve protected spans.
- Generated-line Ctrl-right-click menu plumbing.
- Focused C++ tests, specifications, code-structure documentation, task log,
  status, and changelog.

## Out of scope

- Changing native 3D fiber tracing search, scoring, coordinate conversion, or
  the fixed 20-base-voxel acceptance threshold.
- Adding a generic annotation history or undo stack.
- Protecting arbitrary user-selected line ranges that were not produced by
  the native fiber tracer.
- Changing unrelated fiber geometry, atlas, or branch-link semantics.

## Correctness constraints

- In line order, CP `i` owns the optional optimization information for segment
  `i -> i+1`; the last CP cannot own a following segment.
- New files use `vc3d_fiber` version 2 and object-valued control points. Every
  version-2 reader must parse and validate `segment_to_next`; treating a CP
  object as an unknown extension is not allowed.
- Version-1 array-valued control points remain readable as ordinary CPs so
  existing fiber annotations are not discarded. Writers emit version 2.
- Metadata moves with its owning CP when CPs are sorted or remapped. Do not
  maintain a separate endpoint-signature registry.
- Line-point ranges remain runtime values derived from the owning CP and its
  current successor.
- A protected span includes both endpoint samples and every stored line sample
  between them.
- Protection and explicit revert must use shared optimizer APIs; do not copy
  optimizer logic into the controller.
- Applying a trace or revert is transactional: failed work must not partially
  update geometry, protection records, or saved state.
- Physical voxel size remains reporting-only and is not part of record
  validity or optimization acceptance.
