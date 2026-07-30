# Task Plan: Persistent Fiber-Traced Segments And Explicit Lasagna Revert

## 1. CP-Owned Data Contract

### 1.1 Attach following-segment state to each VC3D control point

Add a VC3D control-point state type that contains the existing
`vc::lasagna::LineControlPoint` plus optional `segmentToNext` metadata. In line
order, CP `i` owns the state for span `i -> i+1`; the last CP must have no
`segmentToNext` value.

The optional value contains:

- `optimizer: "native_fiber_trace3d"`;
- a metadata schema version and native tracer algorithm/config version;
- the regular-normal and fiber-inference manifest source locations;
- the effective trace-to-base scale;
- the effective `FiberTraceConfig` values that affect geometry/acceptance,
  excluding runtime-only pointers and optional physical-size reporting;
- the accepted maximum endpoint error in base voxels.

Do not persist endpoint signatures or a separate collection of segment
records. The owning CP and its immediate successor are the endpoints. Do not
add tracer fields to `vc::lasagna::LineControlPoint`, because that shared
low-level type is used outside VC3D; convert between VC3D CP state and the
low-level optimizer inputs at the existing task boundary.

Make the ownership explicit in JSON instead of using a parallel array:

```json
{
  "type": "vc3d_fiber",
  "version": 2,
  "control_points": [
    {
      "position": [100.0, 200.0, 300.0],
      "segment_to_next": {
        "optimizer": "native_fiber_trace3d",
        "metadata_version": 1,
        "tracer_version": 1,
        "max_endpoint_error_base_voxels": 3.5
      }
    },
    {"position": [132.0, 204.0, 301.0]}
  ]
}
```

The complete `segment_to_next` object also contains the source locations,
trace scale, and effective config listed above. `line_points` remain numeric
arrays.

Bump `vc3d_fiber` to version 2. Version 2 requires every control point to be an
object with a finite three-element `position`; `segment_to_next`, when
present, must be fully recognized and valid, and it is forbidden on the final
CP. Unknown optimizer kinds, metadata versions, missing required fields, or
wrong types reject the file. This makes semantic reader gaps fail at the
existing version/schema boundary instead of hiding provenance.

Updated readers continue accepting version-1 array-valued CPs as ordinary
unprotected CPs so existing annotations remain usable. All VC3D writers emit
version 2. Old binaries reject version 2 at their existing version check,
which is the intended loud failure.

The persisted endpoint error is the base-voxel value used for acceptance.
Micrometer output remains an ephemeral diagnostic because physical voxel-size
metadata can be absent or can differ in another project context.

### 1.2 Mutation rules follow normal CP container operations

Implement small helpers for CP-owned metadata mutation:

- move CP `i`: clear `segmentToNext` on CP `i-1` and CP `i` because the moved
  CP is the target of the former and source of the latter;
- delete CP `i`: erase its own metadata with the CP and clear the previous
  CP's `segmentToNext`, whose successor changed;
- insert CP at `i`: insert it with empty metadata and clear the previous CP's
  `segmentToNext`, because its old span was split;
- sort/remap CPs without changing geometry: move the complete CP state so the
  metadata stays with its owner;
- edit elsewhere: no action is required for unrelated CPs;
- replace/reseed the complete line: clear all following-segment metadata;
- non-unit import/export scaling: clear all following-segment metadata because
  its base-space trace configuration no longer describes the scaled fiber.

To protect a span, resolve the owning CP and its immediate successor to their
current line indices. No endpoint matching or metadata reconciliation table is
needed. Version-2 schema violations are hard load errors rather than warnings
or best-effort repair.

Expected files:

- new `volume-cartographer/apps/VC3D/LineAnnotationFiberSegments.hpp`
- new `volume-cartographer/apps/VC3D/LineAnnotationFiberSegments.cpp`
- `volume-cartographer/apps/VC3D/LineAnnotationController.hpp`
- `volume-cartographer/apps/VC3D/LineAnnotationController.cpp`
- `volume-cartographer/apps/VC3D/CMakeLists.txt`

## 2. Fix Trace Completion And Auto-Save

Carry a typed pending segment record in `OptimizationTaskResult`; stop encoding
the only trace provenance in `LineOptimizationReport.message`.

On accepted native tracing:

1. Build the replacement line and segment record in the worker.
2. Apply geometry and metadata together on the GUI thread.
3. Mark the result `SessionOptimizationState::Optimized`, because tracing is a
   completed operation for the selected span and the rest of the input line was
   already finalized.
4. Refresh generated views and auto-save that exact state.
5. Confirm `saveSessionAsFiber()` does not call
   `finalizeSessionOptimizationSynchronously()` for this result.

A rejected/failed trace changes neither line points nor CP metadata. Replacing
a previously traced span with a successful new trace overwrites the starting
CP's one `segmentToNext` value.

Expected files:

- `volume-cartographer/apps/VC3D/LineAnnotationController.cpp`
- `volume-cartographer/apps/VC3D/LineAnnotationController.hpp`

## 3. Preserve Traced Geometry During Lasagna Optimization

Resolve every CP with `segmentToNext` and its immediate successor before
starting an ordinary optimization, then pass those closed line-index ranges
through the existing optimization task path.

### 3.1 Existing-line local/global optimization

Extend the shared `LineOptimizer::optimizeExistingLine()` input to accept
protected ranges (or their expanded fixed sample indices). Every sample in a
protected range must be a Ceres constant, not only the endpoint CPs. Keep
active-range boundary handling and CP fixed points unchanged.

### 3.2 Full reinitialization/finalization

Extend `reinitializeAndOptimizeExistingLine()` with protected CP spans. A
protected span must select the existing stored span directly and skip its
Lasagna reinitialization/optimization, while unprotected spans retain current
full-reinitialization behavior. This preserves native-traced geometry even
when save/close finalizes unrelated incremental edits.

Use these shared APIs from async optimization, synchronous finalization, and
the test factory seam. Do not splice protected geometry back into a completed
result in controller-only code; protection must be enforced by the optimizer
that owns point/range changes.

After an optimization result returns, reattach the moved CP state by CP order
and assert in tests that each protected range is bit-exact. If the optimizer
violates that invariant, reject the result instead of silently clearing the CP
metadata.

Expected files:

- `volume-cartographer/core/include/vc/lasagna/LineOptimizer.hpp`
- `volume-cartographer/core/src/lasagna/LineOptimizer.cpp`
- `volume-cartographer/apps/VC3D/LineAnnotationController.cpp`
- `volume-cartographer/apps/VC3D/LineAnnotationController.hpp`

## 4. Make CP Mutations Protection-Aware

Centralize CP-state mutation alongside the existing branch metadata
synchronization hook so every mutation path applies the same rules.

- Existing CP move: clear the previous CP's and moved CP's following-segment
  metadata before local reinitialization.
- CP insertion: insert empty metadata and clear only the previous CP's
  following-segment metadata.
- CP deletion: erase its metadata with it and clear only the previous CP's
  following-segment metadata.
- No-reoptimize mode: apply the same metadata invalidation even though Lasagna
  is not run.
- Auto-reoptimize mode: the current radius-three active range may remain, but
  valid traced ranges inside it are fixed/protected.
- New seed/replacement line: construct CPs with empty metadata.

The generic session state may still become `Unoptimized`/`Incremental` for CP
edits. Save-time finalization then optimizes only unprotected geometry.

Expected files:

- `volume-cartographer/apps/VC3D/LineAnnotationController.cpp`
- `volume-cartographer/apps/VC3D/LineAnnotationFiberSegments.cpp`

## 5. Persist, Reload, And Update Every Reader

Use the CP-owned type in `LineAnnotationSession` and the corresponding stored
CP type in `StoredFiber`. Update all conversion paths:

- session to stored snapshot, using the existing sorted CP order;
- JSON serialization/deserialization;
- stored fiber to a reopened line-annotation session;
- asynchronous save snapshots and in-memory `_fibers` replacement;
- exact-scale bundle export/import;
- version-2 object-valued `control_points` serialization/deserialization;
- non-unit scale import/export invalidation with an explicit warning;
- `scripts/fiber_merge.py`, so whichever geometry carrier supplies a CP also
  supplies its metadata and any synthetically merged CP boundaries are cleared.

Audit every in-repository `vc3d_fiber` reader rather than assuming unknown
fields are harmless. At minimum update:

- VC3D `LineAnnotationController` load/save/import/export;
- shared Python `fiber_trace.fiber_json` parsing and its `fiber_trace_2d`
  re-export, retaining geometry convenience arrays while exposing typed CP
  segment metadata;
- native `vc_fiber_trace_metric` parsing in `vc_fiber_tracer`;
- `vc_lasagna_line_probe` and any other C++ point-array probe found by the
  audit;
- `scripts/fiber_merge.py`, `scripts/vc_sync.py`, and their validators/tests.

Version-1 inputs synthesize CP state with no `segmentToNext`. Version-2 inputs
are rejected if any reader cannot parse all required CP/segment fields. Do not
infer traced state from report strings, endpoint coordinates, or fiber tags.

For sync/merge, compare CP geometry through `position` but carry the complete
CP object from the chosen geometry side. If geometry is synthetically merged
and `line_points` are replaced by the CP polyline, clear all
`segment_to_next` values because none of the traced dense spans survived.

Expected files:

- `volume-cartographer/apps/VC3D/LineAnnotationController.hpp`
- `volume-cartographer/apps/VC3D/LineAnnotationController.cpp`
- `volume-cartographer/apps/VC3D/LineAnnotationFiberSegments.*`
- `volume-cartographer/core/src/fiber_tracer/FiberTrace.cpp`
- `volume-cartographer/apps/src/vc_lasagna_line_probe.cpp`
- `volume-cartographer/scripts/fiber_merge.py`
- `volume-cartographer/scripts/vc_sync.py`
- `vesuvius/src/vesuvius/neural_tracing/fiber_trace/fiber_json.py`

## 6. Ctrl-Right-Click Revert

Reuse the existing Ctrl-right-click span resolver in
`showGeneratedControlPointContextMenu()`.

- For an untraced span, retain `Optimize segment with native fiber tracer`.
- For a valid traced span, show `Revert segment to Lasagna optimization` in
  its place.
- Pass whether the resolved starting CP has `segmentToNext` into the menu
  options and add a typed revert callback through generated views,
  `LineAnnotationDialog`, and `LineAnnotationController`.
- Disable mutation actions while any line task is running, consistent with the
  existing controller guard.

Revert is transactional:

1. Resolve the clicked starting CP, its `segmentToNext`, its successor, and
   their current line indices.
2. Run existing-line Lasagna optimization only on that closed span, protecting
   all other traced spans but excluding the selected record.
3. Keep both endpoints fixed exactly.
4. On success, clear that CP's `segmentToNext`, apply/refresh/save the Lasagna
   result, and mark it finalized so auto-save does not run a second full pass.
5. On failure, keep the old traced geometry and CP metadata and report the
   error.

Expected files:

- `volume-cartographer/apps/VC3D/LineAnnotationGeneratedViews.hpp`
- `volume-cartographer/apps/VC3D/LineAnnotationGeneratedViews.cpp`
- `volume-cartographer/apps/VC3D/LineAnnotationDialog.hpp`
- `volume-cartographer/apps/VC3D/LineAnnotationDialog.cpp`
- `volume-cartographer/apps/VC3D/LineAnnotationController.hpp`
- `volume-cartographer/apps/VC3D/LineAnnotationController.cpp`

## 7. Testing

### 7.1 Pure segment-state tests

Add focused tests for:

- version-1 CP arrays loading as ordinary CPs and version-2 CP objects
  round-tripping all `segment_to_next` fields;
- version-2 malformed positions, final-CP metadata, unknown optimizer kinds,
  and unknown metadata versions failing loudly in C++ and Python;
- CP sorting moving metadata with its owner;
- unrelated insertion/deletion/index shifts preserving metadata;
- source/target move, deletion, and insertion inside the span clearing exactly
  the affected CP metadata;
- malformed length/final-entry/schema metadata being rejected
  deterministically;
- non-unit scale conversion clearing metadata;
- sync merge carrying metadata with its geometry carrier and clearing metadata
  at synthetically merged boundaries.
- native metric and line probe accepting version 2 while extracting the same
  geometry as version 1.

### 7.2 Optimizer tests

Extend `test_lasagna_line_optimizer` to verify:

- local/global existing-line optimization leaves every protected sample
  bit-exact while changing an unprotected sample;
- full reinitialization preserves protected CP spans bit-exact and still
  reinitializes unprotected spans;
- multiple disjoint protected spans and active-range overlap behave correctly.

### 7.3 VC3D lifecycle/menu tests

Extend the headless line-annotation test seams and generated-view tests to
cover:

- accepted trace is marked finalized and auto-save never launches a full
  Lasagna task;
- saved/reopened fibers restore CP-owned protection;
- deleting a CP adjacent to, but not part of, a traced span preserves its line
  samples through the radius-three optimization;
- moving/deleting an endpoint clears only affected `segmentToNext` values;
- Ctrl-right-click chooses trace for an untraced span and revert for a traced
  span;
- failed revert retains metadata/geometry; successful revert clears only the
  selected CP's metadata and saves the Lasagna result.

Keep menu decision logic separately testable without invoking modal
`QMenu::exec()`.

### 7.4 Build and run

Use all 32 requested build threads:

```bash
cmake --build volume-cartographer/build --target \
  test_line_annotation_generated_views \
  test_lasagna_line_optimizer \
  test_fiber_trace3d \
  VC3D -j32
volume-cartographer/build/bin/test_line_annotation_generated_views
volume-cartographer/build/bin/test_lasagna_line_optimizer
volume-cartographer/build/bin/test_fiber_trace3d
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src \
  python -m pytest \
    vesuvius/tests/neural_tracing/test_fiber_trace.py \
    volume-cartographer/scripts/tests/test_fiber_merge.py \
    volume-cartographer/scripts/tests/test_vc_sync_helpers.py
```

Then run the configured broader VC/CTest suite relevant to VC3D/Lasagna if its
runtime remains practical. Record exact commands/results in `task_log.md`.

Manual GUI regression with the user's local project:

1. Trace a span and confirm the saved JSON immediately contains traced line
   points plus one readable `segment_to_next` entry inside the starting CP
   object.
2. Delete a CP adjacent to but outside the span and confirm the traced points
   are unchanged before and after save/reopen.
3. Move one endpoint and confirm only the affected CP metadata disappears.
4. Trace again, Ctrl-right-click the span, revert it, and confirm the record is
   removed and the span changes to Lasagna output.

## 8. Spec Update

Update `planning/specs.md` to define CP-owned following-segment semantics:

- CP `i` owns optional native-trace information for span `i -> i+1`;
- moving a CP clears its own and its predecessor's metadata;
- deletion/insertion clears only metadata whose adjacency changed;
- edits outside the adjacency preserve metadata naturally with the owning CP;
- ordinary local/full optimization protects valid spans;
- native trace completion is final/saveable without an implicit full Lasagna
  pass;
- Ctrl-right-click exposes explicit Lasagna revert for a traced span.

Document version-2 `control_points[].position` and
`control_points[].segment_to_next`, strict reader behavior, version-1 loading,
and base-coordinate units.

## 9. Docs Updates

Update
`vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md`
with:

- the VC3D CP-owned `segmentToNext` model and adjacency invalidation rules;
- optimizer protection flow for local and full Lasagna passes;
- save/reload behavior and the JSON field;
- trace versus revert Ctrl-right-click actions.

Add a concise fiber JSON section to the most relevant VC3D document (or a new
focused `volume-cartographer/docs/line_annotation_fibers.md` if no suitable
document exists) so the persisted schema is documented near the C++ owner.

## 10. Changelog Update

After implementation and validation, add one 2026-07-30 changelog entry:
VC3D now stores native-fiber-traced span information on the starting CP,
protects those spans through ordinary optimization, clears metadata only when
the CP adjacency changes, and supports explicit Ctrl-right-click reversion to
Lasagna optimization.

## 11. Risks And Review Gates

- Full reinitialization currently reconstructs CP spans, so protection must be
  implemented in the shared reinitializer rather than only by adding Ceres
  fixed points to the existing-line path.
- The version bump intentionally requires a repository-wide reader audit. Any
  missed old reader will reject version 2 loudly; the audit and cross-reader
  fixtures are therefore release gates.
- Sync merge must operate on complete CP objects. Comparing or merging only
  their positions would silently detach `segment_to_next` from its owner.
- Independent agent review is required by the nested workflow but cannot be
  run under the active no-delegation policy unless the user explicitly asks
  for sub-agent work. Keep that status open and do not represent the direct
  consistency review as independent.
