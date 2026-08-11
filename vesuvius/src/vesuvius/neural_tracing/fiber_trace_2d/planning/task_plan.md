# Plan: dense-fiber failure replay

## Command and inputs

1. Add a C++ `fiber-replay` subcommand to `vc_fiberlets`, leaving the existing
   `anchors` and `paths` commands intact. The command takes the fiber-prediction
   manifest, strict VC3D fiber JSON, and output directory as positional inputs,
   plus the required normal manifest. All spatial CLI arguments use base
   voxels. The short replay controls are `--fail 20`, `--after 100`,
   `--along 512`, `--radius 128`, and `--match-refine 1`, where the final value
   is a multiple of the nominal base-coordinate trace step on either side of
   the predicted reference arclength.
2. Extract the native trace-option parser, defaults, validation, and JSON
   formatter currently private to `vc_fiber_trace_metric` into a shared C++
   CLI/helper module used by both tools. Do not duplicate it and do not invoke
   Python. Replay unconditionally forces the regular native tracer into greedy
   mode (`beam_width=1`, `beam_lookahead=1`) and rejects conflicting beam CLI
   options. Record both requested trace settings and the replay-effective
   settings, including forced values.
3. Open two explicit sets of data bindings:
   - replay tracing uses `FiberPredictionFieldBindingMode::TraceOptions`, the
     tracer's `trace_to_base` scale, and a Lasagna normal sampler whose
     `workingToBaseScale` is exactly `trace_to_base`;
   - anchor/path extraction uses `CanonicalStoredGrid`, its independent
     `prediction_to_base` scale, and the normal sampler/binding required at that
     scale.
   Never reuse one field or sampler for the other. Store both scales, resolved
   manifests, channel bindings, and effective configurations in the bundle.

## Reference and local correspondence

4. Load the fiber using strict `loadFiberJson`. Preserve the original dense
   `line_points` and their indices. Begin at the first control point's exact
   dense-line index and use the first subsequent nonzero finite edge as the
   initial forward direction. Reject a reversed CP order or a reference with no
   usable forward edge. For arclength calculations only, skip consecutive
   zero-length edges; never globally deduplicate repeated vertices because a
   valid fiber can return to an earlier coordinate.
5. Extract only genuinely shared polyline operations from existing private C++
   implementations: cumulative length over nonzero consecutive edges, exact
   sampling/tangent at an arclength, interval slicing with interpolated
   endpoints, and exact closest point over a bounded arclength interval.
   Preserve source segment/vertex indices and deterministic ties (lowest
   admissible arclength, then original segment index). Migrate existing callers
   only where their semantics are identical; record any helper that must remain
   specialized rather than copying it.
6. Port the reference-correspondence behavior to C++ and make it the single
   replay failure rule. The initial cursor is the first CP arclength. For every
   newly committed greedy trace point:
   - predict `candidate_arc = previous_arc + nominal_step_base`;
   - search the exact dense polyline only in the monotone interval
     `[previous_arc, candidate_arc + match_refine * nominal_step_base]`, with
     the direct-search refinement initialized at `candidate_arc`;
   - select the exact closest point in that interval and advance the cursor to
     its arclength;
   - use the Euclidean base-XYZ distance from trace point to that matched point
     as the error.
   The default factor of one therefore permits an advance of zero through two
   nominal steps, corrects moderate speed drift, and cannot jump backward or to
   an unrelated winding outside the local forward window. Store predicted arc,
   matched arc/point, window, raw error, and `error/fail`. Failure is the first
   raw error strictly greater than `--fail`. The existing Python code supplies
   arclength/interpolation reference behavior but has no equivalent continuous
   3D failure cutoff; therefore retain the established 20-base-voxel default
   rather than claiming a nonexistent Python threshold rule.

## Greedy replay lifecycle

7. Add a typed `FiberReplayTraceRequest/Result` beside the native tracer API.
   Reuse `traceOneWayCore` candidate generation, scoring, normal-aware terms,
   validity rules, and chosen-point/loss publication. Add a committed-step
   observer/stop condition only; with no observer, existing call results must
   remain bit-identical. Replay must use normal fixed-length greedy steps and the
   explicit hard budget
   `ceil(max_step_factor * remaining_reference_arc / nominal_step_base) +
   after + 1` (the shared trace default keeps `max_step_factor=3`); it must not
   use the extrapolation distance endpoint code that emits a shortened last
   step.
8. Evaluate correspondence after the single greedy winner is committed and
   before publishing progress. The failing committed point is postroll step
   zero; accept at most exactly `--after` further committed points. Once failure
   is found, reference matching stops and postroll may continue past the
   reference end. Before failure, if the next matching window has no forward
   reference extent, discard that overshooting candidate and finish
   `no_failure`. Initial invalid prediction is converted into a typed replay
   diagnostic rather than escaping as the ordinary tracer exception.
9. Use exhaustive statuses:
   `failure_with_postroll`, `failure_truncated`, `no_failure`, and
   `trace_terminated_before_failure`. A truncated failure records whether the
   cause was invalid prediction/edge, volume boundary, native hard budget, or
   another native stop reason. Store the full committed trace and cumulative
   losses, failing trace index, predicted and matched reference data, threshold,
   requested/completed postroll, and native reason. Only failure statuses run
   local extraction; nonfailure statuses still produce a diagnostic bundle and
   replay geometry with extraction artifacts explicitly absent.

## Tube-scoped anchors and paths

10. Center the selected interval on the failure's locally matched reference
    arclength and clamp it to
    `[failure_arc-along, failure_arc+along]`, inserting exact endpoints. Define
    tube membership as Euclidean distance `<= radius` from the exact interpolated
    reference interval, including endpoint caps. A prediction cell is selected
    when the base-coordinate AABB of its prediction-sample voxel footprint
    (sample center plus/minus half `prediction_to_base`) has exact minimum
    distance `<= radius`. Clip to canonical grid coverage. Derive the viewer's
    half-open integer base-XYZ crop using documented floor/ceil rules over the
    tube bounds; the explicit cell set, not this enclosing crop, is the anchor
    extraction domain.
11. Generalize anchor selection to either the existing `box` or canonical,
    sorted, unique, in-grid `cells_zyx` plus exact bounds/count. The two forms
    are mutually exclusive in the strict experimental version-1 schema, with
    no compatibility repair. Block sampling may still read halo/context. Reject
    refined anchors outside the authoritative tube before NMS so they cannot
    suppress valid in-tube anchors. External NMS-context anchors may suppress
    selected anchors exactly as in standalone extraction, but are never emitted;
    report both rejection classes separately.
12. Extract one canonical corridor enumeration shared by preload and DP solve.
    For replay, intersect candidate integer DP nodes and virtual endpoints with
    the authoritative tube using base-coordinate voxel-center distance to the
    exact interval; every stored path point must pass the same check. Include
    all trilinear prediction and normal-interpolation dependency voxels needed
    by those admissible nodes, even where that sampling halo extends immediately
    outside the tube. Union and sort the dependency keys, sample each once, and
    expose a checked immutable sparse lookup. Keep standalone `paths` on its
    existing dense preload backend, but make both backends and DP consume the
    same enumerator. Prove identical path bytes and costs when the sparse domain
    includes the full ordinary corridor, plus deterministic rejection when a
    replay corridor is disconnected by the tube.
13. Run the existing refined anchor fitting, NMS, candidate shell, DP objective,
    path quality, and quality-color data unchanged on the selected anchors.
    Replay omits the standalone central-slice OBJ products because napari loads
    the actual external Zarr; standalone `paths` output remains unchanged.

## Atomic replay bundle

14. Make bundle JSON authoritative for reference/trace/failure geometry. Write
    derived, strict base-XYZ OBJ files (`reference.obj` and `trace.obj` each
    containing ordered `v` plus one `l` record; `failure.obj` containing one `v`
    plus one `p` record only for failure statuses). Readers verify hashes and
    verify parsed OBJ geometry equals the authoritative JSON arrays/point.
15. Publish each run through a staging directory. Write and hash the complete
    status-applicable artifact set, atomically rename it to
    `runs/<content-hash>/`, then atomically replace root `fiber_replay.json`
    whose relative artifact paths point into that immutable generation. An old
    bundle therefore continues to reference a complete old generation if a
    rerun fails. Old generations may remain. Nonfailure statuses require null
    anchor/path/failure-marker fields; failure statuses require
    `anchors/anchors.json`, `anchors/anchors.obj`, `anchors/anchors_0.obj`,
    `anchors/anchors_1.obj`, `paths/fiberlets.json`, `paths/fiberlets.obj`,
    `replay/reference.obj`, `replay/trace.obj`, and `replay/failure.obj`.
16. Version-1 bundle fields include source locators/hashes, coordinate order and
    base units, trace/canonical scales and bindings, requested/effective configs,
    status/termination, authoritative reference interval and trace arrays,
    complete matching diagnostics, tube/cell counts, the half-open
    `volume_crop_base_xyzwhd`, artifact hashes, and normalized relative paths.
    Do not store or infer the fiber-presence Zarr locator. Strict readers reject
    absolute paths, lexical `..` escapes, canonical/symlink escapes, status/path
    inconsistencies, hashes or OBJ geometry mismatches, and malformed fields;
    no repair or backwards compatibility is added for this unshipped format.

## Napari replay mode

17. Extend `view_fiber_presence.py` with `--replay <fiber_replay.json>`, while
    retaining the fiber-presence Zarr as the separate positional input.
    Replay mode gets its crop/overlays from the bundle and rejects `--crop`,
    `--anchors`, and `--paths`; manual mode continues to require `--crop`.
    `--level` remains allowed. Validate the chosen external Zarr level's shape
    and scale against bundle prediction metadata without comparing/storing its
    path.
18. Add independently toggleable presence, selected reference, greedy trace,
    failure marker, anchors, and fiberlet layers. Apply all six crop sliders to
    every layer, expose width controls for all line layers and a failure-point
    size control, and retain fiberlet quality features and runtime colormap
    selection. Statuses without extraction/failure artifacts load only their
    applicable layers.

## Tests and validation

19. Add C++ geometry and matching tests for Python-generated arclength fixtures,
    exact vertices, consecutive zero edges, nonconsecutive repeated vertices,
    deterministic ties, differing trace/reference speeds on identical geometry,
    self-near/crossing fibers, bounded correction, monotonicity, strict threshold
    equality, and the exact match-window boundary.
20. Add native replay tests for the first-CP start/tangent, trace/canonical scale
    mismatch, forced greedy settings, invalid initial samples, monitored versus
    unmonitored prefix point/loss identity, reference-end overshoot, exact
    postroll count, every stop/status reason, and no-failure null artifacts.
21. Add extraction tests for curved and zero/one-segment intervals, endpoint
    caps, just-inside/outside cell footprints, floor/ceil crop bounds, stable cell
    order, outside-tube rejection before NMS, external context suppression, and
    diagnostics. Add replay-path tests for in/out endpoint and node centers,
    tube-disconnected corridors, complete node/attachment/interpolation
    dependencies, one sample per key, checked misses, deterministic parallel
    results, full-corridor dense parity, and a bent-tube memory reduction
    fixture.
22. Add artifact and viewer tests for every status-specific layout, hashes,
    deterministic output, failed-rerun preservation of the prior root bundle,
    JSON/OBJ equality, lexical and symlink escapes, missing files, replay/manual
    CLI conflicts, external Zarr metadata mismatch, crop adoption, clipping,
    layer properties, width/size controls, and quality colormap data without a
    Qt session.
23. Build with 32 jobs and run `test_fiber_trace3d`, anchor/path/replay C++
    suites, the `vc_fiberlets` CLI smoke tests, focused viewer pytest, Ruff,
    Python compile checks, and diff hygiene. Run one small real replay fixture
    when the manifests/fiber are available and report the exact command, build
    type, trace/failure/postroll counts, error, selected cells, sparse samples,
    anchors/fiberlets, phase times, and estimated/peak preload memory.

## Spec update

- Add the C++-only greedy replay start, local monotone dense-reference matching,
  failure threshold, postroll, hard-budget, scale-binding, and status semantics.
- Define base-coordinate tube/cell/path membership, pre-NMS filtering, replay
  corridor clipping, sparse preload invariants, and experimental schema policy.
- Define bundle authority, exact status-dependent artifact layout, hashes,
  generation-based atomic publication, path containment, and external-Zarr
  napari contract.

## Docs updates

- Extend `volume-cartographer/docs/fiberlets.md` with `fiber-replay`, defaults,
  local matching behavior, statuses, tube extraction, artifacts, and the direct
  napari command.
- Update fiber-tracing code-structure documentation for the shared polyline and
  trace CLI helpers, the committed-step replay observer, separate scale
  bindings, and sparse corridor preload.

## Changelog update

- Record C++ dense-fiber replay, tube-selected anchor/fiberlet diagnostics,
  sparse corridor preload, atomic replay bundles, and napari replay loading.
