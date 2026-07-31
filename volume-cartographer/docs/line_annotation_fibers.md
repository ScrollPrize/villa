# VC3D Line Annotation Fibers

VC3D writes line annotations as `vc3d_fiber` JSON. Version 3 stores
`control_points` as objects with a required `position`. Every non-final control
point owns a required `segment_to_next` descriptor for its span to control point
`i+1`; the final control point cannot contain `segment_to_next`.

The top-level `optimization_mode` is either `lasagna` or
`native_fiber_trace3d`. It is required in version 3; only legacy version-1
files may omit it and default to `lasagna`. The mode
selects extrapolation and resolves CP spans whose persisted `interp_goal` is
`global`. A segment goal is `global`, `cspline`, `lasagna`, or `trace`; its
`interp_mode` is the actual producer of the stored geometry and is one of
`cspline`, `lasagna`, or `trace`. The actual mode is recomputed from the goal
whenever the span is dirty, so a previous fallback is retried rather than
treated as permanent. Newly created fibers default to `native_fiber_trace3d`;
the Lasagna default for older files with no mode remains unchanged.

The fallback order is `trace -> lasagna -> cspline` or `lasagna -> cspline`.
For global Lasagna/trace goals only, endpoint distance below 100 base voxels
selects `cspline` immediately; exactly 100 voxels still attempts the global
mode. Explicit goals never use this shortcut. Adjacent cubic-spline spans are
interpolated jointly with exact CPs, shared internal tangents, and hard boundary
directions from neighboring stored geometry. The spline helper uses no normal
or prediction data.

Trace attempts remain per span. Lasagna candidate failure also demotes only
that span; other usable Lasagna spans continue into the protected joint Ceres
refinement. Trace and cubic-spline spans, plus untouched manual spans, are
fixed during that solve and provide hard endpoint directions. CP edits dirty
only adjacent spans and expand through connected cubic-spline runs. Changing
the global mode retries global goals while initially protecting explicit goals.
Ctrl-right-clicking a generated span opens a checked `Interpolation goal`
submenu for all four goals.

The generated annotation workspace places a compact full-width schematic map
below the toolbar, followed by the current/side cut views and two stacked
volume-rendered strips. The first rendered strip is `lineSurface`; the second
is `lineSideSlice`. The schematic map shows the whole line and control-point
positions, but does not replace either rendered view. Both rendered strips are
ordinary interactive viewers: they can be panned, zoomed, and scrolled, drive
the current cut from mouse position, accept control-point interactions, and
show the per-span status labels described below. Cameras and splitter sizes
survive generated-view updates.

The Annotation popup owns Auto-reoptimize, Reinit reoptimization, and Show as
mesh. The toolbar retains the fiber-global Lasagna/Fiber model selector and the
base-voxel extrapolation-distance control. Tag pills are edited directly from
the same toolbar.

Each direction continues until it reaches all target-local planes within the
20-base-voxel endpoint threshold or exhausts its step budget. VC3D then moves
locally tangent planes along both complete traces and intersects the opposite
trace. It selects the smallest meeting error and accepts it when the error is
at most `max(10 base voxels, 10% of the combined partial traced length)`. This
can succeed even when neither direction reached its endpoint planes. The
accepted partial traces are warped by arc-length fraction to their shared
midpoint, concatenated, and resampled, with the original CP endpoints restored
exactly. Rejected spans display the generic `fiber gap` failure label because
the threshold is no longer ratio-only.

Successful native spans are fixed
during the fallback solve. At each native-adjacent control point, VC3D derives
the tangent from the control point to the first distinct dense native point.
Lasagna geometry on the opposite side is hard-constrained to leave the control
point along the negative of that tangent. VC3D creates and fixes one adjacent
proxy point on that direction, then runs the ordinary Lasagna Ceres solve and
its existing smoothness terms for the remaining points. With one constrained
endpoint, that fiber direction is the span's only rollout candidate. With two,
one rollout is generated from each constrained endpoint. Reinitialization does
not submit the previous Lasagna span or its endpoint directions as candidates;
a direction propagated from an already solved neighbor also replaces generic
CP/chord initialization. Degenerate tangent-plane projection selects a
deterministic perpendicular tangent instead of continuing along the sampled
normal. This applies at both ends of a fallback span when it lies between
native spans.

Every segment descriptor stores a compact `msg` and an optional mode-dependent
`metric`: trace stores minimum meeting-plane error in base voxels, Lasagna
stores maximum normal-alignment error in degrees, and cubic spline stores no
metric. Detailed trace and Lasagna failures remain in their mode-specific
fields. `normal_manifest` stores the Lasagna manifest location used by the
span, and `fiber_manifest` stores the fiber-inference manifest location. A
direct Lasagna span stores only the former; trace stores both because it samples
Lasagna normals; direct cubic spline stores neither. Fallbacks retain the
locations consulted by failed higher-priority attempts.

For ordinary project datasets these values are the configured local or remote
manifest paths. The open-data catalogue has no artifact UUID: it identifies a
Lasagna artifact by public artifact URL plus sample ID, volume ID, coordinate
level, optional model ID, and manifest artifact index. Segment metadata stores
the reconstructed exact public manifest URL, never its local cache path.

Strip labels prefix the actual mode as `C`, `L`, or `T`, then display
the metric and message. Labels are laid out in viewport pixels, remain visible
while any part of their span intersects the view, and use a deterministic
second row when one row cannot avoid overlap. Legacy version-1 fibers remain
readable and acquire explicit version-3 descriptors on the next save. The
unpublished file version 2 and its pre-v3 descriptor schemas are unsupported.
This does not affect the current `tracer_version: 2` stored inside a version-3
segment descriptor.

All v3 readers validate this complete contract before using even geometry-only
data. VC3D, native CLI tools, Atlas, Lasagna, Spiral, Python training loaders,
and sync reject a missing mode, a missing non-final descriptor, a descriptor on
the final CP, malformed config, or inconsistent mode diagnostics. They never
repair, normalize, tag, or rewrite invalid v3 input. Sync routes it to the
manual conflict workflow and keeps local, remote, and base files unchanged.

## Sync Conflict Handling

`scripts/vc_sync.py` compares local and S3 content with the last successfully
synced shadow copy. When both versions changed, `scripts/fiber_merge.py`
performs a three-way merge. Version-3 fibers are divided into complete stored
span results: each result contains its dense CP-to-CP line slice and the
starting CP's complete `segment_to_next` descriptor. Geometry and metadata are
never merged field by field or selected by generation.

A span run changed on only one side is retained verbatim, and identical
two-sided results converge. Separate local and remote changes can be combined
only when at least one complete base span between them is unchanged on both
sides. Adjacent edits, different edits to the same run, overlapping topology
changes, missing ordered CP/line anchors, and inexact joins are conflicts.
`optimization_mode` follows the same base-aware policy: one changed side wins,
equal changes converge, and different two-sided changes conflict.

An ambiguous merge does not modify the fiber. The sync tool stores local,
remote, and base copies under `.s3sync-conflicts/` and asks whether to keep the
complete local version, keep the complete remote version, or skip. Existing
base-aware tag, branch-link, reciprocal-peer, and manual-HV-tag handling runs
only after geometry merges cleanly. Version-1 fibers retain the older merge
behavior, including the CP-polyline `needs_reoptimization`
fallback for disjoint geometry edits.

`vc_lasagna_line_probe --reopt` and `--reinit-reopt` write a coherent new v3
Lasagna result: exact CP positions and existing goals are retained, actual span
modes become `lasagna`, the input normal manifest and per-span maximum normal
alignment error are recorded, trace-only diagnostics are cleared, and geometry
and descriptors receive one generation update. The probe validates this result
before atomically replacing its output. Without either optimization flag,
`--output` copies the validated input without inventing new producer metadata.

The line-annotation extrapolation control is in base voxels. Lasagna mode grows
normal-based tails. Native mode attempts each tail with the shared one-way
fiber tracer after converting the requested distance to trace voxels. That
distance defines `ceil(distance / nominal step)` generations: extrapolation
uses no target planes, ignores `max_step_factor`, and uses the remaining nominal
distance for its final generation. Completing the planned generations is
success; accumulated measured arc length is not consulted.
Stored line and control points always remain in base coordinates. When the
prediction field returns no valid next direction at a volume edge, the tracer
retains its last valid partial path and VC3D stops the native tail there. A
failure before the first outward step keeps the Lasagna fallback. A retained
Lasagna tail adjacent to a successful native span uses the same hard
continuation direction. Each retained fallback emits a terminal warning
containing `side`, the full `reason`, `trace_points`, and `source`
(`trace_result` or `exception`). Completed length-based tails and accepted
data-edge truncation do not emit this warning.
