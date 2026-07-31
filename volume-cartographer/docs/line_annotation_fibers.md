# VC3D Line Annotation Fibers

VC3D writes line annotations as `vc3d_fiber` JSON. Version 3 stores
`control_points` as objects with a required `position` and optional
`segment_to_next`. Control point `i` owns the metadata for its span to control
point `i+1`; the final control point cannot contain `segment_to_next`.

The top-level `optimization_mode` is either `lasagna` or
`native_fiber_trace3d`. Files without this field default to `lasagna`. The mode
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
second row when one row cannot avoid overlap. Version-1 and version-2 fibers
remain readable; VC3D writes explicit version-3 descriptors on the next save.

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
