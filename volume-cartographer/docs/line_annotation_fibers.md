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
the Lasagna default for older files with no mode remains unchanged. For a new
native-mode fiber with a selected or uniquely attached fiber-inference
dataset, seed placement first computes an internal Lasagna reference line and
then immediately replaces both open tails with the existing single-control-
point native extrapolator. The reference line is not displayed or saved as the
finished fiber. If no inference dataset is selected or uniquely attached,
seed creation remains Lasagna and does not open a dataset picker.

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

The main window's **BBox** checkbox beneath **Focus** enables an annotation
focus region. The adjacent minimum and maximum fields use inclusive absolute
base-volume `x, y, z` coordinates. An unset box starts at the current volume
extent; edited coordinates are normalized and clamped to that extent. The box
is session-only, survives volume/channel changes in the current package, and is
cleared when the project is replaced or closed.

When enabled, ordinary plane views and all line-annotation cut/strip views show
the area outside the box at half brightness. The dimming is applied after base
and attached-volume composition and does not dim control markers. The main
segmentation surface view is not covered by this version of the feature.

New seeds, linked branch seeds, and control points must be inside the active
box. Existing controls remain valid outside it, so a saved seed-only fiber can
still be reopened after the box is enabled or moved. A trace or reoptimization
captures the box when it starts; later box changes do not cancel that work or
alter untouched fibers. After a solve, the path between the outer control
points is preserved even if it briefly leaves the box. Only open tails are
shortened: they retain the first sample outside the box as a small overshoot. A
lone outside control retains any connector that reaches the box and its in-box
run; if neither tail reaches the box, one neighboring line sample is kept so
the fiber remains valid. Manual/no-reoptimization edits use the same open-tail
rule.

The rendered strips are derived views, not stored line geometry. Their columns
retain every annotation control point and both line endpoints. The optimized
line points between adjacent controls define the polyline path but are not
mandatory columns. Each control-point span is resampled by polyline arclength
using the interval count whose physical spacing is closest to the
32-base-voxel target; a short span remains one interval with both controls
unchanged. Explicit
support arclengths provide a bidirectional mapping that keeps control points,
span labels, hover positions, cut planes, and saved line positions in the
original fractional point-index coordinate. The strip grid always declares an
along-line scale of `1/32`, so a short physical control-point span expands to
one nominal display interval instead of changing the scale of the rest of the
strip. Both ribbons have a fixed seven-row cross grid at 32 voxels per row,
giving a 192-base-voxel first-to-last-row extent close to the previous typical
width without depending on optimized-line spacing.

Clicking to place a control point uses optimized-polyline arclength in base
voxels. Every existing control within an inclusive 32-voxel radius is collapsed
into one control at the clicked location. This keeps adjacent control spans from
becoming shorter than the generated strip's nominal sampling distance. Seed,
surviving span policy, and branch links follow the collapsed control.

With automatic reoptimization, VC3D prepares the edit before changing the live
session. The same local update is used for insertion, one-control replacement,
and collapse: it reconstructs the surviving spans on both sides of the clicked
control from that control's known line position, then starts full fiber
optimization from the updated line. A collapse that leaves only one control
reinitializes from the clicked point and derives its tangent from the known old-
line position. It does not locate that tangent by nearest 3-D distance, which
could select a neighboring winding. If local preparation fails, the prior line,
controls, branches, focus, and optimization status remain unchanged. Reciprocal
branch updates for a multi-control collapse are saved only after asynchronous
optimization and generated-view rebuilding succeed.

The independent **Max extrap CP dist** setting limits how far a new control may
be placed beyond the first or last control point. It is measured along the
optimized polyline in base-volume voxels from the relevant outer control. It
does not restrict insertion between existing controls, and `0` means unlimited.
The current-position marker shows allowed or blocked state using this same
base-voxel arclength calculation.

The current cut view draws its solid yellow control-point marker only while the
control point is inside the cut plane's thin slab, so fast panning would
otherwise skip past control points unseen. To keep them findable, the view also
always draws two parallax ghosts: a hollow dashed yellow ring for the nearest
control point behind the cursor and one for the nearest ahead. Each ghost sits
at its true in-plane landing spot shifted horizontally toward the side it will
arrive from: the ring for a control point ahead of the cursor (higher line
position) sits to the right, matching the strips where line position runs left
to right. The shift is proportional to the signed line-position delta over a
fixed 8 line-position slide range and is clamped at 35% of the visible view
width, and the ring brightens from a faint floor at or beyond that range to
nearly opaque as the delta closes. Ghosts only appear while the control point
is within ten times the solid-marker window (so they don't linger far from any
control point), fading out over the outer quarter of that distance. Because the
shift decays continuously to zero, the ghost converges on the solid marker as a
landing ring instead of popping into place.

The Left and Right arrow keys pan the current position between control points
with a smooth velocity ramp. A tap accelerates, brakes, and lands exactly on
the nearest control point in that direction; holding the key cruises straight
through the intermediate points at a constant speed and, when it is released,
decelerates onto the next control point ahead (never short of the one a tap
would have reached). Beyond the outermost control point the pan continues one
more hop, to the Max extrap CP distance allowance or the end of the extrapolated
line, whichever is shorter. The boundary is converted from base-voxel
arclength back into optimized-line position. Pressing the opposite arrow
mid-pan decelerates through zero and reverses. Up and Down scale the cruising
speed (default 12 line positions per second, roughly 360 voxels per second),
which is shown in a transient badge and remembered between sessions. A Left or
Right press pauses the mouse hover-follow exactly as the space bar does, so the
❚❚ badge appears; space (or a click in a strip or cut view) resumes hover-follow
and cancels the pan. While the keyboard is panning, the strips stay centered
on the current-position line and scroll underneath it.

`/` and `0` both place a control point on the blue current-position dot in the
current cut, so points can be dropped without leaving the keyboard while
arrow-panning along the line. The key stops an active pan, because the
placement renumbers the line positions the pan is steering by, but unlike a
click in the cut view it leaves hover-follow exactly as it was rather than
resuming it; the panes then land on the new control point once the
re-optimized line arrives. It does nothing while the Max CP distance rule
blocks placement at the current position, and stays inert while a spinbox or
combo box has the keyboard.

The toolbar's hamburger menu owns Auto-reoptimize, Reinit reoptimization,
Show as mesh, the Lasagna/Fiber dataset submenus, embedded spinbox rows for
the initial centerline length and the base-voxel extrapolation distance,
Mirror cursor across panes, and Reset views. Mirror cursor across panes drives
the shared cursor cross between the four generated panes: on by default,
remembered between sessions, and independent of the global "Sync cursor across
views" setting, so unchecking it keeps the cross in the hovered pane even while
that global setting is on. The toolbar retains the fiber-global Lasagna/Fiber
model selector. Tag pills are edited directly from the same toolbar.

Switching the fiber-global mode asks for confirmation before it re-optimizes,
because the switch overwrites the current line: to Fiber model it re-traces
every global-goal span with model predictions; back to Lasagna it re-fits them.
Suppressed (agent-driven) sessions skip the prompt. Either direction also
strips the `reviewed` tag on the save that follows the successful
re-optimization, because the human verdict no longer covers the new geometry.

Review state is the ordinary free-form `reviewed` tag — there is no
specialized review mechanism. It is set and cleared through the generic tag
UI like any other tag: the fiber panel's tag checkboxes and the Line
Annotation toolbar's tag pills, where `reviewed` is pinned first and always
offered even in a volpkg where no fiber carries it yet. The only
programmatic change is the mode-switch strip described above; ordinary
control-point edits, merges, and splits never touch it, and
`scripts/fiber_merge.py` treats it as a plain tag under the usual
three-way tag merge. `scripts/vc_sync.py hfsync` publishes gated on it
(`reviewed` is its default `--tag`), so the tag doubles as the publish gate.

The fiber panel's `interp` column shows the interpolation provenance per
fiber — `legacy` (no trace spans), `predictions` (trace spans, native mode),
or `mixed` (trace spans under a lasagna-global fiber); the review state
itself is visible as an ordinary tag. Span child rows show the stored
producer marker `C`/`L`/`T`. Predictions provenance is the per-span
`segment_to_next.fiber_manifest` written at trace acceptance (the selected
fiber-inference manifest identity); the panel surfaces it as a tooltip on
the `interp` cells, and `fiber.list` over the agent bridge exposes the same
data as `traceState` and per-span `interpMode` plus `fiberManifest`.


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
