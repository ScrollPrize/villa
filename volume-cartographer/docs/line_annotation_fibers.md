# VC3D Line Annotation Fibers

VC3D writes line annotations as `vc3d_fiber` JSON. Version 2 stores
`control_points` as objects with a required `position` and optional
`segment_to_next`. Control point `i` owns the metadata for its span to control
point `i+1`; the final control point cannot contain `segment_to_next`.

The top-level `optimization_mode` is either `lasagna` or
`native_fiber_trace3d`. Files without this field default to `lasagna`. The mode
selects the fiber-wide interpolation and extrapolation policy for future edits.
It does not replace `segment_to_next`, which records the native attempt outcome
for one concrete span. `accepted_native` means the stored geometry came from
the native tracer and is protected from normal optimization;
`lasagna_fallback` retains diagnostics for a failed native attempt but does not
protect the Lasagna geometry.

In native mode, VC3D traces changed CP-to-CP spans against the selected fiber
inference manifest. A rejected or invalid native result falls back only that
span to the selected Lasagna normal dataset. A later native retry replaces the
fallback record with its new outcome. Moving a CP or changing adjacency clears
either outcome on affected spans while preserving unrelated records.

Each direction continues until it reaches all target-local planes within the
20-base-voxel endpoint threshold or exhausts its step budget. VC3D then moves
locally tangent planes along both complete traces and intersects the opposite
trace. It selects the smallest meeting error and accepts it when the error is
at most 10% of the combined partial traced length. This can succeed even when
neither direction reached its endpoint planes. The accepted partial traces are
warped by arc-length fraction to their shared midpoint, concatenated, and
resampled, with the original CP endpoints restored exactly.

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

Generated-strip labels use the CP-owned record directly. Accepted native spans
show their persisted meeting error in base voxels; fallback spans show a
compact stable failure reason, and the tooltip includes the full stored detail.
Spans without a native attempt retain the ordinary Lasagna normal-alignment
display. Reoptimization and branch-overlay refreshes repopulate these labels
from the current records. Ordinary Reoptimize preserves accepted native spans;
switching explicitly to Lasagna mode or reverting a span clears them.

The line-annotation extrapolation control is in base voxels. Lasagna mode grows
normal-based tails. Native mode attempts each tail with the shared one-way
fiber tracer after converting the requested distance to trace voxels; a failed
native tail keeps its Lasagna fallback. Stored line and control points always
remain in base coordinates. When the prediction field returns no valid next
direction at a volume edge, the tracer retains its last valid partial path and
VC3D stops the native tail there. Step-budget and other failures still keep the
Lasagna fallback. A retained Lasagna tail adjacent to a successful native span
uses the same hard continuation direction.
