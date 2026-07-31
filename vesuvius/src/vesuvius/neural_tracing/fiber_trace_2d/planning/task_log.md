# Per-Segment Interpolation Goals And Cubic-Spline Fallback Task Log

## Planning Findings

- Current VC3D persistence has a fiber-wide Lasagna/native-trace mode and an
  optional native-specific `segment_to_next` record. The general goal/actual
  descriptor is therefore a schema change, not just an added trace field.
- Control point `i` already owns the segment to `i+1`, which remains the correct
  location for the new descriptor.
- Lasagna failure belongs to its per-span initialization/rollout: a span can
  fail to produce a usable candidate. The later joint Ceres pass refines the
  already initialized usable spans and is not the fallback decision point.
  The current combined API stops at the first failed initializer, so its private
  span initializer must be extracted into a shared API that lets the coordinator
  classify every span without copying candidate logic.
- The existing Ctrl-right-click path already resolves the containing ordered CP
  pair and provides one-span native trace/revert actions. It can be generalized
  to a checked goal submenu without introducing another hit-test path.
- The current line-model helper requires valid sampled normals, but a synthetic
  geometry-only model helper already exists. Spline geometry can therefore be
  independent of normals while still attaching normal samples when available.
- Current strip labels use only the CP midpoint in scene coordinates. They do
  not test whether any part of the span is visible, clamp into the viewport, or
  resolve collisions. The new layout must operate in viewport pixels because
  the label graphics items ignore scene transformations.

## Review

- Reviewed the plan directly against the current schema, controller, generated
  menu, native fallback coordinator, and Lasagna reinitialization API.
- Independent subagent review was not used because the active collaboration
  instructions prohibit spawning subagents unless the user explicitly requests
  delegation.
- Confirmed that the 100-voxel rule uses Euclidean base-coordinate CP distance,
  Lasagna fallback is decided per initialized span before joint refinement, and
  the global selector remains Lasagna/trace while all three modes are available
  as explicit segment goals.
- Added persisted mode-dependent `metric` and compact `msg` fields plus
  viewport-aware, collision-resolved labels for every visible span.

## Implementation And Validation

- Not started; this task is currently at the requested planning stage.
