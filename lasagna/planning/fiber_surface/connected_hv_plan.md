# Connected H/V surface fitting

Reuse deformed retained selection, saved event/node identities, actual polylines,
aligned normal sampling, Lasagna fitting and sliding regularizers. H initializes
the ruled surface. Add V targets shifted toward H by signed normal distances at
saved crossings, interpolated along source arc (constant ends; global signed mean
for unconnected V). Freeze these measured targets; V data weight defaults to 0.1.

Read H/V mapping correctly: signed arc advances U for H and V for V. Insert saved
crossing nodes into attachment sampling. Union connected source fibers using saved
crossing edges. One trainable UV translation per component plus per-attachment
local displacement, exactly zero at crossings. Preserve input crossing coordinates
and relative positions, rather than introducing another UV alignment solve.

## Spec update
Replace H-only restriction with opt-in connected-HV deformed mode; retain old H
workflows. No new normals or geometry optimization upstream. Record offset and
component statistics and save sufficient attachment metadata to audit constraints.

## Tests
Synthetic offset interpolation/end/fallback, source-component union, exact shared
crossing translation after optimizer updates, free intermediate movement, weighted
normal/tangent gradient routing and old sliding regressions. Audit real saved
crossing identities and UV agreement; fail clearly on invalid normals or missing
nodes. Run approved runner on a small synthetic CPU fit and real preparation.

## Docs updates
Document configuration, sign convention, weighting, topology and persistence in
fiber_surface.md, status/task_log, changelog. Archive previous active log first.
