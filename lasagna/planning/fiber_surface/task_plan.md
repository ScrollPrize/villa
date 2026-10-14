# Fixed-UV H-fiber surface reconstruction

Active task: [render_plan.md](render_plan.md). Untangling is paused, not solved.

Active follow-up: [smoothing_plan.md](smoothing_plan.md), independently reviewed.
The joint sliding implementation below is complete; tune only existing weights.

This is the completed baseline plan. Current reviewed implementation plan:
[joint_uv_plan.md](joint_uv_plan.md), superseding fixed-UV-only for the opt-in mode.

## Task
Construct an explicitly ruled/extrapolated XYZ surface from retained H curves
in the solved UV map. Optimize it with existing Lasagna mesh scale-space,
smoothing and quad geometry losses. Fixed sub-quad UV positional targets pull
to actual fiber XYZ; normals and V-fiber supervision are not used. Ramp a
strictly positive data weight from weaker to stronger, never free smoothing.

## Implementation plan
1. Reuse the solved-map geometry and actual AFV curves. Select joint probability
   >0.5. Sample endpoints, grid crossings, and curve detail with arc weights.
2. Construct ruled H brackets at equal U, extrapolating missing support to fill
   a rectangular initial surface; report extrapolated coverage and degeneracies.
3. Add a fixed_uv term to the existing Lasagna optimizer, using its Model3D
   scale-space and existing smooth/step/bend losses without a separate solver.
   Cache bilinear indices/weights and support a positive stage-weight ramp.
4. Save init/final tifxyz, parameters and before/after fit/geometry diagnostics.
5. Test sampling, reversed curves, extrapolation, fixed correspondence gradients,
   ramp, and a curved synthetic solve; time a small real regional pilot.

## Spec update
This is a separate surface experiment; no winding solver or existing surface
construction behavior changes. No ray search, normal loss, V supervision,
new collision solver, or winding re-optimization.

## Documentation and changelog
Document command, units, weights, extrapolation limitations and measured pilot
results in docs and task_log.md. Update status.md and changelog.md.

## Validation
Regression tests run on CPU. A real pilot must report timings, fiber errors,
finite geometry and local fold diagnostics; no claim of global nonintersection.

Read this together with task.md and spec.md in this directory; the historical
global-fiber-sgd/surface_plan.md remains a different H/V interpolation experiment.
Refactor CurveBank geometry access into a reusable superclass so the new path
does not load normals or duplicate existing polyline interpolation.
