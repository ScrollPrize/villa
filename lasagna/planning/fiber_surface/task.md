# Task

Active: halve future render dimensions and fiber widths; preserve existing files.
See render_resolution_plan.md.

Previous: add and trial tangential geometry fitting with100x UV smoothing;
the surface should follow the regularized fiber map. See tangential_geometry_plan.md.

Previous: stronger UV-offset anchoring and smoothness, keeping between-crossing
movement and component crossing locks. See uv_regularization_plan.md.

Previous: filter whole connected components to >=3 distinct source fibers with
at least one H and one V, before initialization and fitting. See component_filter_plan.md.
Follow-up: render the completed filtered solve with matched CT/H/V overlays.

Previous: run the connected H/V deformed solve and visualize CT plus fiber overlays.
See connected_solve_plan.md. Preserve full graph; deliver central inspection view.

Previous: add weaker offset V supervision and preserve the deformed crossing graph.
Use signed normal offsets measured at saved H/V crossings, linearly interpolated
along V with constant ends and a global crossing-mean fallback. Each connected
fiber component has one UV translation; local sliding vanishes at crossings.

Previous: explicit signed-area untangling using aligned fiber normal evidence,
then feasibility-backtracked joint continuation. Test real patches and both
regional inputs. See untangle_plan.md; previous repair results remain below.

Current (2026-10-14): repair local folded patches in both saved 3x-smoothness
surfaces and continue the joint solve. After small-first repair gave only small
gains, user requested larger patches first. Test halos16/8/4/2, then improve seam
behavior with boundary-slope preservation and damped steps. See repair_plan.md.
Shared GPU grants limited to0–5; preserve and report all comparisons.

Previous (2026-10-14): implement jointly optimized per-attachment UV sliding and
normal-only geometry fitting for both supplied implicit original/deformed layouts.
Use identical deformed retained H selection. Follow joint_uv_plan.md, preserve the
fixed-UV baseline below. Run both regional inputs and report fit/shift/mesh quality.

## Completed baseline

Build a surface from the solved fiber UV map using only retained H curves.
Initialize explicitly by ruled interpolation and extrapolation, then optimize
using Lasagna's existing scale-space mesh, smoothing and quad geometry losses.
Keep target UV positions fixed, pulling their bilinear XYZ to actual curve XYZ.
Sample endpoints and grid-edge crossings. Increase a nonzero positional loss
weight through the solve; do not allow free smoothing to unravel the spiral.
No normal or V-fiber loss in this first version. Save tifxyz for vc3d inspection.

Development follows vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/AGENTS.md
in this isolated task directory, lasagna/planning/fiber_surface/.
