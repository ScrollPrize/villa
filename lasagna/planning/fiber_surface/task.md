# Task

Active: render cropped deformed surface from CT and export matching fiber overlay.
See render_plan.md. Do not change the fitted surface during rendering.

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
