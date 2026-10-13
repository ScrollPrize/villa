# Task

Build a surface from the solved fiber UV map using only retained H curves.
Initialize explicitly by ruled interpolation and extrapolation, then optimize
using Lasagna's existing scale-space mesh, smoothing and quad geometry losses.
Keep target UV positions fixed, pulling their bilinear XYZ to actual curve XYZ.
Sample endpoints and grid-edge crossings. Increase a nonzero positional loss
weight through the solve; do not allow free smoothing to unravel the spiral.
No normal or V-fiber loss in this first version. Save tifxyz for vc3d inspection.

Development follows vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/AGENTS.md
in this isolated task directory, lasagna/planning/fiber_surface/.
