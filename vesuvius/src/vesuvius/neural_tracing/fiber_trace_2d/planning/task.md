# Task: dense-fiber failure replay

Add a C++-only `fiber-replay` workflow for finding where the regular greedy
native fiber tracer first diverges from an existing dense VC3D fiber and then
extracting local fiberlet diagnostics around that failure.

Given a fiber-prediction manifest, regular Lasagna-normal manifest, and VC3D
fiber JSON, start at the first control point with the direction of the first
non-degenerate dense `line_points` edge. Trace forward using the existing native
greedy implementation and compare every traced point with the dense reference,
not only control points. Stop after the first configured distance failure plus
100 further trace steps.

Track the corresponding dense-reference point incrementally. At every accepted
greedy step, initialize the next reference arclength at one nominal trace step
past the previous match, then directly refine the closest point only within a
limited forward arclength window. This local monotone search must correct step-
length drift without jumping to a nearby crossing, return, or winding. Use the
resulting point-to-point distance as the failure error.

Select a configurable interval forward and backward from the matched failure
arclength, and operate only on anchor cells within an initially 128-base-voxel
tube around that reference interval. Extract anchors and fiberlet paths for
those anchors and retain their normal machine and napari visualization
artifacts. Also retain the selected reference segment, traced line, failure
location, and termination diagnostics.

Write one strict JSON replay bundle that records provenance, parameters,
failure diagnostics, a base-coordinate volume crop, and relative paths to all
derived visualization artifacts. It must not embed or assume a path to the
fiber-presence Zarr; the viewer receives that separately. Extend the napari
fiber-presence viewer so the Zarr plus replay bundle loads the volume, reference,
trace, anchors, and fiberlets together.

All workflow logic is C++. Port any needed Python reference-distance/arclength
rule into shared C++ code with parity tests; do not call Python or add bindings.
