# Trace2CP Shared Strip Builder And Tighter Reset Threshold

Regenerated/refined native 3D Trace2CP strips should not depend on the original
source strip grid. The actual side/top strip construction must be shared with
the original CP-pair strip path; only the argument preparation should differ.

The shared builder should take an explicit vector of 3D line points plus the
line-local metadata needed to build a `FiberStripLineWindow`, sample/align
Lasagna normals, and construct the side/top grids. It must not assume that all
inputs come from `record.fiber` or from an old source-strip coordinate system.

Also lower the native whole-fiber CP-plane reset/error threshold from 100
selected-scale voxels to 10 selected-scale voxels.
