# Sign solution visualization

Export original unsplit AFV polylines and actual signed sheet-normal arrows from
stored constraint endpoints (baked normal * solution sign), with broken fibers
red and retained fibers neutral/blue. No normal resampling/inference/GPU needed.
Use standard OBJ line elements for fibers and a binary colored PLY triangle mesh
for arrows (shafts and cone heads) so direction is visible without viewer glyphs.
Separate files can be overlaid in MeshLab; native XYZ/base-voxel coordinates.

Select explicit fiber IDs or R-tree bbox candidates, keeping full source fibers
(no clipping); document bbox selection semantics and fail on excessive count.
Default preview:512base-voxel box centered on a deterministic broken fiber's
constraint endpoint. Thin arrows along arc with configurable spacing/length.
Metadata retains per-fiber IDs, sign/probabilities, geometry index ranges and each
arrow's exact arc/point/normal. Original fibers/solutions untouched.

Reuse raw AFV reader by adding optional ID selection, preserving existing default.
Do not use AFVFibers.geometry which repairs/densifies training data. Approved
evaluation runner dispatch for repeatable CPU export. Native existing schema used.

Tests: mesh indices/arrow orientation, raw AFV selective-reader equivalence, verify
export sign against solution and normals, parse/count generated OBJ/PLY. Record
paths/counts/size in docs/log; no claim of anatomical correctness.
Spec update: visualization-only export. Docs updates: sign_resolution.md and
results export metadata/task log/status/changelog. Independent review per AGENTS.
