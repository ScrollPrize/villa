# Matched CT and fiber surface render

Render the cropped deformed `continued.tifxyz`, without further optimization.
Reuse the existing TIFXYZ loader and bilinear field tile interpolation, and the
native vc Volume blocking trilinear sampler with its shared remote chunk cache.
Open the authoritative Paris4 scan URL so unfetched chunks cannot silently be
treated as empty local data. No GPU is required.

Use 4 base voxels/output pixel and CT level2. Render in bounded row bands with
native sampling threads. Use the saved final attachment UVs, grouped by piece
and ordered by source arc, not nearest-XYZ projection. Apply the exact inverse
vertex-aligned grid mapping for both shared bilinear sampling and drawing (pixel
0 is grid vertex0, sample only through the last vertex). Export LZW TIFF CT,
RGBA fiber overlay, and combined RGB, identical sizes/orientation. Stable colors
identify source fibers. Save provenance, mapping, validity and timings in JSON.

## Spec update
Rendering is read-only with respect to solver outputs. The overlay represents
fitted H attachment polylines; no additional V fibers or inferred correspondences.
Report unresolved surface folds, attachment interpolation and scan resolution.

## Validation and documentation
Test affine surface pixel registration and fiber coordinates, piece separation,
output dimensions and valid coverage. Inspect an actual image crop. Record exact
runner config, outputs and timings in task_log/status/docs/changelog. Preserve
previous experiment logs in a historical file before replacing active log.
