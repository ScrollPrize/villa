# Paris4 reference surface: acquisition and comparison proposal

## Downloaded reference

Source: `s3://philodemos/sean/paris4_full_100226/mesh/spiral-output-generation-19-lasagna.tifxyz/`

Local: `data/reference/spiral-output-generation-19-lasagna.tifxyz/`

Six files: x.tif,y.tif,z.tif,d.tif,meta.json,model_params.json.
870572039bytes (830.24MiB). download_manifest.json adds source paths, timestamps,
sizes, ETags and local SHA256 fingerprints. No credentials. ETags are not assumed
to be MD5; no publisher-supplied checksum comparison claimed.

Grid956rows x56913columns, float32/uncompressed.47026606of54408828XYZpoints valid
(86.43194%,finite and no-1coordinate markers). Coordinate bbox:
X1288.2297..7588.8501,Y1191.5481..7441.3271,Z4004.9302..17746.8672.

Metadata voxel_size_um=9.6, base_shape_zyx=[18946,8174,8174], downscale=4;
these match ceil(AFVL0shape/4) for the2.4um Paris4 scan, not evidence of the
older nominal9um scan. Multiply XYZ by4 for our native2.4um fiber coordinates.
Verify with geometric residuals before adopting exact registration. The actual
native-Z support is approximately16020..70987; reference does not cover all
75784nativeZvoxels. Output step20referencevoxels corresponds to80nativevoxels;
metadata scale0.05 is surface-grid sampling, not an XYZ scaling factor.

## Important missing sidecar

meta.json explicitly names winding-ids.tif, and lists126winding IDs10..135.
That image is NOT in the supplied S3 prefix. The parent mesh/ listing contains
no files and only this surface prefix. d.tif is0on valid surface points (-1
elsewhere), not a substitute. winding_bounds are bounding rectangles with
overlap, not exact winding membership. Local spiral-fitting/lasagna_publish.py
_mapped_winding_ids shows the sidecar comes from categorical mapping through
the flattening correspondence; rectangles cannot reproduce its boundary pixels.

For exact winding-color evaluation obtain winding-ids.tif (and ideally the
flattening correspondence/source manifest). Without it, unique-bounds interiors
can be provisionally labeled, ambiguous regions omitted, or geometric angular
unwrapping can estimate labels anchored to unambiguous bounds. Those are
approximate reconstructions and must never be advertised as the original labels.

## Proposed comparison

SUPERSEDED: the exploratory proposal below is historical, not the implemented
contract. The user explicitly ruled out correspondence/path optimization and
normal-guided assignment. Current implementation follows
reference_comparison_plan.md: nearest geometry, original-fiber weighted median
discrete winding, chosen-winding reprojection, rectangle midpoint interpolation.
No continuity solve, no normal-based assignment, no optimizer changes.

1. Build a reusable spatial projection index on valid reference quads/triangles
   in native XYZ. Use coarse spatial blocks then exact local triangle projection;
   avoid a huge all-pairs search or nearest-vertex quantization at80vx grid pitch.
   Mask holes/degenerate faces and suspicious long bridges. Reference is ~47M
   vertices, so tiled indexing is preferable to a Python object per triangle.
2. Sample original AFV fiber curves uniformly by arc length (also piece boundaries).
   Find several surface candidates, using distance, signed-normal agreement and
   continuity along the fiber. Preserve ambiguity between adjacent windings;
   no solve assignment may influence this independent reference association.
   H/V separated: establish which sheet surface the reference represents and
   use aligned normals to associate V with its intended inner H sheet, not blindly
   the nearest outer/inner surface. Report distance distributions before choosing
   confidence thresholds. Do not silently assign uncovered fiber length.
3. Interpolate reference UV at projection; use categorical winding labels without
   averaging across a seam. Arc-length-weighted median reference winding gives
   a stable whole-fiber display color; also retain per-sample labels, dominant-label
   fraction, variation, coverage and distance. A single median can hide switches
   or legitimate winding-seam crossings, so scores use local correspondences.
4. Derive solved winding from actual continuous q(s)=U_inverse(u(s)), including
   phase/offset and signed arc travel, not bare candidate-bucket ID. Match winding
   seam conventions using angle/reference correspondence; fit one global numbering
   offset per solution from high-confidence matches. No independent per-fiber
   offset, remapping or arbitrary scale that hides missed turns. Verify reference
   winding ordering; CW/CCW hypotheses are not permission to reverse radial order.
5. Report matched length%, ambiguous/uncovered length%, exact-winding length%,
   within±1winding%, median/p95absolute winding error and signed-error histogram.
   Include winding confusion matrix and median residual versus reference winding
   and height, to expose drift/compressed winding count. Score retained-only first;
   disabled fibers separate and conditional-placement results explicitly separate.
   Reference is an independent reconstruction, not guaranteed anatomical truth.

126referenceIDs vs100candidatebuckets is an important check: do matched fibers
span all126turns, and does our effective q-range cover them? The free shared phase
and offsets mean bucket count is not automatically a hard physical q-range,
but a100-bucket assumption may still be insufficient. Do not force126reference
windings into100by scaling labels. Distinguish coverage, gauge and real undercount.

## Proposed visualization (recommended first)

- Render existing CCW/CW solved UV fiber maps using the SAME color per reference
  winding, assigned by arc-length-weighted median. Good solutions should show
  ordered color bands; mixing, folding or repeated bands highlight wrong placement.
  Unmatched/ambiguous gray; confidence/coverage legend separate from solver opacity.
- Reference UV map with projected fiber curves in those same colors alongside it.
  Use categorical palette/number legend (126colors are not individually easy to
  distinguish); add a continuous winding gradient and zoomable vector outputs.
- Companion solved-UV error maps: signed winding difference diverging palette,
  zero neutral,±1clearly visible and large errors saturated/labeled. More legible
  than guessing exact errors from126colors. Per-sample coloring reveals switches.
- Optional3D PLY fibers/reference mesh with same reference colors; distinguish H/V
  and rejected fibers with separate files rather than competing color semantics.

Start with reference-colored maps plus difference maps and coverage. This compares
the existing solutions without fitting them toward the reference. Then use the
quantitative table to decide whether another solve or increased bucket range helps.

## Reproduce download (approved runner)

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/download_reference_surface.json
```

Network execution uses normal sandbox escalation. Boto3 uses default credentials;
completed files are reused only when manifestETag,size,andlocalhash match.
Partial files renamed only after download/size check. No source writes/deletes.
Download path safety/hash regression passed; comparison and colors are a proposal,
not implemented or evaluated yet.
