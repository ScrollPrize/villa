# Joint sliding attachments — 2026-10-14

## Task and specification update

Fit both supplied original/deformed implicit layouts, using the deformed retained
mask identically. Keep XYZ curves immutable and H-only. Replace fixed UV in this
opt-in experiment with per-attachment UV displacements, jointly optimized with
the existing Lasagna mesh. Normal residual updates geometry only; tangential
residual updates UV only, using detached current surface normals. Regularize
displacement, displacement variation and local stretch relative to input UV.
Allow about 0.1 local winding width and 1000 base voxels vertically, with strong
penalties beyond those scales. No alternating/frozen warm-up optimization.

## Implementation

1. Validate implicit input schema and reuse/extract canonical implicit-coordinate
   mapping rather than applying the old categorical-state reader. Include offset
   knots and interpolate source/arc offsets; preserve identical H selection.
2. Generalize sampling and ruled initialization to curved UV polylines, including
   both U/V grid crossings; record physical arc weights and owner ordering.
3. Add differentiable quad sampling and attachment parameter groups to existing
   optimizer. Use padded mesh domain and smooth, non-collapsing displacement costs.
4. Regression tests for gradient isolation, cell crossing, nonconstant V,
   retained-mask identity and offset interpolation. Preserve fixed-UV regressions.
5. Run both full supplied regional layouts with matched settings, save tifxyz,
   attachment coordinates, input hashes, errors, shifts, fold diagnostics/timings.

## Documentation and changelog

Update surface docs/spec with opt-in sliding semantics, record results and commands
in a separate joint-UV results document and active task log/status. Existing pilot
results remain intact. No global intersection-free claim or source-tree writes.
