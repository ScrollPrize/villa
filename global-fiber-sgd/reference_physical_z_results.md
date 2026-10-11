# Matched physical-Z views, completed

Output: data/reference_fullpass151_ccw_physical_z_v1/.
All maps:1000px high; y=Z*999/80000, with physical base-Z0–80000.
Equal physical U/Z pixel scales, LZW TIFF and fiber SVGs. Reference background
remapped from actual surface Z; projected fiber Z uses cached projection triangle
geometry. Fitted fiber Z uses original XYZ, not optimized V.

Files:
- reference_own.tif: reference coverage and reference winding colors.
- ccw/fit_own.tif: fitted horizontal U, physical Z, fitted winding colors(+11).
- ccw/fit_reference.tif: same geometry, reference winding colors/gray unmatched.
- ccw/reference_fit.tif: reference horizontal U, projected physical Z, fitted colors.

The reference physical extent16019.72–70987.47 occupies approximately rows200–886;
other rows remain white. Horizontal layouts differ; vertical mapping is identical.
These are diagnostic U-versus-Z maps, NOT flattened UV surfaces.

Run: approved run_classic_fiber_evaluation.py crop-benchmark with
run_reference_fullpass151_ccw_physical_z.json. No inference or optimization.
13regression tests passed, including TIFF/LZW padding, reversed intervals,
invalid-row gaps and triangle-based Z interpolation. Runner exit0.
