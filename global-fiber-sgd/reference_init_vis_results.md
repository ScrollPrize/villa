# Reference-initialized solve visualization

Completed successfully in236.89seconds, CPU-only using existing projections.
Source: data/uv_reference_init151_v1/ccw_joint, completed10passes.
Output: data/reference_init151_ccw_comparison_v1.

All four maps available together in output root:
- reference_own.tif: full reference, own winding colors.
- ccw_fit_own.tif: solve, own winding colors.
- ccw_fit_reference.tif: solve geometry, reference winding colors.
- ccw_reference_fit.tif: fibers placed on reference, solve winding colors.

CCW filenames are relative links to completed TIFFs under ccw/. All verified
LZW RGB,1000px high. Fit maps80893px wide,reference maps56857px wide. Identical
vertical mapping0..80000; affine reference alignment preserves originalUV
curves and full extent. Color offset0. Existing SVGs remain in ccw/.

Run:
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_reference_init151_ccw_comparison.json
```

Validation:15comparison/affine tests passed; successful process exit; TIFF headers,
links,summary and complete mapping entries checked. No optimizer changes.
