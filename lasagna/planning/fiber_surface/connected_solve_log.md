# Connected deformed solve and rendering — 2026-10-14

- Previous implementation log archived as connected_hv_implementation_log.md.
- Independent run-plan review: no blockers. Preserve full graph; crop inspection
  output only, not optimization topology. Mark H/V overlay families.
- Reused approved evaluation runner. Manager status initially had no live socket;
  runner auto-started shared manager and obtained a nonexclusive8GiB grant on GPU0,
  task3965783e-0a48-4203-a419-1a603c652894. No bypass/exclusive allocation.
- Config fiber_surface_connected_solve.json:600steps meshLR1,ramp1→20, then200steps
  meshLR0.2,data20; smooth1.5,bend5,step0.5,V factor0.1. UV updates remain joint.
- Completed:22.79s total,11.38s fit,11.22s core optimization. Not proven converged:
  final fixed-weight objective19.0579 (settling start24.2532).
- 849H/1342V pieces;1518crossings retained exactly coincident (saved UV difference0).
- Full-domain H normal RMS13.4093→1.0313basevx; V offset RMS729.942→126.8568.
  Arc-weighted positional mean312.143→26.3861, median39.887→12.0861,
  p95 1357.277→81.3097. 2.6776% attachments exceed soft movement allowances.
- Full-domain folded quads513→988; finite geometry,no degenerate quads. This is
  worse fold count, not a repaired surface or a replacement best result.
- Central inspection columns244:833 correspond to fitted winding band14..21:
  256x589vertices,735folded quads. Final in-band H normal RMS1.0124basevx,
  V offset RMS6.0967basevx. Large full-domain V RMS is concentrated outside this
  band. These regional metrics select final UV inside the band.
- Rendering uses shared native CT sampler,level2,trilinear,4baseUV/pixel,8px lines,
  H/V family tags in fibers.json. CT JPG; overlay/composite/validity LZW TIFF.
  Export inspection.tifxyz is visualization-only, not a cropped solve checkpoint.
- Four renderer regressions passed including graph-preserving inspection crop.
- Render completed in58.72s at14701x6376 with100%CT coverage; CT image visually
  inspected. Overlay TIFF dimensions/registration checked by tests and file audit
  (image viewer does not support TIFF). Both H and V are present in overlay index.
- Artifacts: lasagna/data/render_connected_hv_deformed_v1/{ct.jpg,fibers.tif,
  ct_fibers.tif,inspection.tifxyz}; full solve in connected_hv_deformed_solve_v1.

## Commands
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/fiber_surface_connected_solve.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/render_connected_hv_deformed.json
```
