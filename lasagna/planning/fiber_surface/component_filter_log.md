# Connected-component filtering — 2026-10-14

- Prior solve/render log archived in connected_solve_log.md.
- Confirmed connected components, not graph cliques: >=3 distinct source fibers,
  containing H and V. Split pieces do not count as additional fibers.
- Filter before sampling/grid bounds/ruled initialization. Preserve all crossings
  within kept components; reject half-selected edges. Upstream inputs unchanged.
- Independent plan review requested legacy replay clarity: component_min_fibers=0
  disables filtering. Default is3; historical smoke/solve configs explicitly use0.
- 28 regression tests passed in2.89s; git diff --check clean.
- Completed real800-step fit via approved runner using fiber_surface_connected_min3.json,
  nonexclusive8GiB GPU0 grant a3dc702b-bcd6-4c3a-be26-0b2eba7a0a71.
- Retained1134/2185 distinct sources (531H,603V),1136/2191pieces,
  73/1056components,1450/1518crossings; no unconnected V fallback remains.
- Total16.684s,fit11.925s. H normal RMS8.9201→1.0655basevx;
  V-offset RMS101.3535→5.0502basevx. Final positional mean17.3747,
  median10.9505,p95 52.3626basevx. No attachments beyond soft allowance.
- Grid256x823 (was256x2039 unfiltered). Local folds141→215; finite,zero
  degenerate quads. Cohort/domain changed: not a matched quality comparison.
- Actual saved crossing-pair UV remains exactly coincident. Output:
  lasagna/data/connected_hv_min3_deformed_v1/surface.tifxyz.
- Independent implementation review passed. Added input/retained crossing counts
  to selection reporting afterwards (this run already records1450 downstream;
  its pre-reporting-change code hash is preserved). No optimization changed.
- Follow-up requested: render filtered solve. Independent config/plan review
  approved; reuse unchanged native renderer with CT level2,4baseUV/pixel,8px lines.
  Match previous UV band using columns100:689 for the new origin34100 (previous
  origin19700,columns244:833). CT JPG,overlay/composite LZW TIFF,inspection tifxyz.
  No refit; original unfiltered visualization remains unchanged.
- Filtered render completed in68.28s:14701x6376pixels,100%CT coverage,no chunk
  errors. CT visually inspected; TIFF dimensions match,4renderer tests pass.
  Results: lasagna/data/render_connected_hv_min3_deformed_v1/{ct.jpg,fibers.tif,
  ct_fibers.tif,inspection.tifxyz}. Existing fold defects remain unchanged.

## Reproduce
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/fiber_surface_connected_min3.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/render_connected_hv_min3_deformed.json
```
Use a fresh output directory when rerunning; completed outputs are not overwritten.
