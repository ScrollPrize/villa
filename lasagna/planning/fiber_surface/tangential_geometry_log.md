# Tangential geometry fitting

- Prior offset smoothing task archived in uv_regularization_log.md.
- Implement optional geometry tangent weight; compare0.1/0.3/1 against saved0
  with identical smooth100 settings and input. Preserve gradient routing and locks.
- Independent plan and implementation review passed. Default0 preserves old loss;
  weight1 gives full positional geometry error with detached UV/normals. UV loss
  remains tangent-only. H/V separate family weighting and offset targets unchanged.
- Weight serialized in attachments and restored during continuation. Legacy crop
  now preserves scalar metadata instead of trying to index it as sample arrays.
- Added per-family tangent/full-position RMS. Audit recomputes them for historical
  baseline using the same sampler/weights; initial samples/IDs/arcs are identical.
- 32 regression tests passed (routing,0 compatibility,full-position equivalence,
  invalid weights,restoration,existing renderer/repair/adapter tests).
- Three fresh800step solves finished on nonexclusive8GiB GPU0 manager grants,
  shared concurrently;49.79/50.19/49.90s total,not isolated performance figures.

| Geometry tangent weight | H full XYZ RMS | V-offset full XYZ RMS | H normal RMS | V normal RMS | Local UV offset p95 | Offset slope RMS | Folds |
|---|---|---|---|---|---|---|---|
| 0 baseline | 50.7463 | 101.5481 | 1.8856 | 10.9044 | 35.2809 | .030984 | 279 |
| .1 | 8.3366 | 45.9148 | 1.7558 | 7.2020 | 18.4982 | .015056 | 358 |
| .3 | 8.2144 | 44.1721 | 1.8341 | 5.4120 | 17.7414 | .014670 | 418 |
| 1 | 7.4657 | 41.8743 | 2.0530 | 9.5239 | 17.4119 | .014251 | 517 |

- All distances basevoxels. Tangential RMS H/V: baseline50.7113/100.9609,
  .1:8.1496/45.3465,.3:8.0071/43.8393,1:7.1779/40.7768.
- All1450crossings exactly coincident. UV offset slope-change RMS decreases from
  .025583 to .013181/.013198/.012777; local deformation stays smooth. Shared
  component translation still moves: totalU/V p95 are187.71/337.36,.1;
  133.20/331.23,.3;130.03/311.14,1 (baseline140.77/165.36).
- Geometry tangential correction substantially improves correspondence, but folds
  worsen. Render .1 as conservative tradeoff and1 as full-fit comparison. No
  universal default promotion; not a fold repair.
- Both renders complete: render_connected_hv_geom01_v1 (32.92s) and
  render_connected_hv_geom1_v1 (32.65s), each14701x6376,100%CT coverage,
  same UV band and8px overlays. CT JPG,transparent/combined TIFF dimensions checked;
  inspection.tifxyz exported. No render code changes.

## Commands
Use approved runner with configs fiber_surface_geom01.json, fiber_surface_geom03.json,
fiber_surface_geom1.json; corresponding render configs render_connected_hv_geom01.json
and render_connected_hv_geom1.json:
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/fiber_surface_geom01.json
PYTHONPATH=/home/hendrik/villa4/vesuvius/src:/home/hendrik/villa4 /home/hendrik/.venv_las/bin/python lasagna/audit_uv_offsets.py lasagna/data/connected_hv_min3_uvsmooth100_v1 lasagna/data/connected_hv_uvsmooth100_geom01_v1 lasagna/data/connected_hv_uvsmooth100_geom03_v1 lasagna/data/connected_hv_uvsmooth100_geom1_v1
```
