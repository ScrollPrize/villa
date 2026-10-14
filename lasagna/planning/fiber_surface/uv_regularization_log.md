# Stronger UV-offset regularization

- Previous filtering/render log archived in component_filter_log.md.
- Reuse existing offset-gradient and anchoring losses; no formula changes.
- Matched candidates: displacement0.1, smooth10/100 versus baseline0.01/1.
- Keep input cohort,800steps,mesh losses,UV learning rate and crossing locks fixed.
- Independent plan/config review approved. Measure local deformation separately
  from shared component translation, and slope-change as well as displacement.
- 12 connected/sliding/render regressions passed in2.37s. Both real solves finished
  using nonexclusive8GiB manager grants on GPU0; simultaneous shared-GPU timings
  are not an isolated throughput comparison. No changes to solver formulas.
- Added read-only audit_uv_offsets.py: checks identical initialUV/source/arc/piece
  samples, reports local-displacement p95, slope RMS and consecutive slope-change
  RMS along piece arc, fit errors and exact crossing coincidence.

| Offset smooth weight | Anchor weight | Local offset p95 vx | Offset slope RMS | Consecutive slope-change RMS | H normal RMS vx | V-offset normal RMS vx | Folds | Total seconds |
|---|---|---|---|---|---|---|---|---|
| 1 baseline | .01 | 186.0405 | .206069 | .216281 | 1.0655 | 5.0502 | 215 | 16.68 |
| 10 | .1 | 101.5380 | .090961 | .082116 | 1.4919 | 6.5187 | 272 | 21.42 |
| 100 | .1 | 35.2809 | .030984 | .025583 | 1.8856 | 10.9044 | 279 | 21.30 |

- All1450crossing pairs remain exactly coincident. Strongest smoothing reduces
  local p95 by81.0%,slope RMS85.0%,slope-change RMS88.2%. TotalU p95 nevertheless
  rises127.98→140.77vx (shared component motion remains),V p95 falls243.20→165.36.
- Select smooth100 for requested smoother-offset inspection, not as an overall
  better surface: fit and folds worsen. No universal API default change.
- Existing penalty is first derivative of the added displacement, not a new
  curvature penalty. It demonstrably reduces rapid slope changes; original
  deformed input curvature remains untouched. No skipped implementation needed.
- Smooth100 render completed: identical global UV crop and8px overlays,
  14701x6376pixels,100%CT coverage,30.98s. TIFF dimensions validated.
  lasagna/data/render_connected_hv_min3_uvsmooth100_v1 contains ct.jpg,
  fibers.tif,ct_fibers.tif and inspection.tifxyz. Results logged; folds not repaired.

## Commands
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/fiber_surface_uvsmooth10.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/fiber_surface_uvsmooth100.json
/home/hendrik/.venv_las/bin/python lasagna/audit_uv_offsets.py lasagna/data/connected_hv_min3_deformed_v1 lasagna/data/connected_hv_min3_uvsmooth10_v1 lasagna/data/connected_hv_min3_uvsmooth100_v1
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/render_connected_hv_uvsmooth100.json
```
