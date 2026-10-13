# Initial implementation results — 2026-10-13

Frozen source: `villa2/global-fiber-sgd/data/uv_gt_l1_search_v1/trials/9cac11649421bd1f7352`.
Solution SHA256: `ee4bcbac01f4c749955b0fc18f85f53e3c01e391d96018284284b689fc459ed2`.
This is a pinned completed candidate, not a claim to the other search's latest best.
Patch U=[60000,80000], V=[36000,44000], grid100 base voxels; 56 H pieces,
2490 observations, 81x201 vertices, 16000 quads. All runs used 8 CPU threads.

## Final corrected implementation pilot (v4)

| Metric | Ruled initialization | After 600 iterations |
|---|---:|---:|
| Arc-weighted mean XYZ error, base vx | 116.3484 | 11.0572 |
| Arc-weighted RMS XYZ error, base vx | 318.8216 | 68.4135 |
| Sample median XYZ error, base vx | 25.1016 | 2.6817 |
| Sample p95 XYZ error, base vx | 766.1236 | 9.7707 |
| Maximum XYZ error, base vx | 4364.2031 | 841.5975 |
| Opposing-triangle quads | 106 (0.6625%) | 181 (1.1313%) |
| Degenerate quads | 0 | 0 |

Initialization/export setup 0.051s; solve 19.326s (~31 iterations/s), whole
surface stage 19.386s. Adapter loading/imports are outside this stage timer.
Regularizers smooth/step/bend = 0.5/0.5/5; data weight 1→20, LR5.
All coordinates finite. No volume reads, new inference, or GPU allocations.
No global intersection-free claim; fold count increased despite lower data error.

Output: `/home/hendrik/villa4/lasagna/data/fiber_surface_pilot_v4/`.
Open `initial.tifxyz` and `surface.tifxyz` in vc3d for comparison.
Provenance includes code hashes and frozen targets for reproducibility.

Support: 50.9182% bracketed, 35.1514% V extrapolated, 10.9453% singleton H,
0% empty-column interpolation, 2.9851% U extrapolated.

## Earlier pilot findings (retained, not promoted)

- v1: unbounded extrapolation sent vertices to -1.1M base voxels; rejected.
- v2: bounded extrapolation, LR0.5/300 steps, weak geometry. Mean error61.34,
  RMS208.95; 276 opposing-triangle quads; 10.93s.
- v3: stronger existing geometry/LR5/600 steps. Mean11.86/RMS68.45,
  177 opposing-triangle quads; 19.80s. Review then fixed the singleton donor
  slope discontinuity; v4 above uses that correction.

## Tests

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q --tb=short --noconftest \
  -o pythonpath=/home/hendrik/villa4/vesuvius/src \
  lasagna/tests/test_fiber_surface.py lasagna/tests/test_opt_loss_step.py \
  lasagna/tests/test_optimizer_stage_weights.py \
  vesuvius/tests/test_fiber_uv_surface.py \
  vesuvius/tests/test_fiber_lasagna_surface.py \
  -k 'not atlas_snap_flow_speculative'
```

47 passed, 1 deselected, 2.66s. Full selected suite initially confirmed that
one unrelated pre-existing failure: `test_atlas_snap_flow_speculative_config_parses_snap_plus_pred_dt`
expects debug=true and interval10 while the unchanged committed config contains
debug=false and interval1. No test expectations/config altered to hide it.
`git diff --check` passes.

Covered: signed actual-curve mapping, old interpolation compatibility, endpoint/
grid-edge sampling, arc weights, bilinear gradients/boundaries, scale-space
correspondence, positive monotone ramp, bounded extrapolation and singleton
profile continuity, real optimizer synthetic fit, and managed-launch dispatch.
CUDA solve execution remains unmeasured; whole-scroll fitting has not been run.
