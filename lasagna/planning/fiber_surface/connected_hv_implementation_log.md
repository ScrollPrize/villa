# Connected H/V surface fitting — 2026-10-14

- Prior state committed as 9fc1bffba; rendering log archived in render_task_log.md.
- Independent review approved mechanism with explicit signed normals, separately
  normalized H/V data weights, total-motion regularizers and topology persistence.
- Offset = (H−V) dot aligned inward V normal; frozen target = V + offset*normal.
  Duplicate edges removed; coincident V-arc observations averaged. Global fallback
  is the signed mean of distinct observed crossings. V data factor defaults 0.1.
- Connected sources share one trainable translation; local corrections are masked
  to zero at every crossing attachment copy. Intermediate samples remain free.
- Initial real audit detected 0.01714 base-voxel UV mismatch. Root cause was
  adding float32 winding+phase before converting to float64. Both terms now convert
  before addition. Also use original signed event ds at crossing nodes rather than
  re-deriving it from serialized source arcs. Input crossing error now 3.4e-11 vx.
- First pytest picked globally installed villa; reran with explicit villa4
  PYTHONPATH. 12 tests passed.
- Old H-only paths preserved. Legacy post-fit per-sample crop explicitly rejects
  connected-HV checkpoints: dropping part of a component breaks this contract.
  Whole-component crop support is not included. No new repair claim.
- Real three-step CPU smoke: 2191 pieces (849 H, 1342 V), 2185 sources, 111630
  attachments, 1518 unique crossings, 1056 components. Mean signed offset 34.6801
  base voxels (median31.9868, range−106.6823..111.8641). All saved crossing-pair UV
  differences exactly zero after optimization. The 0.0030vx lock diagnostic is
  float32 subtraction roundoff in measuring translation, not H/V separation.
- Smoke wall time19.65s; fit wrapper7.80s; core optimizer3.35s. Full retained layout
  rectangle256x2039, NOT the earlier central cropped domain. Deliberately only an
  implementation smoke, not a production solve: H normal RMS13.409→12.563vx,
  V offset RMS729.942→670.102vx; folds513→515. Unconnected/outlier V coverage has
  substantial extrapolation and poor initial fit, no quality improvement claimed.
- Review caught stale component/local deltas in repair continuation outputs;
  shared attachment_state writer now updates all motion state, with roundtrip test.
- Auxiliary repair guides also preserve H/V family weighting; not benchmarked as
  a repair improvement. Existing repair algorithms remain experimental.
- 26 regression tests passed, including renderer/repair/untangle and old H adapter.
- Final-code smoke rerun: connected_hv_smoke_v2,16.21s wall /5.93s fit wrapper /
  3.31s core optimizer; identical metrics and zero saved crossing-pair UV separation.
  669 unconnected V sources use the measured global fallback. Final diff check clean.

## Reproduction

```bash
PYTHONPATH=/home/hendrik/villa4/vesuvius/src:/home/hendrik/villa4 /home/hendrik/.venv_las/bin/python -m pytest -q lasagna/tests/test_connected_fiber_surface.py lasagna/tests/test_sliding_fiber_surface.py lasagna/tests/test_fiber_surface_repair.py lasagna/tests/test_fiber_surface_render.py lasagna/tests/test_surface_untangle.py vesuvius/tests/test_fiber_lasagna_surface.py
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/fiber_surface_connected_smoke.json
```

Runner refuses existing output: choose a fresh output_root to repeat. No GPU used.
