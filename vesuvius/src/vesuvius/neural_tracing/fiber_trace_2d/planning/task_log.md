# Task log: fixed nonlinear uint16 fiberlet costs

- Raw-total per-chunk uint8 previously produced three failures versus two for
  float. The rejected adaptive density variant produced five failures because
  outliers yielded a median density step of `0.0363` and a maximum of `0.563`.
- A discarded fixed-sqrt uint8 sweep produced 4, 4, 2, and 4 failures for
  ceilings 1, 2, 4, and 8. Even ceiling 4 was considered too close to the
  precision limit, so none of those scenarios remain in the implementation.
- The retained mapping is
  `q = round(65535 * sqrt(clamp((total / length) / 256, 0, 1)))` and
  `decoded_total = 256 * (q / 65535)^2 * length`.
- Full Paris4 input: `fiber_s1_002.lasagna.json` and
  `dj_20260805T025256484_000003.json`; radius 768 base voxels, 32 threads,
  beam 16, exact search, lookahead 384 prediction voxels, checkpoint distance
  48 prediction voxels.
- Results: the exact-geometry baseline, compact-direction float-cost control,
  and compact-direction fixed-sqrt uint16 scenario all produced two failures.
  The uint16 scenario and compact-float control have identical failure arcs
  `42747.298` and `44748.209` base voxels.
- Fixed-sqrt uint16 line distance versus the exact-geometry baseline: mean
  `1.1717`, median `0.1716`, maximum `71.7780` base voxels. The compact-float
  control gives `1.1743`, `0.1724`, and `71.7780`, showing that these differences
  are dominated by compact direction geometry rather than uint16 cost storage.
- Fixed-sqrt uint16 replay-to-reference distance: mean `5.6107`, median
  `3.5495`, maximum `71.5886` base voxels. The compact-float control gives
  `5.6147`, `3.5520`, and `71.5886`.
- The warm cache remained at 28,959 files with fingerprint
  `1d0810a4cecf8017697adc0f3a20d41b179947c7f07aac6cb0a8a6f32cc4ec85`.
  Fixed sqrt uint16 cost evaluation used zero compact cost-range chunks and generated
  no cost-specific cache namespace.
- Build: `cmake --build volume-cartographer/build --target vc_fiberlets
  test_fiberlet_storage test_fiberlet_paths test_fiber_replay -j32`.
- `test_fiberlet_storage`: 15 cases passed. `test_fiber_replay`: 12 cases
  passed. `test_fiberlet_paths` retains exactly 298 known fixture failures; no
  additional failure was introduced.
