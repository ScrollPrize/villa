# Task log: parallel-separate-winding labeling ablation

- Baseline committed as `d7ac464b1` (`Add tightened LP fiber label diagnostics`).
- Baseline centered-384 threshold counts: H/even `128`, H/odd `17`, V/even
  `6`, V/odd `18`, broken `21`; objective `1358.9023` over `3,353` links and
  `27,615` triangles.
- The ablation must filter only model input. Full constraint visualization is
  deliberately preserved for inspection.
- Perpendicular visualization is now required as two disjoint winding classes:
  `[0,0.5)` same and `[0.5,1.5)` separate; exact `0.5` belongs to separate.
- Independent review required an explicit output basename, baseline archival,
  immutable full-report boundary, pinned solver/data/build settings, and both
  predicate-boundary/default-off tests. These are incorporated in the plan.
- Benchmark settings: CMake `Release` build at `volume-cartographer/build`, one
  run, host-default 32 threads, `--lp-relaxation --lp-parallel`, trace artifact
  `/tmp/crop_traces_central_384.zarr`, and normal manifest
  `$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json`.
- `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk
  test_fiberlet_crop_trace -j32` succeeded; `test_fiberlet_crop_trace` reports
  27 passing cases.
- The ablation retained `2,378` of `3,353` links, excluded `975`, and reduced
  LP triangles from `27,615` to `12,186`. Full constraints remain visualized as
  252 perpendicular/same, 1,208 perpendicular/separate, 907 parallel/same, and
  975 parallel/separate links.
- Broken-cost experiments use the same retained graph. Root artifacts contain
  cost `0.9`; cost `0.5` and `0.75` are archived under `broken-0.5/` and
  `broken-0.75/`, while the original full-constraint baseline is under `full/`.

| Broken cost | Objective | Solve | H/even | H/odd | V/even | V/odd | Below-mean activity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 931.5014 | 26.156 s | 31 | 17 | 2 | 38 | 102 |
| 0.75 | 1119.9057 | 64.428 s | 60 | 3 | 1 | 47 | 79 |
| 0.9 | 1190.4724 | 67.157 s | 56 | 8 | 12 | 50 | 64 |

For cost `0.9`, continuous active min/mean/median/max is
`0.5/0.9195195/0.9857019/1`; vertical is
`0/0.4274330/0.4592902/0.5919412`; odd is
`0/0.4253304/0.5/0.5715863`.
