# Task Log: staged fiberlet anchor acceleration

## Baseline

- Commit: `73fe64e09` (`Use float peak scoring for fiber anchors`).
- Command:
  `volume-cartographer/build/bin/vc_fiberlets fiberlet-replay /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fibers/david/Paris4_fibers/dj_20260805T025256484_000003.json /tmp/fiberlet-replay --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --threads 32 --length 5000 --maximum-iterations 2`.
- Latest measurement: 16.37 seconds total wall, 9.76 seconds anchor wall,
  277.38 seconds anchor CPU, 5.97 seconds fiberlet wall, and 159.68 seconds
  fiberlet CPU.
- Work/quality: 2520 anchors, 48,972 searched / 24,518 accepted fiberlets,
  170,500 sampled voxels, 47,924,048 interpolated/DP node entries, 2 greedy
  failures, and 1 fiberlet failure.

## Workflow Notes

- Implement one checkpoint at a time and stop after reporting its benchmark.
- Small numeric differences are allowed; accepted populations and replay
  quality remain explicit review inputs.
- Independent subagent plan review was not run because this session has no
  user authorization to delegate work. The plan was checked directly against
  `AGENTS.md`, the current specs, and the previous measured task record.

## Checkpoint 1: Tile-Owned Compact Observations

- Production extraction now creates one compact float32 observation per tile
  voxel, normalizes each sampled direction once, and reuses those records via
  canonical-order 32-bit cell indices. A parallel byte vector preserves the
  previous cell-halo gradient-validity rule. The expanded public vector API and
  indexed production path instantiate the same fitter.
- The memory budget includes compact tile records and maximum cell index/
  validity scratch. Temporary decoded samples and gradient caches are released
  before cell fitting.
- Three canonical runs produced identical work/quality populations: 2520
  anchors, 48,972 searched / 24,518 accepted fiberlets, 170,500 sampled voxels,
  47,924,054 DP node entries, 2 greedy failures, and 1 fiberlet failure.

  | metric | minimum | median | maximum | baseline |
  |---|---:|---:|---:|---:|
  | total wall | 14.54 s | 14.73 s | 15.18 s | 16.37 s |
  | total CPU | 385.20 s | 387.73 s | 401.72 s | 439.28 s |
  | anchor wall | 8.03 s | 8.16 s | 8.52 s | 9.76 s |
  | anchor CPU | 224.27 s | 224.72 s | 230.33 s | 277.38 s |
  | fiberlet wall | 5.88 s | 5.93 s | 6.02 s | 5.97 s |
  | fiberlet CPU | 158.76 s | 160.83 s | 162.01 s | 159.68 s |

- Relative to the single recorded baseline run, median total wall improved
  10.0%, median anchor wall improved 16.4%, and median anchor CPU improved
  19.0%. Observation-construction worker time fell from 41.89 to a 17.08-17.20
  second range; final-evaluation and setup time also fell because directions
  are normalized once per tile.
- Float normalization changed seed work slightly (95,344 to 95,450 seeds) and
  a few downstream DP counts, but did not change retained populations or replay
  failures. User review is required before checkpoint 2.
