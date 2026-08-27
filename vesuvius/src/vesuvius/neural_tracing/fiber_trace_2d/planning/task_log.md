# Task Log: Per-fiber BP constraint-consistency diagnostics

## 2026-08-27

- Replaced the previous BP implementation notes with this focused diagnostic
  task as required by the subproject workflow.
- Defined hard mismatch only over resolved endpoints and retained unresolved
  degree/strength separately so ambiguous BP outputs are not mislabeled as
  definite violations.
- Chose the exact merged BP factors for all metrics, preventing duplicate
  measurements from changing graph degree while preserving their summed
  evidence in strength-weighted values.
- The centered-384 experiment will use natural BP and a BP-only CLI path. The
  fixed-target balance draft does not implement the later clarified minimum
  H/V population prior and is not used for this experiment.
- Independent review identified divergence risk from duplicating the labeling
  filter, undefined-rate bias, overstated probability semantics, and the lack
  of a direct separation statistic. The implementation now shares the selector,
  emits `NA` for undefined values, names the independence proxy explicitly,
  reports neighbor certainty with support balance, and calculates tie-aware
  Mixed-vs-trusted AUROC.
- Centered-384 full-Mixed BP-only run: 179 fibers (50 Direction1, 45 Direction2,
  84 Mixed), 1324 unique factors, 5 components, 4 isolates. Natural BP
  converged in 42 message iterations and 0.000522 s; the complete command took
  0.08 s wall, 0.58 s user, 0.06 s system, and 66,444 KiB peak RSS.
- The strongest tested separation was the soft same-label proxy (AUROC
  0.646835), followed by hard mismatch rate (0.633989), strength-weighted hard
  mismatch (0.623754), and neighbor support balance (0.605135). These are weak
  to moderate signals rather than reliable standalone Mixed detectors.
- Validation commands:
  - `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_fiberlet_crop_trace -j32`
  - `volume-cartographer/build/bin/test_fiberlet_crop_trace` (56 cases passed)
  - `git diff --check`
- The BP-only experiment used
  `volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation /home/hendrik/vesuvius/crop_traces_central_384.zarr --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --output /home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/384/384 --direction-dominance 0.9 --piece-length 1000000000 --perpendicular-only --bp-only`.
