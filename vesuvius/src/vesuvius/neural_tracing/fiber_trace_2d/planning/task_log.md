# Task log: integrate broader peak evidence for fiber anchors

## Requested change

- Increase transverse peak integration to sigma `1.5` prediction voxels.
- Add a still larger along-direction Gaussian so straight-fiber evidence from
  multiple cells contributes with roughly comparable weight.

## Decisions

- The axial default is `1.5 * cell-size`: for default four-voxel cells this is
  sigma `6`, giving weights `exp(-0.5*(4/6)^2) ~= 0.80` one cell away and
  `exp(-0.5) ~= 0.61` at the ends of a centered three-cell span.
- Candidate movement remains two-dimensional in the normal plane; only its
  three-dimensional scoring kernel becomes anisotropic.

## Plan review

- Independent review approved the weighting choice and required the peak stage
  to replace, rather than retain, the old `+-6` axial slab. Peak cutoff is
  therefore `gaussianCutoffSigmas * peakAxialSigmaPredictionVoxels`.
- Halo sizing must take the maximum of the separate broad-fit bound and the
  orientation-independent anisotropic peak bound. Tests will cover evidence
  beyond the old slab and non-default cell-size CLI defaults.

## Implementation

- The final peak response now uses independent transverse and axial Gaussian
  factors. Defaults are `1.5` and `6.0` prediction voxels respectively for the
  default four-voxel cells.
- The peak stage uses three-sigma cutoff bounds in both dimensions; the old
  fixed axial half-width remains exclusive to broad direction refinement.
- The sampling halo is the maximum of broad support and the conservatively
  rotated anisotropic peak support including candidate displacement.
- `--axial-sigma` accepts the peak axial sigma in base voxels. Both anchor CLI
  paths derive its default as `1.5 * cell-side`; config, artifacts, C++ path
  loading, and Python stage loading require a positive finite axial sigma.
- Candidate motion, owner-cell/pivot-plane constraints, broad support
  reevaluation, and NMS are unchanged.

## Validation

### Baseline

- Command: `volume-cartographer/build/bin/vc_fiberlets fiber-replay
  /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json
  /home/hendrik/business/aiconsulting/vesuviuschallenge/data/train_fibers/fibers_test_paul_4/kb_20260605T150824406_000001.json
  /tmp/vc-fiber-replay-integration-{baseline|new}-N --normal-manifest
  /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json
  --fail 0 --after 1 --along 16 --radius 16 --threads 32`.
- Input: the existing Paris 4 small local replay fixture; default build in
  `volume-cartographer/build`; Release; five iterations.
- Anchor-stage seconds: `0.386275`, `0.381347`, `0.390340`, `0.367209`,
  `0.369914`; mean `0.379017`, median `0.381347`, interpolated p95 `0.389527`,
  min/max `0.367209/0.390340`.
- Every run retained two anchors from eight selected cells and 59 NMS-context
  cells.

### Updated measurement

- Same command/input/build/iteration count as baseline.
- Anchor-stage seconds: `1.066030`, `1.051550`, `1.048530`, `1.069560`,
  `1.049650`; mean `1.057064`, median `1.051550`, interpolated p95 `1.068854`,
  min/max `1.048530/1.069560`.
- The broader integration is `2.79x` slower on this small fixture. Every run
  retained three anchors from eight selected cells and 64 NMS-context cells.

### Correctness and strict consumers

- `cmake --build volume-cartographer/build --target test_fiber_anchors
  test_fiberlet_paths test_fiber_replay vc_fiberlets -j32` passed.
- `test_fiber_anchors`: 40 test cases passed.
- `test_fiberlet_paths`: 23 test cases passed.
- `test_fiber_replay`: 2 test cases passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src python -m pytest
  -q vesuvius/tests/test_view_fiber_presence.py`: 48 tests passed in 2.04
  seconds.
- Ruff, Python bytecode compilation, and `git diff --check` passed.
- The regenerated real replay loaded five stages by default and zero with the
  explicit opt-out. The standalone `paths` command strictly loaded its new
  `anchors.json`.
- With prediction-to-base scale `8`, `--cell-size 5` stored the default axial
  sigma as `7.5` prediction voxels; adding `--axial-sigma 32` stored `4.0`,
  confirming base-coordinate override conversion.

## Deviations and limitations

- The requested transverse sigma `1.5` mathematically blends two equal ridges
  separated by only three prediction voxels into a midpoint response. Narrow
  `0.75` regression cases still verify local-mode selection, but the wider
  default intentionally trades that resolving power for greater area
  integration.
- The representative replay changed from two to three retained anchors. Visual
  scientific assessment on the user's larger replay remains required; the
  automated checks establish mechanics, bounds, determinism, and strict format
  behavior, not anchor-quality superiority.
