# Task Log: reduce fiber-anchor NMS radii

## Discovery

- Current transverse NMS uses `localWindowRadiusPredictionVoxels`, whose
  default is one cell side (4 prediction voxels). Reducing it in place would
  also shrink the peak-search/refinement domain, contrary to the NMS-only
  request.
- Current longitudinal NMS is assigned half the cell side by CLI resolution,
  giving 2 prediction voxels for the default four-voxel cell.
- The final anchor JSON is consumed strictly by `FiberPaths.cpp`; adding an
  independent field therefore requires coordinated writer, reader, and fixture
  changes.

## Independent Review

- Review confirmed that the CLI resolver must stop replacing the longitudinal
  default from cell size and requested coverage across cell sizes 2/4/8.
- Review clarified that external-context pivot reach must retain the full
  refinement displacement plus the narrower NMS ellipsoid; only the ellipsoid
  transverse term changes.
- Review requested inclusive direct NMS boundary tests, strict writer/reader
  coverage, and base-voxel reporting for both effective radii.

## Implementation

- Added `nmsTransverseRadiusPredictionVoxels = 2` and changed the longitudinal
  default to 1 prediction voxel. Removed the CLI resolver's cell-size-derived
  longitudinal assignment.
- NMS neighbor tests and spatial bins use the independent transverse radius.
  External context remains refinement window plus `hypot(transverse,
  longitudinal)` so narrowed suppression does not under-enumerate fitted
  anchors from outside the selected crop.
- Final and diagnostic anchor JSON serialize the transverse field; the strict
  `FiberPaths.cpp` reader requires and validates it without compatibility.
- CLI output reports both effective radii in base voxels and `--window` is now
  described as refinement-only.

## Validation

- `cmake --build volume-cartographer/build -j32 --target test_fiber_anchors test_fiberlet_paths test_fiber_replay vc_fiberlets`
  passed.
- `ctest --test-dir volume-cartographer/build --output-on-failure -R 'test_fiber_(anchors|replay)|test_fiberlet_paths' -j3`
  passed: 3/3.
- `PYTHONPATH=vesuvius/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q vesuvius/tests/test_view_fiber_presence.py`
  passed: 53 tests.
- Real extraction used the Paris4 fiber manifest with
  `--crop 21786,18266,54388,128,128,128 --threads 32`. It completed in 2.04 s,
  reported refinement/NMS radii of 32/16/8 base voxels, retained 58 anchors,
  and suppressed 47. Its strict JSON stores refinement 4, transverse NMS 2,
  and longitudinal NMS 1 prediction voxels.
- `git diff --check` passed.
