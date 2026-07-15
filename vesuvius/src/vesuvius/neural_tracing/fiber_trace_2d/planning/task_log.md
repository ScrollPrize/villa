# 3D Fiber CP Model Variant Task Log

- Implemented the V0 sibling package
  `vesuvius.neural_tracing.fiber_trace_3d`.
- Added Lasagna 3x2 direction helpers in `fiber_trace_3d/direction.py`.
  Channel order is `dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x`.
- Added `FiberTrace3DNet`, a seven-channel `Vesuvius3dUnetModel` wrapper:
  six direction channels plus one sheet/fiber-presence channel.
- Added `FiberTrace3DLoader`:
  - deterministic pseudo-random CP sample stream;
  - JSON/NML fiber loading through the existing 2D fiber parser;
  - dataset-level affine transform support;
  - Lasagna manifest shape validation for normal dataset entries;
  - ordinary CP-centered 3D source-block loading, not strip/slice loading;
  - coordinate-space CP shift, isotropic scale, 3D rotation, and axis flips;
  - value augmentation with torch operations;
  - line-derived Lasagna direction and presence targets;
  - conservative base-volume prefetch from augmentation envelopes.
- Added `fiber_trace_3d/projection.py`, a V0 3D-to-2D evaluation bridge that
  decodes six-channel directions with a deterministic unit-sphere candidate
  table and projects them into a caller-provided 2D frame.
- Added `fiber_trace_3d/train.py` with training, prefetch, benchmark, load-only,
  TensorBoard config/scalar logging, micro-batch support, and current/best
  snapshots.
- Added checked-in 3D configs:
  - `fiber_trace_3d/configs/loader_example.json`;
  - `fiber_trace_3d/configs/train_s1a_nml_all.json`.
- Added focused tests in
  `vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`.

## Deviations / V0 Boundaries

- Shared common module extraction was kept minimal for reviewability. The 3D
  path reuses existing fiber parsing, Lasagna manifest validation, transform
  loading, U-Net helpers, and Zarr cache behavior, but does not move large 2D
  loader internals into a new `fiber_trace_common` module yet.
- Smooth 1D/2D/3D displacement fields, anisotropic arbitrary blur, and ringing
  artifact augmentation are documented as future work. Non-zero 3D shear/ringing
  keys are rejected instead of silently accepted.
- The 3D-to-2D projection bridge is implemented as a helper and covered by a
  synthetic test. Full Trace2CP metric wiring for real 3D checkpoints remains a
  later integration step.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/__init__.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/direction.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/model.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/projection.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/tests/neural_tracing/test_fiber_trace.py`
  passed: 325 tests.
