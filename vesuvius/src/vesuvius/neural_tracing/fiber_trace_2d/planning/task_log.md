# NML Fiber Loading With Affine Volume Transforms Task Log

## Planning Notes

- Current loader record construction only accepts VC3D JSON fibers through
  `load_vc3d_fiber`.
- NML example structure is XML with `<thing>`, `<nodes>`, and `<edges>`.
  Edge ordering is required; XML node order is not a reliable fiber order.
- `vesuvius.data.affine` supports Vesuvius registration `transform.json`
  documents with `p_fixed = M @ p_moving` in XYZ.
- `lasagna/tifxyz_lasagna_dataset.py` supports Lasagna-style inline
  `transform` plus `transform_invert` and builds ZYX scaled affines for patch
  coordinate mapping.
- Plan chooses to normalize JSON and NML inputs into `Vc3dFiber` before the
  existing strip-coordinate, prefetch, training, and Trace2CP paths.
- Added `configs/loader_example_s1a_nml.json` from `loader_example.json`.
  It replaces only the training `datasets` entry with
  `/home/hendrik/business/aiconsulting/vesuviuschallenge/data/train_fibers/fiber_vols/fibers_s1a_*.nml`
  and keeps PHercParis4 base-volume / Lasagna manifest settings.
- Inspected `fibers_s1a_00497z_01497y_03997x_256_v00.nml`; its node
  coordinates and user bounding boxes are absolute source-scan coordinates
  matching the filename offsets, not local cube coordinates.

## Follow-up Search

- Rechecked with `s1 == s1a == scroll1a`. The concrete existing transform is in
  `/home/hendrik/business/aiconsulting/vesuviuschallenge/villa2/lasagna/configs/tifxyz_train_s3_dbg.json`,
  dataset entry for
  `s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr`.
- The same matrix is present in
  `/home/hendrik/business/aiconsulting/vesuviuschallenge/villa2/lasagna/tmp/transform.json`.
  Lasagna uses the inline 3x4 `transform` with `transform_invert: true`.
- Added that exact inline `transform` plus `transform_invert: true` to
  `configs/loader_example_s1a_nml.json`.

## Implementation Notes

- Added NML parsing in `fiber_json.py`. Each edge-ordered open simple path
  component becomes one normalized `Vc3dFiber`; branch/closed/malformed
  components are skipped or rejected with diagnostics.
- Added `load_fiber_file` so `.json` and `.nml` inputs share one loader-facing
  API and both return normalized `Vc3dFiber` objects.
- Added loader dataset transform parsing for `fiber_transform_json`,
  `fiber_transform_json_path`, inline `fiber_transform`, and Lasagna-compatible
  `transform`, plus corresponding invert flags.
- Transform matrices are XYZ 3x4/4x4 and are applied once immediately after
  parsing, before bounds checks, identity/cache keys, prefetch, training, or
  Trace2CP use the fiber.
- Updated flat path lookup so an NML file with multiple simple components
  returns all matching flat CP indices instead of only the first record.
- Kept the existing deterministic sample key format unchanged after a full-test
  failure showed that adding `fiber_identity` changed legacy JSON sample order.
  Transform-aware record/cache identity still includes the transform, and
  transformed coordinates change sample keys when the matrix is non-identity.
- Added unit tests for edge-ordered NML parsing, multiple components, branch
  diagnostics, inline transforms, transform.json inverse handling, and loader
  integration with fake zarr/Lasagna data.

## Validation

- `python -m json.tool vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example_s1a_nml.json >/tmp/loader_example_s1a_nml_pretty_check.json`
- `python -m json.tool vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example_s1a_nml.json`
  - Result: parsed successfully after adding the S1A transform.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'nml or transform_json'`
  - Result: 5 passed, 259 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - First result: 2 failed, 262 passed because adding `fiber_identity` to the
    deterministic sample key changed existing JSON sample order.
  - Final result after preserving the legacy sample key: 264 passed.
