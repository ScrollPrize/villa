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

## Open Questions

- Need the exact old-s1-to-current affine config/file spelling used in the
  local Lasagna training setup before adding transform keys to the S1A NML
  example config. Checked the available Lasagna run snapshot configs; they only
  point to older checkout config paths that are not present locally.

## Validation

- `python -m json.tool vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example_s1a_nml.json >/tmp/loader_example_s1a_nml_pretty_check.json`
