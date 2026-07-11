# TTA Reference Mapping Performance Plan

## Spec Update

- TTA reference-to-field point mapping must use the augmentation forward map,
  sampled directly at the reference point.
- Median-TTA tracing must not map reference points into TTA images by scanning
  full dense coordinate images.

## Implementation

- Return both maps from TTA patch construction:
  - output-to-reference `source_xy_grid` for mapping TTA directions/traces back;
  - reference-to-output `reference_to_tta_xy_grid` for mapping reference trace
    points into TTA image coordinates.
- Update `_TtaDirectionField` and median-TTA sampling to use the direct forward
  map.
- Remove the dense nearest-grid scan helper from the tracing path.

## Tests

- Add/update tests proving reference-to-TTA mapping is direct and that the
  dense nearest scan helper no longer exists.
- Run py_compile and focused pytest.

## Docs

- Update specs/docs/status/task_log/changelog.
