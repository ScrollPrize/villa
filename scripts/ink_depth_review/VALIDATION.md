# Validation record

Date: 2026-09-07. Platform: macOS arm64. Official checkout: `b218ace879d3239a63047e1e2039f327929d283e`.

## Executed

- Python 3.13: 12 regression tests passed on the real downloaded PHercParis4 crop.
- Python 3.14.6 (upstream required version): the same 12 tests passed, and the prepare command generated the review page successfully.
- Tests in the proposed upstream script directory also passed on the same real sample.
- Official `label_zarr.create_label_group` / `create_label_array` produced an exact voxel-for-voxel round trip of the real sample's center-plane prior. See `benchmark.py` and the JSON results.
- TIFF and Zarr versions of the real scan loaded identically. Reviewed export preserved the original scan intensities.
- Export tests covered unknown/background distinction, duplicate indices, negative indices, invalid classes, noninteger indices, mismatched source hashes, empty reviews and overwrite refusal. RGB TIFF input and malformed prior/parameters were rejected.
- Suggestions were deterministic for identical inputs and restricted to the stated search region.

## Browser checks through actual UI

- The page opened with the real scan and three orthogonal views.
- Next suggestion moved the cursor from depth 32 to depth 25 and updated its displayed coordinates.
- A radius-one test brush marked exactly five voxels on the selected depth.
- Undo removed that ink edit; background mode then marked five background voxels.
- Undo returned all test voxels to unknown. Test marks were not retained as scientific annotations.
- Saving without a reviewer and reviewed voxels displayed the refusal message.

The successful browser JSON download/resume round trip has not been independently exercised; the Python export path was tested with schema-equivalent review files. No trained ink model, expert-reviewed 3D annotation set, native tifxyz back-projection, held-out ink accuracy or human time study was completed. The visual examples and counts demonstrate functioning tooling, not ink-discovery quality.
