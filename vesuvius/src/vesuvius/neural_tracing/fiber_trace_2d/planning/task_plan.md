# Coordinate-Space Geometric TTA Plan

## Goals

- Make the code and specs unambiguous: geometric augmentation is always a
  coordinate transform before patch sampling/slicing.
- Remove all geometric image-space augmentation functions from `fiber_trace_2d`;
  they must not exist anywhere in this subproject.
- Rebuild Trace2CP TTA so each TTA image is sampled from volume coordinates,
  not warped from an already sampled base strip image.
- Add `--vis-tta` debug output for inspecting individual Trace2CP TTA slices
  and their transformed base-strip footprint.

## Non-Goals

- Do not change value-only augmentation semantics.
- Do not change regular training batch sampling except where stale image-space
  geometric helper code is removed.
- Do not change the Trace2CP scoring definition from the current bidirectional
  averaged score.

## Spec Update

- Add a hard normative rule near the augmentation spec:
  **we must never do geometric augmentations as image space operations.**
  Geometric augmentation is always a coordinate manipulation before
  sampling/slicing the patch from the volume.
- State that no helper/function/API may perform geometric augmentation by
  warping, rotating, scaling, shearing, translating, or flipping an already
  sampled image. Such functions must not exist in `fiber_trace_2d`.
- State that image-space operations are allowed only for value-only changes
  after sampling, such as brightness, contrast, gamma, noise, and blur.
- Update Trace2CP TTA spec:
  - TTA variants must be built by constructing augmented coordinate grids and
    sampling the volume at those coordinates.
  - TTA must not call or implement any image-space geometric warp on the
    already sampled base strip.
  - The base segment strip remains the reference geometry for scoring/tracing.
  - Each TTA output strip must be sized to contain the transformed base strip
    footprint under that TTA. Invalid regions outside the expanded coordinate
    strip/volume are allowed and should stay invalid/black.
  - `--vis-tta` writes per-TTA debug images when TTA is active, with the
    transformed base-strip corner quad/outline drawn on each TTA image.

## Docs Updates

- Update `docs/code_structure.md` to document:
  - training coordinate augmentation path;
  - Trace2CP coordinate-sampled TTA path;
  - removal of image-space geometric helper functions;
  - `--vis-tta` output files and what the corner outline means.
- Update `planning/task_log.md` with implementation notes, deviations, and
  validation results.
- Update `planning/changelog.md` after implementation.

## Implementation Plan

### 1. Delete All Image-Space Geometric Warp Functions

- Delete geometric image-space augmentation functions entirely. These functions
  must not exist anywhere in `fiber_trace_2d`.
  Known current offenders include:
  - `_warp_patch_by_matrix`
  - `_warp_patch_by_augment_params`
- Remove or rewrite tests and expectations that rely on geometric image
  warping.
- Keep value-only augmentation functions in `augmentation.py`; these are not
  geometric transforms and can still run after sampling.
- Search for any remaining function that applies geometric transforms to an
  already sampled image/tensor, including uses of `grid_sample`, `rot90`,
  affine image matrices, image flips, image resizing, or PIL/OpenCV geometric
  transforms. Delete any such geometric image-space code.
- Leave `grid_sample` only where it resamples coordinate tensors/maps before
  volume sampling, performs sparse coordinate-map lookup, or performs
  value-only/non-geometric image processing.

### 2. Add Coordinate-Space Trace2CP TTA Source Construction

- Extend the Trace2CP loader/runner interface with a coordinate-grid builder
  for CP segment strips that can:
  - build the unaugmented base strip as today;
  - build an oversized source coordinate strip around the same selected segment;
  - apply a `FiberStripAugmentParams` geometric transform to coordinates before
    volume sampling;
  - return the sampled TTA image, valid mask, transformed start/target CP
    positions, transformed base-strip corner points, and source/reference maps
    needed by median TTA tracing.
- Prefer reusing the existing fused augmentation transform machinery:
  - `strip_augment_transform(...)`
  - output-to-source coordinate maps for image sampling;
  - source-to-output point maps for CPs, lines, and base-strip corners.
- Avoid introducing any separate geometric math path that can diverge from the
  training augmentation implementation.

### 3. Size TTA Output To Preserve Base Strip Footprint

- For each Trace2CP TTA variant:
  - define base-strip corners in the unaugmented base strip coordinate frame;
  - transform those corners through the same source-to-output augmentation map
    used for CP/line mapping;
  - compute an output canvas large enough to contain the transformed corner
    footprint plus the existing RF/visual margin;
  - translate the TTA output coordinate frame so the transformed footprint is
    inside the image.
- At 45-degree rotations and similar cases, allow the output TTA image to be
  substantially wider/taller than the base strip.
- Keep invalid areas black in exported/debug images and false in valid masks.
- Record each TTA image shape and transformed corner coordinates in the summary
  or TTA debug metadata.

### 4. Use Coordinate-Sampled TTA In Median Trace2CP

- Replace the current Trace2CP TTA branch:
  - currently: base strip image -> image-space warp -> model prediction;
  - target: augmented coordinate grid -> volume sample -> model prediction.
- Keep the reference base strip direction field as the first median field.
- For each TTA field, map reference trace points into TTA coordinates using
  the coordinate transform/mapping produced by the TTA builder.
- Map TTA directions back into the reference frame using the same local
  transform/Jacobian approach already used by median TTA, but fed from the new
  coordinate-space maps rather than image-warp source grids.
- Continue bidirectional Trace2CP scoring unchanged after median direction
  tracing.

### 5. Add `--vis-tta`

- Add CLI flag `--vis-tta`.
- Only produce TTA debug slices when TTA is active for the selected mode
  (`--trace2cp-vis --med-tta --vis-tta` initially).
- Export one image per TTA variant, plus optionally a compact contact sheet:
  - base/reference image;
  - each TTA sampled image;
  - transformed base-strip corner outline drawn as four connected line
    segments;
  - start/target CP markers if they are valid in that TTA image.
- Name files deterministically, e.g.:
  - `trace2cp_tta/reference.jpg`
  - `trace2cp_tta/random_000.jpg`
  - `trace2cp_tta/random_001.jpg`
- Include skipped TTA variants in summary text with the reason.

## Testing Plan

- Add a unit test that proves Trace2CP TTA no longer calls image-space
  geometric warp helpers and that those helpers no longer exist.
- Add a synthetic coordinate sampler test where a rotated TTA image is sampled
  from known coordinate-dependent volume values, proving the TTA path samples
  by coordinates rather than warping a base image.
- Add a 45-degree rotation sizing test:
  - transformed base-strip corners fit inside the TTA output shape;
  - output shape expands relative to the base strip when needed.
- Add a test for invalid/black regions outside the expanded coordinate strip.
- Add a `--vis-tta` test that checks debug files are written only when TTA is
  active and that the corner outline path has finite coordinates.
- Keep existing Trace2CP bidirectional scoring tests unchanged.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/augmentation.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Risks And Checks

- Large rotations can create very large TTA canvases for long Trace2CP segment
  strips. Add size logging and consider a future max-canvas guard if memory
  becomes a problem.
- Median TTA mapping relies on reference/TTA coordinate transforms. The tests
  must cover reverse point mapping and direction mapping, not just image
  appearance.
- If any old line-trace debug visualization still relies on image-space
  geometric warp helpers, either port it to coordinate sampling or remove that
  TTA visualization mode in the same change so the hard spec rule is true.
