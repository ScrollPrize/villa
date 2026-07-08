# Task Plan: 2D Fiber Strip Augmentations

## Scope

Implement augmentation support for the existing fiber-strip patch loader.

The loader must keep using the Lasagna-manifest data path and the current strip coordinate generation. This plan only adds augmentation behavior and a debug visualization mode for it.

## Requirements From `task.md`

- Implement all augmentations for strip patches.
- Add an augment debug mode that loads only the center strip patch for each sample.
- In augment debug mode, apply every augmentation once per sample and save a contact sheet.
- Include one unaugmented sample in the contact sheet.
- Augment contact sheets must use three rows: lower-limit examples, upper-limit examples, and all-random combined examples.
- Random augmentation ranges are rotation `+-180` degrees, shear/skew up to `+-1` px per px, offsets up to one quarter of the patch size per axis, scale around `sqrt(0.5)x..sqrt(2.0)x`, smooth curve offset with sampled control offsets and cubic interpolation, brightness `+-0.25` of the valid patch image range, contrast from `0.5x` to `2.0x` around the valid patch center, gamma, valid-range-relative noise std up to `0.125`, and Gaussian blur sigma up to `2.0`.
- All geometric augmentations must modify strip coordinates before image sampling.
- Geometric augmentations must use an oversized strip-coordinate source area so the final patch can be sampled from augmented coordinates without edge or image reinterpolation artifacts.
- Zarr sampling must still happen only once for the center strip patch.
- All augmentations that happen after loading from Zarr must be GPU based. For this task, that means image/value augmentations run on GPU.
- Contact sheets must draw the fiber line over each patch at 50 percent opacity. The line is normally straight, and geometric distortion augmentations must distort the overlay consistently with the image.
- Contact-sheet cells must include a small top label naming the shown augmentation.

## Implementation Plan

### 1. Define Augmentation Config

Add explicit config keys for the augmentation families used by the strip loader.

Use explicit extrema for random training augmentation and deterministic limit examples for visualization:

- `augment_enabled`
- `augment_seed`
- `augment_shift_x=patch_width/4`
- `augment_shift_y=patch_height/4`
- `augment_rotation_degrees=180`
- `augment_shear_x=1`
- `augment_shear_y=1`
- `augment_scale_min=sqrt(0.5)`
- `augment_scale_max=sqrt(2.0)`
- `augment_smooth_offset=8`
- `augment_smooth_offset_stride=16`
- `augment_brightness=0.25`
- `augment_contrast_min=0.5`
- `augment_contrast_max=2.0`
- `augment_gamma_min=0.5`
- `augment_gamma_max=2.0`
- `augment_noise_std=0.125`
- `augment_blur_sigma=2.0`
- geometric controls:
  - horizontal shift along the strip
  - vertical strip-z shift
  - in-plane rotation
  - scale
  - shear/skew
  - smooth curve distortion by offsetting strip columns/rows along the strip-normal image axis with a cubic-interpolated random field
  - horizontal flip along the fiber direction
  - vertical flip across strip-z
- value controls:
  - brightness/offset
  - contrast min/max scale
  - gamma
  - additive noise
  - Gaussian blur

Do not add training-only hooks for future augmentation types unless they are implemented in this task.

### 2. Split Augmentation Into Two Stages

Use two explicit stages:

- coordinate augmentations: build an oversized strip-coordinate source area, transform output patch pixels into that source, apply affine transforms and smooth curve offset in strip-patch coordinates, then sample the volume at the final transformed coordinates.
- image augmentations: transform sampled image values after coordinate augmentation on GPU.

The center strip coordinate source is built once per variant. Geometric augmentation never resamples an already loaded image patch; it derives final volume coordinates first and then samples Zarr at those coordinates.
Any augmentation after the Zarr-loaded patch becomes an image tensor must use torch tensor operations on the selected device, not CPU numpy/PIL-style processing.

### 3. Implement Coordinate-Space Sampling

Add coordinate-space derivation for augmented patches:

- input: oversized strip coordinate grid, oversized valid mask, and 2D float source coordinates.
- output: final augmented volume coordinates and augmented valid mask.
- coordinate interpolation: bilinear over the oversized coordinate grid.
- smooth curve offset: sample deterministic random offsets every `augment_smooth_offset_stride` pixels, cubic-interpolate them across the strip, and add them along the strip-normal image axis before coordinate-grid lookup.
- valid mask interpolation: nearest with explicit invalid handling.
- out-of-bounds pixels: invalid in the mask and visually distinguishable in debug output.

Keep coordinate math in strip-patch pixel units. Do not reinterpret Lasagna normals or rebuild the 3D strip path for augmentation.

### 4. Deterministic Augmentation Selection

Generate augmentation parameters from stable fields:

- base seed
- sample index
- augmentation name
- combined-training augmentation marker

This keeps debug sheets reproducible and prevents call-order-dependent augmentation behavior.

### 5. Add Augment Debug Mode

Extend the runner with an explicit augment visualization mode.

Behavior:

- build an oversized center strip coordinate source per rendered patch.
- render row 1 as lower-limit augmentation examples.
- render row 2 as upper-limit augmentation examples.
- render row 3 as random combined-training augmentation variants with all augmentation families active together.
- draw the corresponding fiber line overlay at 50 percent opacity on every rendered patch.
- draw a small label at the top of every patch naming the augmentation.
- render raw clipped image values without percentile or per-cell normalization so value augmentations remain visible.
- save JPG contact sheets only.

The output should make it clear which images are min limit, max limit, and combined training-style random augmentations.
The debug path may move final images back to CPU only for JPG encoding after GPU augmentation has completed.
The line overlay can be rasterized with OpenCV or an equivalent existing image utility, but it must use the same coordinate transform as the geometric augmentation so distorted patches and overlays stay aligned.

### 6. Wire Training Loader

When augmentations are enabled for normal loading:

- build an oversized strip-coordinate source for the selected patch.
- draw deterministic augmentation parameters for that sample.
- derive final output patch coordinates from the oversized source and sample the volume once at those final coordinates.
- move the loaded patch tensor to the training device.
- apply image/value augmentations afterward using GPU torch operations.
- return the augmented image and valid mask in the existing batch structure.

Keep non-augment mode byte-for-byte equivalent where possible.

### 7. Tests

Add focused tests for:

- augmentation config parsing and defaults.
- Zarr/base patch load count remains one in augment debug mode.
- unaugmented debug sheet entry matches the center strip patch content.
- individual geometric augmentations alter final volume coordinates, not an already loaded image patch.
- geometric augmentations use an oversized coordinate source before volume sampling.
- combined augmentation is deterministic for a fixed seed/sample.
- image/value augmentations run on torch tensors on the configured device after Zarr loading.
- gamma augmentation is applied as a torch tensor operation using the valid patch range so it is independent of raw intensity scale.
- Gaussian blur is applied as a torch convolution on the selected device.
- contact-sheet fiber line overlays are present and follow the same geometric transform as the image.
- contact-sheet cells include visible augmentation labels.
- contact-sheet min/max rows include scale, smooth curve offset, and gamma examples.
- valid mask marks out-of-bounds augmented pixels invalid.
- disabled augmentations preserve current loader behavior.

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

## Fulfillment

- `task.md`: "implement all the augmentations" - add implemented config-backed geometric and value augmentation paths.
- `task.md`: "augment mode" - add a runner mode that writes augmentation contact sheets.
- `task.md`: "load just the center strip-patch" - debug mode restricts sampling to the center strip offset and derives each view from augmented center-strip coordinates.
- `task.md`: "only once per sample" - deterministic one-pass individual augmentation rendering per sample.
- `task.md`: "one sample ... not-augmentationed" - first contact-sheet entry is unaugmented.
- `task.md`: "all augmentations applied as they would be used together" - contact sheet row 3 contains random combined training-style augmentations.
- `task.md`: "three rows ... min, max, all-random" - contact sheet uses lower-limit, upper-limit, and random-combined rows.
- `docs/plan.md`: "affine: rotate, skew, scale, flips" - scale is included as a coordinate-space augmentation.
- `docs/plan.md`: "image: contrast, brightness, gamma, blur, noise" - gamma, blur, and noise are included as GPU value augmentations.
- `docs/plan.md`: "curve distortion ... smooth field ... cubic interpolation" - smooth curve offset is included as a coordinate-space augmentation with deterministic sampled offsets and cubic interpolation.
- `task.md`: "geometric augmentations should happen on the coords" - geometric augmentation is implemented as 2D coordinate transforms before image sampling.
- `task.md`: "oversized area of strip coords" - geometric augmentation derives final patch coordinates from an oversized strip-coordinate source before Zarr sampling.
- `task.md`: "sampling is still only once from the zarr" - remote/base Zarr read happens before augmentation and is not repeated for variants.
- `task.md`: "after loading from zarr should be gpu based" - image/value augmentations operate on torch tensors on the selected device after the single Zarr load.
- `task.md`: "contact sheets should visualize the fiber line" - each debug patch gets a 50 percent opacity line overlay transformed with the same geometry as the patch.
- User follow-up: contact-sheet labels - each debug patch is labeled with its augmentation name.
