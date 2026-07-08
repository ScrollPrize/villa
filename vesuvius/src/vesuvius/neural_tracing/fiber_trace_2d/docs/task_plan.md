# Task Plan: 2D Fiber Strip Augmentations

## Scope

Implement augmentation support for the existing fiber-strip patch loader.

The loader must keep using the Lasagna-manifest data path and the current strip coordinate generation. This plan only adds augmentation behavior and a debug visualization mode for it.

## Requirements From `task.md`

- Implement all augmentations for strip patches.
- Add an augment debug mode that loads only the center strip patch for each sample.
- In augment debug mode, apply every augmentation once per sample and save a contact sheet.
- Include one unaugmented sample in the contact sheet.
- The second half of the contact sheet must show all augmentations applied together as they would be during training.
- All geometric augmentations must modify coordinates before sampling from the already loaded strip representation.
- Zarr sampling must still happen only once for the center strip patch.
- All augmentations that happen after loading from Zarr must be GPU based. For this task, that means image/value augmentations run on GPU.
- Contact sheets must draw the fiber line over each patch at 50 percent opacity. The line is normally straight, and geometric distortion augmentations must distort the overlay consistently with the image.

## Implementation Plan

### 1. Define Augmentation Config

Add explicit config keys for the augmentation families used by the strip loader.

Keep defaults conservative and deterministic for tests:

- `augment_enabled`
- `augment_seed`
- geometric controls:
  - horizontal shift along the strip
  - vertical strip-z shift
  - in-plane rotation or shear if supported by the strip coordinate representation
  - horizontal flip along the fiber direction
  - vertical flip across strip-z
- value controls:
  - brightness/offset
  - contrast/scale
  - additive noise
  - optional blur/sharpen only if already available without adding dependencies

Do not add training-only hooks for future augmentation types unless they are implemented in this task.

### 2. Split Augmentation Into Two Stages

Use two explicit stages:

- coordinate augmentations: transform the 2D sampling coordinates inside the loaded center strip patch.
- image augmentations: transform sampled image values after coordinate augmentation on GPU.

The center strip patch is loaded once from Zarr. Every geometric variant samples from that loaded patch, not from the remote volume again.
Any augmentation after the Zarr-loaded patch becomes an image tensor must use torch tensor operations on the selected device, not CPU numpy/PIL-style processing.

### 3. Implement Coordinate-Space Sampling

Add a small local sampler for the loaded strip patch:

- input: loaded strip image, loaded valid mask, 2D float coordinates.
- output: augmented image and augmented valid mask.
- image interpolation: bilinear.
- valid mask interpolation: nearest or thresholded bilinear with explicit invalid handling.
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

- load the center strip patch once per sample.
- render one unaugmented patch.
- render one patch for each individual augmentation.
- render the combined-training augmentation variants in the second half of the contact sheet.
- draw the corresponding fiber line overlay at 50 percent opacity on every rendered patch.
- save JPG contact sheets only.

The output should make it clear which images are individual augmentations and which are combined training-style augmentations.
The debug path may move final images back to CPU only for JPG encoding after GPU augmentation has completed.
The line overlay can be rasterized with OpenCV or an equivalent existing image utility, but it must use the same coordinate transform as the geometric augmentation so distorted patches and overlays stay aligned.

### 6. Wire Training Loader

When augmentations are enabled for normal loading:

- load the center strip patch once.
- draw deterministic augmentation parameters for that sample.
- apply coordinate augmentations through the local strip-patch sampler.
- move the loaded patch tensor to the training device.
- apply image/value augmentations afterward using GPU torch operations.
- return the augmented image and valid mask in the existing batch structure.

Keep non-augment mode byte-for-byte equivalent where possible.

### 7. Tests

Add focused tests for:

- augmentation config parsing and defaults.
- Zarr/base patch load count remains one in augment debug mode.
- unaugmented debug sheet entry equals the center strip patch.
- individual geometric augmentations alter coordinates, not the source Zarr request.
- combined augmentation is deterministic for a fixed seed/sample.
- image/value augmentations run on torch tensors on the configured device after Zarr loading.
- contact-sheet fiber line overlays are present and follow the same geometric transform as the image.
- valid mask marks out-of-bounds augmented pixels invalid.
- disabled augmentations preserve current loader behavior.

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

## Fulfillment

- `task.md`: "implement all the augmentations" - add implemented config-backed geometric and value augmentation paths.
- `task.md`: "augment mode" - add a runner mode that writes augmentation contact sheets.
- `task.md`: "load just the center strip-patch" - debug mode samples the base strip once and augments from that loaded patch.
- `task.md`: "only once per sample" - deterministic one-pass individual augmentation rendering per sample.
- `task.md`: "one sample ... not-augmentationed" - first contact-sheet entry is unaugmented.
- `task.md`: "second half ... all augmentations applied" - contact sheet includes combined training-style augmentations after individual views.
- `task.md`: "geometric augmentations should happen on the coords" - geometric augmentation is implemented as 2D coordinate transforms before local patch resampling.
- `task.md`: "sampling is still only once from the zarr" - remote/base Zarr read happens before augmentation and is not repeated for variants.
- `task.md`: "after loading from zarr should be gpu based" - image/value augmentations operate on torch tensors on the selected device after the single Zarr load.
- `task.md`: "contact sheets should visualize the fiber line" - each debug patch gets a 50 percent opacity line overlay transformed with the same geometry as the patch.
