# Task Plan: 2D Fiber Strip Loader Correctness And Sampling

## Scope

Fix the 2D fiber-strip loader so it follows the original VC3D/Lasagna data path instead of reimplementing slow or divergent sampling behavior in Python.

The current strip images look correct enough to remove the temporary planar debug slices. The primary work is to replace the Python zarr point sampler with the VC3D chunked coordinate sampler, align normal/frame construction with VC3D/Lasagna, make prefetch use the exact coordinates that runtime sampling will use, and keep existing augmentation behavior on top of that corrected sampling path.

## Requirements

- Use the Lasagna manifest path only for dataset volume/channel discovery.
- Use VC3D side-strip/surface/segment coordinate semantics for strip construction.
- Use VC3D chunked coordinate sampling for image reads, not Python zarr advanced indexing over 8 trilinear corners.
- Use VC3D/Lasagna normal sampling semantics instead of a one-normal-per-control-point approximation where possible.
- Keep `base_volume_scale` semantics unchanged: it selects the zarr level and the output pixel scale.
- Keep geometric augmentations as coordinate-space transforms before image sampling.
- Keep image/value augmentations after volume sampling as GPU torch operations.
- Represent training/debug fiber lines as transformed output pixel coordinates, not raster masks.
- Do not apply geometric transforms by image-space resampling of any image, valid mask, line mask, or label mask.
- Remove planar debug slice sampling and output from loader batches and runner/debug exports.
- Make prefetch derive chunk keys from the exact final coords used by normal runtime sampling, including geometric augmentations when enabled.
- Preserve deterministic sample and augmentation selection.
- Keep contact-sheet JPG augmentation visualization with labels and 50 percent line overlay.

## Implementation Plan

### 1. Introduce A VC3D Sampling Bridge

Add a narrow Python-facing wrapper around the VC3D coordinate sampler used by line probe rendering:

- Input: selected zarr/volume handle, selected level, `H x W x 3` coordinates, and valid mask.
- Output: sampled image and coverage/valid mask.
- Sampling mode: trilinear for now, matching current output semantics.
- Internal implementation should use VC3D `IChunkedArray` plus `ChunkedPlaneSampler::sampleCoordsFineToCoarse` or the equivalent `readInterpolated3D` path.
- The bridge must expose dependency collection for the same coords using VC3D chunk dependency logic, so prefetch and runtime sampling agree.
- The bridge should own the chunk-cache-backed volume object, not repeatedly reopen it per patch.

Keep the Python API small enough that the loader only asks for:

- `sample_coords(coords_zyx_base, valid_mask) -> image, valid_mask, stats`
- `chunk_requests_for_coords(coords_zyx_base, valid_mask) -> chunk requests`

### 2. Replace Python Zarr Point Sampling

Remove `_sample_array_trilinear()` and `_read_array_points()` from the runtime image path.

Replace all image reads with the VC3D sampling bridge:

- normal strip patches;
- center strip patches;
- augmented center strip patches for contact sheets.

Do not keep a fallback Python sampler for production paths. Tests can use a fake bridge where needed.

### 3. Remove Planar Debug Slice Path

Delete planar debug outputs from the 2D loader:

- remove `planar_images`, `planar_coords_zyx`, and `planar_valid_mask` from `FiberStrip2DBatch`;
- remove `build_planar_side_strip_patch_grid()` usage from `build_sample()`;
- remove runner prints/exports for planar debug data;
- remove tests that assert planar debug output shape/content.

Leave the side-strip path as the only sampled patch path.

### 4. Align Normal And Frame Construction With VC3D/Lasagna

Stop using a single CP normal repeated over every line point as the strip-frame normal field.

Use VC3D/Lasagna semantics:

- sample Lasagna normals for all line points needed to build the strip frame;
- preserve sign ambiguity handling as done by VC3D `LineViewBuilder`;
- resolve invalid/missing normals the same way as VC3D;
- build side-strip frames from those per-point samples;
- keep exact control-point membership validation.

Preferred implementation is to expose/reuse VC3D/Lasagna `LineModel`, `NormalSampler`, and `LineViewBuilder` behavior. A Python port is acceptable only if it is explicitly tested against VC3D fixtures and kept byte/geometry close.

### 5. Keep Coordinate Augmentations Before Sampling

Retain the current augmentation model, but make final coords the single source of truth:

- build oversized side-strip coords;
- apply deterministic geometric augmentation in strip pixel space;
- produce final `H x W x 3` read coords;
- produce transformed fiber-line output pixel coordinates from the same geometric transform;
- pass those final coords to the VC3D sampler exactly once per rendered patch;
- apply value augmentations on torch tensors after sampling.

Avoid image-space resampling of already loaded volume patches for geometric transforms. The same rule applies to masks and line targets: do not transform a rasterized line or label mask. Compute the transformed line coordinates first, then rasterize only at the final consumer boundary.

### 6. Replace Raster Line-Mask Overlay With Coordinate Drawing

Remove the current debug overlay path that creates a center-line raster mask and transforms it with `grid_sample`.

Instead:

- keep the strip centerline as pixel coordinates in the unaugmented strip parameterization;
- map those coordinates through the same geometric augmentation mapping used to derive sampled image coords;
- draw the resulting coordinate polyline directly into the contact-sheet cell at fixed screen-space thickness and 50 percent opacity;
- expose the transformed line coordinates in the loader result if needed for training targets;
- use those coordinates for future line/label supervision so target pixels are aligned with the image sampled from the same coordinate transform.

### 7. Make Prefetch Exact

Change prefetch to use the same coord generation function as runtime:

- for non-augmented samples, collect chunks from the final non-augmented coords;
- for augmented samples, collect chunks from the deterministic augmented coords for that sample/offset;
- include all strip-z offsets that runtime will load;
- deduplicate using VC3D chunk keys;
- skip existing cached chunks before reporting progress;
- report missing count, downloaded count, MiB/s, and ETA.

Do not prefetch Lasagna-local normal channels as remote base-volume chunks.

### 8. Update Batch And Runner Interfaces

Keep batch output focused on training data:

- images;
- coords;
- valid masks;
- transformed line coordinates/targets when enabled for training or debug;
- strip offsets;
- control-point metadata;
- cache/sampling stats.

Remove planar debug fields and any CLI output referring to them.

Keep `--augment-vis` writing JPG contact sheets only. Contact sheets should still show:

- min-limit row;
- max-limit row;
- all-random row;
- unaugmented/base sample;
- labels;
- 50 percent fiber-line overlay drawn from transformed line coordinates with fixed visual thickness.

### 9. Tests

Update focused tests to cover the corrected contract:

- loader no longer returns planar debug arrays.
- runtime sampling calls the sampler bridge once per side-strip patch.
- Python trilinear zarr sampler is not used by runtime paths.
- fake sampler bridge receives coords already scaled to the selected zarr level.
- `base_volume_scale` still controls both level and pixel spacing.
- augmented loader passes final augmented coords to the sampler bridge.
- prefetch uses the same final coords as runtime, including augmentations.
- prefetch deduplicates and skips existing cached chunks.
- normals are sampled for line points, not just repeated from the CP.
- side-strip coordinates match VC3D/Lasagna fixtures within explicit tolerance.
- contact sheet still writes JPG, labels, and coordinate-drawn line overlay.
- geometric line augmentation returns transformed line coordinates, not a resampled raster line mask.
- scale/rotation/shear do not change debug overlay drawing thickness except through intended coordinate position changes.
- value augmentations still run through torch on the configured device.
- deterministic sample and augmentation selection remains stable.

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

If the VC3D bridge adds C++ bindings or build targets, also run the smallest available VC3D sampler/LineViewBuilder tests that cover:

- `LineViewBuilder`;
- `ChunkedPlaneSampler`;
- coordinate dependency collection;
- trilinear coordinate sampling.

## Fulfillment

- Original spec: "use VC3D side-strip/surface/segment sampling semantics" - strip coordinates and frames come from VC3D/Lasagna behavior, not an ad hoc Python approximation.
- Original spec: "use vc3d sampling methods" - image loading goes through the VC3D chunked coordinate sampler.
- Original spec: "prefetch approach" - prefetch uses the exact final coords and VC3D chunk dependencies before sampling.
- Original spec: "do not use neural_tracing crop loading" - loader remains coordinate-strip based and does not use 3D crop readers.
- Current task: "geometric augmentations on coords" - augmentations modify coords before VC3D sampling.
- Current task: "oversized coords for augmentation" - augmented coords are derived from oversized strip-coordinate sources.
- Current task: line must match image by pixel coords - transformed line coordinates are generated from the same geometric transform as image sample coordinates.
- User correction: "do not transform line pixels" - raster line-mask grid sampling is removed; the line is drawn only from transformed coordinates.
- Current task: "image augmentations GPU based" - brightness, contrast, gamma, noise, and blur remain torch operations after sampling.
- User correction: "remove planar dbg slice sampling and output" - planar debug fields and sampling are deleted from loader and runner.
- User correction: "current sampling works but is too slow" - preserve output semantics while replacing the slow Python sampler with VC3D chunked sampling.
