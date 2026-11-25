# Plan

- Delete the explicitly selected scripts/modules and remove stale references.
- Replace local winding bilinear/trilinear interpolation with vc_reader sampling; retain erosion, query order, and application phase validity policy.
- Replace Torch grid_sample and handwritten bilinear lookup in the audited surface-coordinate, augmentation, and CT plane paths with vc_reader.torch.sample_array and resident.GridField. Keep coordinate order, masks, output shape, and gradients explicit. Test fractional and boundary coordinates, including singleton dimensions and nearest ties.
- Route surface rendering through the existing native grid renderer, exposing any necessary in-memory geometry adapter in C++ and Python rather than implementing rendering in Python. Replace the selectable Python surface-transfer implementation with SurfaceTransfer.
- Rebuild existing native targets without installing dependencies. Run focused existing and regression tests. Report numerical differences and environment limitations.

## Spec update

Update specs.md sparse map lookup ownership to name vc_reader.torch.sample_array. No training objectives or dataset policy changes are intended. Shared native sampling owns interpolation; use the shared sampler default numeric and boundary policies as authorized.

## Docs updates

Update sampling documentation with index coordinate order and shared backend ownership. Remove references to deleted tools. Update vc_reader capability documentation if its exposed rendering API needs an adapter.

## Changelog

Record centralized sampling and rendering plus deleted obsolete tooling in planning/changelog.md.

## User-directed behavior

Use vc_reader defaults, as explicitly requested after the plan review. No compatibility paths: nearest ties round up, resident sampling uses float32 with zero fill, surface interpolation defaults to bicubic, and rendering reads volume fill outside bounds. Preserve application masks, query ordering, and tensor layouts. Prepared points/normals must not be reinterpolated by the renderer.
