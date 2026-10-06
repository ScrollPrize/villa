# Width-aware supervision plan

1. Stop approved production runner before edits. Preserve latest saved raw checkpoint and settings.
2. Shared width decode and triangular similarity helper: clamp(1-|w-ref|/(0.2*ref),0,1); invalid/unlabelled reference gives zero. Width gate detached for loss, so only presence receives sideband gradients. Keep GT-only existing denominators (do not renormalize by admitted predictions), single backward and microbatch invariance.
3. Carry original polyline segment indices and clipping fractions (including halo) through the loader. Use shared annotation geometry transformation for source-indexed endpoint widths; no nearest-row reassignment. Extend existing Gaussian rasterizer with a width reference field, interpolated along segment projection. Gaussian voxels use the same winning segment as direction; sideband overlap uses nearest contributing segment. Missing width remains zero/unlabelled. No per-voxel search, no copied rasterizer. Propagate metadata/target through queue/transfer/slicing/rematerialization. SmoothL1 width weighted by Gaussian falloff, denominator GT falloff mass.
4. Extra sideband sum alone gets the detached similarity multiplier to GT width, both train/test. No-width models retain legacy sidebands; unlabelled widths in width-enabled models give zero extra penalty. Ordinary positive/distribution/background loss unchanged.
5. Visualization: one base voxel equals one gray unit; modulate only direction-weighted views, not raw presence/GT/width. Center prediction sampled at actual CP coordinates; reuse same helper and existing oblique sampler on all five planes.
6. CPU regressions and approved manager GPU suite for gradient/count parity, gating separation and target propagation. Resume latest saved raw and verify startup/test/progress.

## Spec update
Document width tube targets, detached width-specific sideband gate, fixed target-only normalization and direct grayscale.

## Docs updates
Annotated heads: panel semantics, units, losses; task log/status record validation and run.

## Changelog update
Record width-conditioned sideband and Gaussian tube width supervision.

## Testing
Known endpoints/variable widths and overlap, output strides, missing labels, slice/to, exact width-loss weighting; gate0/10/20% deviations, no width gradients from gate, unchanged base presence, microbatch parity; annotated/legacy visualization. Use approved cooperative GPU test runner.
