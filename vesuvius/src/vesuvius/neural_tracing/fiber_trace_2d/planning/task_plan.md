# Plan: Native 3D Trace2CP Pyramid Scaledown

## Implementation

1. Import Lasagna's shared `_pyrdown3d` helper from `lasagna.tiled_predict3d`
   with the same package/non-package fallback style used by the fiber 3D
   inference code.
2. Replace Trace2CP raw product scaledown with a wrapper that reshapes
   `B,C,D,H,W` tensors to `B*C,D,H,W`, calls `_pyrdown3d(..., factor=...)`,
   and reshapes back.
3. Keep validity-mask downsampling conservative by retaining the existing
   all-voxels-valid box reduction; this mask is support coverage, not a signal
   product.
4. Preserve the order: model inference, optional pyramid scaledown, optional
   inference-field Gaussian blur, trusted-core crop/cache.
5. Rename helpers/docs from box scaledown to pyramid scaledown where they
   describe product tensor signal scaling.

## Spec Update

Update native 3D Trace2CP scaled inference specs to say Gaussian pyramid
downscale via Lasagna `_pyrdown3d`, not box filtering, while retaining the
conservative validity-mask rule.

## Docs Updates

No standalone docs update is needed beyond specs.

## Tests

Update scaled inference regression tests so their expected values match
Lasagna pyrdown behavior and still verify point routing through the scaled
field. Keep the blur-after-scaledown test.

Run the relevant native 3D Trace2CP tests and `py_compile`.

## Changelog

Add one changelog line noting that native 3D Trace2CP scaled inference now
uses Lasagna Gaussian pyramid scaling.
