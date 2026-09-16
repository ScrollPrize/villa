# Plan

Follow-up weight-only trial: stop prior run with approved launcher; change only
refinement tangent1->10 and direction0.1->1, retain normal0.1 and training LR/loss
weights. CPU regression, then approved production-start from original11800. No
new solver implementation or calibration protocol changes.

1. Extend shared bending stencil with sign-invariant normal projector: Wshape=normal_weight*nnT+tangent_weight*(I-nnT). Preserve isotropic fallback. Full 3x3 block-pentadiagonal solve including cross-coordinate bands.
2. Production direction_weight0.1 (default0). Sample encoded directions at segment quadrature; shared decoder, full base-to-patch Jacobian pullback including smooth warp. Freeze sampled directions within each iteration, analytic endpoint gradient of 1-(t.d)^2, resample next iteration. Detached raw predictions, no extra forward/backward.
3. Separate weighted normal/tangent bending and direction logs; material-length normalization, managed test totals and raw/EMA target identity unchanged.
4. Independent review; CPU dense-reference/axis/sign/gradient/transform tests and manager CUDA tests via approved cooperative launcher. Measure synthetic CPU cost.
5. Approved production launcher: original11800 checkpoint, LR0.01, fresh geometry.

## Spec update
Document anisotropic weights, lagged direction field/full warp pullback. No protocol/loading changes.

## Docs updates
Update refinement docs, specs, changelog, task log/status with objective/settings/validation/run.
