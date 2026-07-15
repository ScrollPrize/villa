# 3D Fiber CP Model Variant

Plan a 3D variant of the current CP-centered 2D fiber model approach.

Requirements:

- Load 3D patches around fiber control points.
- Do not construct fiber-aligned 3D strips or slices for loading; the 3D path
  loads an ordinary CP-centered 3D volume block and applies augmentation to that
  block.
- Apply deterministic coordinate-space geometric augmentation and GPU value
  augmentation around those CPs.
- Use an existing Vesuvius 3D U-Net/fiber model configuration where possible,
  or wrap the regular Vesuvius 3D U-Net backbone with fiber-specific heads.
- Train fiber direction and fiber presence, analogous to the current 2D side
  model.
- Encode 3D direction as Lasagna's 3x2 ambiguous direction channels.
- For the initial evaluation path, run 3D inference, project the predicted 3D
  direction field onto a 2D test fiber strip, and reuse the 2D strip tracer
  there.
- Read the 3D augmentation special-case TODOs from `planning/todo.md` and
  include an explicit augmentation-by-augmentation adaptation plan.
- Keep the current 2D fiber training, runner, prefetch, and visualization code
  supported.
- Share the fast Zarr/chunk loading and prefetch code between 2D and 3D where
  practical, while keeping the 3D model/loader/training path separate.
- Target significantly larger effective training batches, around 192 CP
  patches per optimizer step, with micro-batching if dense 3D U-Net memory
  requires it.
