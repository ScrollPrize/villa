# Task Log: Real Fused Geometric Augmentation Map Tensors

## Planning Notes

- The previous implementation centralized formulas in `StripAugmentTransform`
  but still evaluated transform math during line/CP mapping.
- The corrected requirement is to construct concrete fused map tensors:
  `backward_map_xy` and `forward_map_xy`.
- Smooth and affine stages must be baked into those maps during construction.
- Line/CP mapping must be lookup/interpolation against `forward_map_xy` only.
- `planning/specs.md` now defines "fused map" as actual precomputed coordinate
  map tensors, not a shared formula bundle.

## Implementation Notes

- `StripAugmentTransform` now constructs `backward_map_xy` and
  `forward_map_xy` tensors once in `__post_init__`.
- `output_to_source_grid()` returns the cached backward map tensor used by
  coordinate augmentation.
- `source_to_output_points(...)` and `output_to_source_points(...)` now sample
  cached maps with `torch.nn.functional.grid_sample`.
- Smooth offset interpolation and affine formulas are still used while building
  the maps, but not during runtime line/CP point mapping.
- Smooth nonlinear round-trip tests now assert subpixel cached-map consistency
  rather than exact formula equality. A separate regression test forbids smooth
  interpolation after transform construction.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/augmentation.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `83 passed in 3.94s`.
