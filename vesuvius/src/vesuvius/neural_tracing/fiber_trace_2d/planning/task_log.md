# Task Log: Coordinate Tensor Boundary Cleanup

## Planning Notes

- Current torch strip-grid construction converts dense coordinates and valid
  masks back to NumPy immediately in `strip_geometry.py`.
- Loader geometric augmentation then converts those arrays back to torch and
  returns NumPy again before VC3D coordinate sampling.
- The planned cleanup keeps coordinate tensors in torch through source-grid,
  strip-z offset, geometric augmentation, and line/CP coordinate generation,
  then converts once at explicit NumPy consumers.

## Validation

- Not run yet; this turn produced the plan for review.
