# 3D Test Visibility And Direction Projection

Fix 3D TensorBoard visibility for test metrics/visualization and correct the
training/test sample direction overlay projection length.

Requirements:

- Keep 3D direction loss/error unchanged.
- In each principal slice, draw the projected direction line shorter when the
  3D direction points partly out of that slice.
- Keep the thin anti-aliased line style.
- When configured tests run, write a `test_sample_3d/principal_slices`
  visualization using the same principal-slice sheet as training.
- Flush TensorBoard after configured test logging so step-0 test scalars/images
  are visible promptly.
- Update docs/specs/task log and run focused tests.
