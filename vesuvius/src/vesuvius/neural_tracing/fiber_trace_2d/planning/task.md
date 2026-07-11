# Full Test Trace2CP Evaluation Sentinel

User request:

- Make `training.test_control_points: 0` mean that test evaluation runs over
  every configured held-out control point sample.
- This should make the training test Trace2CP metric comparable to running
  whole-fiber Trace2CP visualization on the same held-out fiber, apart from
  small differences such as whole-fiber row-axis alignment.
- Keep positive `test_control_points` values as the existing fixed-size
  deterministic test subset.
