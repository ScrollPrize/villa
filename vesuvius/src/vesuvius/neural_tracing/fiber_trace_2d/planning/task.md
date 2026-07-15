# 3D Test Visualization And Raw Index Cleanup

Fix the 3D training/test visualization and sample-index handling after the
previous parity update.

Requirements:

- Test visualization must show the actual carried fiber-line context in the
  target/context presence panel, not only the CP-only loss target.
- This visualization fix must not change loss semantics: JSON/non-NML fibers
  still supervise only the CP neighborhood, while NML fibers still supervise
  dense centerline targets.
- Remove the alternate augmentation-index public loader path. The 3D loader
  should consistently interpret `sample_index` as the raw/global deterministic
  training stream index, derive the bounded data sample from
  `sample_index_limit`, and seed augmentation from the raw index.
- Add a config option to show several batch samples in train and test
  TensorBoard visualization. Default to four samples and support separate test
  override.
- Dense 3D tests should evaluate all held-out CPs by default. Positive
  `training.test_control_points` is only a debugging cap.
- Update specs/docs/tests/task log for the above.
