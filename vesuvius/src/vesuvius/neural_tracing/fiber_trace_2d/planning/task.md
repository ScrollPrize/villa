# Task: Report Direction Error In Degrees

Add a human-readable angular direction metric in degrees for V0 2D fiber-strip
training.

- Keep the existing two-channel encoded MSE as the optimization loss.
- Report folded unoriented angular error in degrees, where `0` is perfect and
  `90` is maximally wrong.
- Include the metric in console and TensorBoard output for train and test
  batches.
