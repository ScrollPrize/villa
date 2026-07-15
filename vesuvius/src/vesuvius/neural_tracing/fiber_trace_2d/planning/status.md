# 3D Trace2CP Metric Wiring Status

- [x] Read current task request.
- [x] Confirm current implementation gap in `fiber_trace_3d/train.py`.
- [x] Replace `planning/task.md` with the current missing-wiring task.
- [x] Write focused `planning/task_plan.md`.
- [x] Add explicit no-silent-simplification/no-silent-postponement rule to the
  local workflow.
- [x] Add explicit requirement to remove 3D Lasagna grid-search decoding and
  use analytic decoding/reconstruction.
- [x] Report other deferred/ignored items from the previous follow-up.
- [ ] Implement 3D Trace2CP config parsing and evaluator.
- [ ] Remove 3D grid-search direction decode from Trace2CP projection.
- [ ] Add torch-vectorized analytic Lasagna 3x2 direction decode tests.
- [ ] Wire 3D training test metric, TensorBoard, stdout, and best checkpoint
  selection.
- [ ] Add 3D Trace2CP CLI visualization.
- [ ] Update specs/docs/changelog after implementation.
- [ ] Run focused and regression tests.
