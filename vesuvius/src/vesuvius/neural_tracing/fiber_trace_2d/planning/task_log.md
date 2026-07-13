# Trace2CP Regular Trace With Side-View Z Experiment Task Log

- Started a new current-task log for the regular Trace2CP plus side-view z/top
  experiment.
- Interpreted the user request as a design checkpoint because it ended with
  "makes sense?" rather than an explicit implementation go-ahead.
- Wrote `planning/task.md` and `planning/task_plan.md` to keep the default
  Trace2CP behavior on the regular tracer and add a separate opt-in
  experimental tracer.
- Added the top-patch framing constraint: each top patch is sliced using the
  sampled side-view direction, and the experiment only corrects angle relative
  to the side-view normal rather than optimizing roll around the fiber line.
- No implementation or validation has been run yet for this task.
