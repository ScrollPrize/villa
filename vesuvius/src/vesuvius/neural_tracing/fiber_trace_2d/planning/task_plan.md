# Load-Only Parallelism Diagnostics Plan

## Implementation

- Add per-batch `time.process_time()` measurement around the benchmark batch
  body.
- Print process CPU milliseconds per patch and a process CPU factor
  (`cpu_time / batch_wall_time`) in profile rows.
- Keep the existing loader `work / wall` factor so synthetic worker timing and
  actual process CPU usage can be compared directly.
- Include process CPU time and factor in the profile summary.

## Spec Update

- Document that benchmark/profile output includes both summed loader worker
  timing and real process CPU timing, and that only process CPU factor should be
  used to compare against system CPU utilization.

## Docs Updates

- Update `docs/code_structure.md` profiling description.
- Keep `planning/task_log.md` limited to this diagnostic task.

## Testing

- Compile-check `train.py`.
- Run the load-only profile benchmark with the existing command family.
