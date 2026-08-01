# VC3D GitHub CI Acceleration

- Reduce the wall time of the Volume Cartographer GitHub pull-request CI,
  ideally to less than one minute.
- Treat this as an interactive investigation: measure each proposed build and
  workflow change locally in the exact GitHub CI container before adopting it.
- Preserve compile coverage for every configured target, even when a target is
  not executed by the test job. Compile coverage may live in separate jobs.
- Preserve numerical behavior and portability requirements.
- Record commands, inputs, cache state, timings, successes, failures, and
  workflow findings in `planning/task_log.md` as the investigation proceeds.
