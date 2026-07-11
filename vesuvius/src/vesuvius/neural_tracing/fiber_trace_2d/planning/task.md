# Load-Only Parallelism Diagnostics

Add load-only benchmark profiling that shows real process CPU consumption per
batch in addition to the existing summed loader-worker timings.

Requirements:

- Keep the focus on `--load-only --profile` so loader parallelism can be
  inspected without model or image-augmentation work.
- Report per-batch wall time and process CPU time so the profiler can show
  whether observed worker-time parallelism corresponds to actual CPU usage.
- Preserve the existing loader profiling columns and deterministic loading
  behavior.
- Reuse the existing benchmark command family and keep task notes current.
