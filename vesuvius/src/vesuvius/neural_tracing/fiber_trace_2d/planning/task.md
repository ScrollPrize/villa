# Training Throughput Parallelization

Accelerate `fiber_trace_2d` training throughput for the current `loader_example.json`
workload. The observed throughput is nearly unchanged, CPU utilization is low
relative to available cores, and GPU utilization is low. The target is to make
actual measured patches/s substantially faster by parallelizing the loading and
preparation path, not just changing reported timings.

Requirements:

- Reuse the already approved benchmark command shape and rewrite only
  `/tmp/fiber_trace_p_d2_w1.json` between variants.
- Measure every performance change and record attempts in `planning/task_log.md`.
- Prefer real parallel data loading and preparation over cosmetic timing changes.
- Preserve deterministic sample ordering and training semantics.
- Keep docs/status/task log current.
