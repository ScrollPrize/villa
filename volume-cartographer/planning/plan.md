# Development Plan

- Maintain durable subproject requirements in `specs/`.
- Use synthetic, reproducible workloads to establish rendering correctness and
  performance baselines before changing rendering scheduling or algorithms.
- Prefer deterministic instruction/cache-event regression gates and use native
  timings as calibration measurements.
- Evaluate future rendering plans and implementations against
  `specs/rendering.md` and `specs/benchmarks.md`.
