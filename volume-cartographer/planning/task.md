# Task: remove cross-run worker identity from render Valgrind replay

Fix the synthetic rendering CI evaluator so independently collected Callgrind
and DRD artifacts do not assume that raw Valgrind worker thread IDs identify
the same worker or workload in both executions.

- Keep the initial process thread as the explicit main/control role.
- Aggregate all modeled non-main Callgrind costs into one renderer worker-pool
  cost.
- Let the DRD execution define the worker threads and parallel schedule.
- Distribute worker-pool cost over non-main DRD compute windows using the
  existing work-quantum weighting.
- Preserve the exact complete modeled Callgrind cost.
- Treat idle or differently participating workers as valid rather than as a
  missing-cost error.
- Add regressions for worker-ID relabeling, differing worker participation,
  and cost conservation.
