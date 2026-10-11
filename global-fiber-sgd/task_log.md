# Fused pair evaluation

- uv_pair_fusion_plan.md independently approved. User requests no intermediate
  array materialization where fusion can avoid it; keep4096 minibatch updates.
- Added explicit compiled backend using shared pair_objective, fullgraph,
  no silent fallback; eager remains default. No objective/precision changes.
- Added managed real-data pair benchmark and opt-in short production-update timing.
- 32focused tests passed. GPU benchmark launched via approved runner using
  run_uv_pair_fusion_benchmark.json, exclusiveGPU7 to avoid timing contamination.
- Completed all four managed benchmark configurations. Pair evaluation 2.379×,
  production updates 1.823× faster; pair incremental memory 6.61→1.75GiB.
  Exact commands, input, profiler summary and timings: uv_pair_fusion_results.md.
- Numerical caveat: four MAP choices changed after25 updates; maximum selected
  U-offset drift0.1871turn despite nearly identical aggregate metrics. Compiled
  stays opt-in, not an exact numerical replacement. Independent review approved
  that conclusion. No full solve launched or unchanged convergence claim.
- Plan deviation: optional sign-charge/expected-cost compiled GPU checks deferred;
  current likelihood, real states, uneven batches and boundary tests passed.
- Fixed partial-pass event-count and short-benchmark loss labels after runs;
  preserved original results and documented their metadata limitation.
- Latest focused suite rerun:32passed, existing sparse-invariant warning only.
