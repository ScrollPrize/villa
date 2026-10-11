# Global distance-objective comparison

- Plan uv_distance_objectives_plan.md independently approved.
- Four fused runs: likelihood, squared, Huber, hybrid; GPUs0–3, shared24GiB.
- Fresh radial CCW151 initialization,4096random updates with replacement, seed0,
  16130updates (~10event-pass equivalents). All variables train immediately.
- Reference diagnostics only; no reference initialization or warm start.
- Shared pair geometry, signed expected distance, sign/metric each counted once.
- 49CPU tests passed (existing sparse-invariant warning only). Added nonfinite
  sigma rejection and explicit wrong-sign-distance regression tests after review.
- Managed CUDAvalidation completed: eager/fused loss and all six input gradients
  agree for all three modes at real4096x151. Results saved under
  data/uv_distance_objectives_validation_v1/summary.json.
- All four production jobs submitted through the approved runner with configs
  run_uv_distance_{likelihood,squared,huber,huber_likelihood}.json. Manager
  requests waiting briefly for current training memory release, not bypassed.
- Sampled history now reports interval-average objective, not last minibatch.
- All four grants obtained, initial metrics identical (L1 12.242345,
  within2 12.991196%, retained100%, fittedq10/90 49/122 vs reference42/124).
- First1613updates: L1 baseline12.0835/squared11.9615/Huber12.0546/hybrid12.0544;
  retained86.412/84.038/85.977/82.413%. Continuing full budget, not early stopping.
- Squared/Huber/hybrid completed16130updates and full audits (all exits0).
  Final L1 11.7236/11.8602/11.8717; retained66.023/70.253/64.698%.
  Ranges49/122 unchanged against reference42/124. No large global recovery.
  Baseline finishing; histories persist every1613updates and full report follows.
- Baseline also complete, exit0: L1 11.91029, within2 13.1291%, retained68.3658%.
  All final audits full6.606M. Final reference q90 is125 on squared/hybrid
  retained subsets; full initial reference q90 is124. Documented population effect.
- Full configs, commands, objective definitions, interval histories and final
  tables persisted in uv_distance_objectives_results.md/progress.md.
- Verdict: no large global improvement. Huber retains1.8872pp more length with
  .0501lower L1, but higher sign error. No default promoted or further runs added.
