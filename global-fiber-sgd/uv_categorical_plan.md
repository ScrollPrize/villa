# Aggregated cost-target categorical updates

## Task and plan
Opt-in assignment solver, preserving old solvers.
1. Extract/reuse shared pair-cost geometry. On GPU evaluate all KxK fixed-sign,
   magnitude, soft-distance/metric costs in batches. Accumulate conditional costs
   for BOTH endpoints against neighbor unnormalized winding probabilities; broken
   neighbor contributes zero. Scale incident sums N/E against node rejection R*c_i.
2. Include continuation pair costs, event phase priors and node geometric priors.
   Broken is explicit101st candidate. No selected-pair blanket charge.
3. Target softmax(-cost/T), detached. Synchronous logit SGD with p-target gradient,
   step4 and8inner updates, recomputing targets every sweep. Pilot T=.25/1,
   30sweeps. Full incident graph, not noisy per-edge pseudo-targets.
4. First comparison freezes saved continuous geometry and widths, explicitly
   reported. Same accepted prior labels, cached evidence and warm provenance.
   Pilot original anchor warm start then both full H directions if viable.
5. Test tiny-probability revival, aggregate/scaling correctness, unknown signs,
   broken gating, batching, saved audit. Report sign errors and retained length.
   No guarantee of global convergence; synchronous updates can oscillate.

## Spec update
Opt-in assignment-only targets. Expected soft pair energy replaces mixture
likelihood for this mode. All100hypotheses and broken retained. No posthoc
filtering, altered evidence or tolerance relaxation. Fixed geometry means
geometry-only regularizers are constant; label-dependent node priors retained.
Temperature is target entropy scale, not calibrated uncertainty.

## Docs updates
Task/log/status, configs/results, uv_layout.md and changelog. Independent review,
test commands, prior fixed-charge full results preserved. No new dependencies.
