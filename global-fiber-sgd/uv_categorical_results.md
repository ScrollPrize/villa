# Cost-target winding/broken assignment experiments

## Pilot result

10,005 pieces from10,000 source fibers;105,178 constraints. Earlier accepted
normal/HV/direction stages remain fixed. All signs audited strictly, broken if
its fitted probability exceeds50%. No posthoc filtering. These are internal
model consistency measurements, not anatomical ground-truth accuracy.

| Trial | Sign violations | Retained length | Rejected pieces | Solve time |
|---|---:|---:|---:|---:|
| Earlier anchor state |6,525|95.6542%|1.5692%|—|
| Previous fixed-charge4 result |227|69.4125%|18.8906%|29.67s|
| Aggressive categorical,T=.25,from anchor |3,288|93.6321%|2.4588%|19.04s|
| Aggressive categorical,T=1,from anchor |2,867|87.5463%|6.4168%|19.12s|
| Damped categorical,from anchor |0|52.7570%|32.0040%|120.49s|
| **Damped repair,from fixed-charge4** |**0**|**71.9722%**|**16.6517%**|120.13s|

Aggressive=30full sweeps,8CEupdates per sweep,step4. It oscillates between broad
rejection and recovery; final even sweep is not representative convergence.
Damped=100sweeps,1update per sweep,step1,T=.25. Batch1024 is only a memory
partition; every event participates in each target computation. Shared H100,
fixed seeds/deterministic Torch. Times are observed workload times, not isolated
performance measurements. No native build needed.

Repair:227initial violations comprise137corrected while both endpoints survive
and90removed by endpoint rejection. Final0new violations. Unique rejected pieces
fall1,890→1,666; retained constraints46,697→49,947. Last ten sweeps all have zero
sign violations. Independent saved-array audit verified identities, boundaries,
continuous geometry and widths bit-identical to warm start.

## Important limitations

- Only probabilities are optimized; continuous shared phases/offsets/widths frozen.
- Expected soft pair energy replaces previous mixture likelihood, and target
  temperature replaces entropy sharpening. Not an optimizer-only ablation.
- Initialization matters substantially; the diffuse anchor start loses more data.
- Neither pilot preserves any of the5continuation links. This remains a serious
  layout limitation, not fixed by sign repair.
- Final repair maximum discrepancy between probability and its last target is
  .936. Zero selected-sign errors does not mean posterior convergence.
- Probabilities are fitted model choices, not calibrated error probabilities.
- First pilots record dispatcher hash but not new module hash. Subsequent full
  runs capture both execution module hashes; this provenance omission is noted
  rather than backfilling a guessed hash for historical pilot executions.

## Reproduction and artifacts

See [model and commands](uv_layout.md), [machine-audited pilot report](uv_categorical_pilot_results.md),
and corresponding JSON. Configs `run_uv_categorical_repair.json`, `_damped.json`,
`_pilot_t0.25.json`, `_pilot_t1.json` use the existing approved runner. Outputs:
`data/uv_categorical_v1/pilot_repair`, `pilot_damped`, `pilot_t0.25`, `pilot_t1`.
Completed roots are protected; choose a fresh output_root to rerun.

Validation:65tests passed8.39s, covering exact conditional-energy derivatives,
both endpoint aggregation/scaling, tiny-probability recovery, broken gating,
strict sign handling, warm provenance and legacy solvers.

## Full comparisons

Both full runs completed10sweeps on489,663pieces and6,606,100constraints,
starting from prior full fixed-charge16 states. These are final solver audits;
the separate persisted-artifact report has not yet been run for the full results.

| Direction | Violations before → after | Final violation rate | Retained length before → after | Rejected pieces before → after | Seconds |
|---|---:|---:|---:|---:|---:|
| CCW |28,760→110|0.0048921%|64.4886%→62.4748%|21.1274%→22.7383%|755.14|
| CW |27,968→104|0.0046532%|64.2597%→62.3828%|21.2675%→22.7887%|754.85|

Both used the original same-GPU shared24GiB grants; peak Torch allocated memory
was6.1316GiB per process. Future configs now use separate GPUs and12GiB each.
Retained continuation links:CCW1/466,CW0/466. Sign errors fell by more than99.6%,
but zero was not reached and continuation remains unresolved. No filtering or
additional iterations were applied after these10sweeps.
