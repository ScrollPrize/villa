# Legacy constraints ported to fine pieces

Frozen raw40500 iso20 Paris4 1024 crop,128CPU workers,RelWithDebInfo,500 iterations. Old512/128 measurements preserved once each; only their owner pieces and continuity change. Cuts between old orthogonal observation arcs, with minimum64/128/256 base voxels. No new certification, local geometry scorer, normalization or deduplication. Old reference readout throughout. Measurements retain their old spatial support, not remeasured on small owners. No inference/retracing.

Accuracy denominator26; shift uses24 observed references unless indicated. Higher is better. Replay100% inherited; old oracle80.77%,constraint score−25.24% historical only. No new oracle per variant.

| Variant | Accuracy | Shift (N) | Disabled length | Pieces | Missing | Ties/unbounded | Seconds |
|---|---:|---|---:|---:|---|---|---:|
| baseline | 50.00% | 83.33% (24) | 10.49% | 2732 | ['4', '12'] | 0/0 | 49.6 |
| m128-incident | 46.15% | 70.83% (24) | 7.40% | 4912 | ['4', '12'] | 0/0 | 47.2 |
| m128-length | 50.00% | 79.17% (24) | 6.03% | 4912 | ['4', '12'] | 0/0 | 43.8 |
| m128-length-boundary | 50.00% | 83.33% (24) | 8.02% | 4912 | ['4', '12'] | 0/0 | 46.5 |
| m256-d300-b0 | 50.00% | 83.33% (24) | 7.21% | 2799 | ['4', '12'] | 0/0 | 38.9 |
| m256-d300-b500 | 50.00% | 79.17% (24) | 7.78% | 2799 | ['4', '12'] | 0/0 | 38.3 |
| m256-d75-b0 | 50.00% | 87.50% (24) | 6.07% | 2799 | ['4', '12'] | 0/0 | 40.0 |
| m256-d75-b500 | 50.00% | 83.33% (24) | 9.01% | 2799 | ['4', '12'] | 0/0 | 38.2 |
| m256-incident | 50.00% | 83.33% (24) | 9.89% | 2799 | ['4', '12'] | 0/0 | 41.2 |
| m256-incident-boundary | 50.00% | 83.33% (24) | 8.82% | 2799 | ['4', '12'] | 0/0 | 36.4 |
| m256-length | 50.00% | 87.50% (24) | 6.04% | 2799 | ['4', '12'] | 0/0 | 39.1 |
| m256-length-binary | 50.00% | 83.33% (24) | 7.23% | 2799 | ['4', '12'] | 0/0 | 39.2 |
| m256-length-boundary | 50.00% | 83.33% (24) | 7.54% | 2799 | ['4', '12'] | 0/0 | 39.6 |
| m64-incident | 38.46% | 62.50% (24) | 5.53% | 9265 | ['4', '12'] | 0/0 | 52.7 |
| m64-length | 42.31% | 66.67% (24) | 3.94% | 9265 | ['4', '12'] | 0/0 | 55.0 |
| m64-length-boundary | 46.15% | 75.00% (24) | 7.83% | 9265 | ['4', '12'] | 0/0 | 60.9 |

Incident: original weighted-incidence disabling cost. Length:150 per128 base voxels. Length-boundary additionally charges500 at active/disabled transitions. binary suffix hardens old orientation scores; d75/d300 change length cost only. N/A means multiple active independent gauges, summaries in JSON. Tied assignments use deterministic rank1, not uniquely determined labels. Shift excludes missing references and bridges their gaps.

Historical fine64 with new measurements/tuned costs and old readout:38.46% accuracy/45.83% shift. Old overlap with binary scores or length+boundary:53.85%/79.17%. These are tuning-crop comparisons, not held-out validation.

Artifacts: `/ephemeral/hendrik/fiber_crop_bench/20261002_iso20_legacy_fine_v1`. Exact commands/binary hashes are in run.json files. Approved runner: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, action `iso20_legacy_fine`. `followup:true` runs at most six predefined local followups; `report_only:true` only refreshes this table.

## Outcome

- Completed baseline +9 ports +6 bounded followups. All242762 ordinary observations and total evidence weight preserved; old baseline initial states reproduced exactly.
- Best shift: minimum256, soft legacy measurements, length cost75 or150 per128 voxels, boundary0:50.00% accuracy/87.50% observed shift. No tied/unbounded reference rows. Same13/26 exact references as baseline; offset-change count falls4 to3 among24 observed references. This is a modest metric improvement, not a solution to the remaining winding errors.
- Length150 disables6.04% of physical length versus old baseline10.49%; length75 gives6.07% and the same reference ladder. Minimum256 yields2799 nonoverlapping pieces vs2732 overlapping baseline pieces; the winning control is not the very fine64 layout.
- Fine64 with length+boundary improves to46.15%/75.00%, versus38.46%/45.83% using new measurements with that readout. Minimum128+length+boundary matches50.00%/83.33%.
- Binary scores, doubled length cost or boundary priors do not improve the best port. Prior tuned old-overlap accuracy53.85%/shift79.17% remains an alternative tradeoff, not dominated on both metrics.
-446 native tests +46 Python checks pass. No default promotion, no extra BP iteration budgets, no inference/retracing. One crop only; oracle/removal scores not recomputed. Recorded times are single-run observations, not repeated performance claims.
