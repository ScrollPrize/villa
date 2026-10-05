# Conservative two-sided orthogonal evidence

Frozen raw40500 iso20 Paris4 1024 crop; legacy-to-fine256, length150/128, boundary0. 128CPU workers, RelWithDebInfo,500 iterations, old reference readout. No inference/retracing. Four exact ±span probes: move A against centerB and move B against centerA. Require all four finite probes in the center normal component, otherwise keep original. Median includes center; consensus additionally requires all five signed half-winding quantizations agree. Adapt after legacy acceptance/cutoff. No rejection or parallel-estimator changes; same cuts, scores, weights and owners.

| Variant | Exact accuracy ↑ | Shift ↑ (N) | Disabled length | Changed targets | Changed signs | Seconds |
|---|---:|---|---:|---:|---:|---:|
| baseline | 50.00% | 87.50% (24) | 6.04% | 0 | 0 | 40.7 |
| consensus-16 | 50.00% | 87.50% (24) | 6.03% | 10806 | 1 | 40.6 |
| consensus-32 | 50.00% | 87.50% (24) | 6.00% | 8186 | 2 | 39.1 |
| consensus-64 | 50.00% | 87.50% (24) | 6.04% | 3271 | 2 | 38.3 |
| half-median-32 | 50.00% | 87.50% (24) | 6.02% | 19259 | 3 | 42.4 |
| half-median-64 | 50.00% | 87.50% (24) | 6.02% | 20724 | 7 | 46.3 |
| median-16 | 50.00% | 87.50% (24) | 6.03% | 14999 | 2 | 40.4 |
| median-32 | 50.00% | 87.50% (24) | 5.99% | 19259 | 6 | 39.0 |
| median-64 | 50.00% | 87.50% (24) | 6.02% | 20724 | 7 | 39.5 |
| normal-cosine | 50.00% | 83.33% (24) | 6.06% | 0 | 0 | 40.5 |
| parallel-0p0 | 3.85% | 20.83% (24) | 30.19% | 0 | 0 | 34.5 |
| parallel-0p05 | 61.54% | 87.50% (24) | 6.09% | 0 | 0 | 40.0 |
| parallel-0p2 | 50.00% | 83.33% (24) | 7.14% | 0 | 0 | 39.0 |
| winner-consensus32 | 61.54% | 87.50% (24) | 6.05% | 8186 | 2 | 39.4 |
| winner-median64 | 61.54% | 87.50% (24) | 6.07% | 20724 | 7 | 37.9 |
| winner-repeat | 61.54% | 87.50% (24) | 6.09% | 0 | 0 | 40.8 |

Accuracy denominator26; observed shift skips missing references. Full ladders/ties/missing in JSON. Replay100% inherited; old overlap50%/83.33%, oracle80.77%,constraint−25.24% historical only. No new oracle/removal runs. One tuning crop, no default change.

Artifacts: `/ephemeral/hendrik/fiber_crop_bench/20261002_iso20_orthogonal_probes_v1`. Approved runner action `iso20_orthogonal_probes`; exact commands/hashes in run.json. `report_only:true` refreshes saved results. `followup:true` tests finite parallel magnitude weights0/0.05/0.2 (baseline0.1) and normal-alignment cosine confidence, on the unmodified port without probes. Hard ordering signs stay enabled. These objective settings also apply to independent reference ranking; they are end-to-end objective trials, not readout-isolated graph experiments.

## Findings

- All eight probe-only variants retain50.00% exact accuracy/87.50% observed shift. They change3271–20724 scalar targets but only1–7 signs; most solves have different piece states, so this is not an inactive flag. They add no benchmark benefit, even when combined with the better objective.
- Soft parallel one-winding distance weight0.05 (was0.1) gives61.54% accuracy/87.50% shift:16/26 exact references, three more than baseline. Exact repeat reproduces all solved piece states and reference ladder. Missing references4/12 unchanged, no ties or unbounded rows.
- Weight0 gives3.85%/20.83%; weight0.2 and cosine alignment confidence give50.00%/83.33%. Retain a weak magnitude term; hard signs alone are insufficient here.
- Combining winner with median64 or consensus32 probes keeps61.54%/87.50%; prefer simpler no-probe winner. Disabled length6.09%, vs6.04% prior port and10.49% old overlap.
- Current leading tested independent-score candidate: legacy-fine minimum256, length150/128, boundary0, soft original measurements, `--winding-weights 0,0,0,0.05,0`, hard signs both, old reference readout,500 iterations. The weight also changes reference ranking costs, so this gain is not proven solely attributable to the initial graph solve.
-16 valid reported runs including repeat.447 native cases +48 Python checks passed. Opt-in only, no production defaults changed. One tuning crop and no new oracle/removal scores; not held-out validation.

Reproduce with approved runner action iso20_orthogonal_probes: initial matrix, followup:true, confirmation:true, or report_only:true. Build/tests via the same runner action build. Per-run command and binary/input hashes are retained.
