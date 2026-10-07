# Width variation on reference/unweighted trace paths

Artifacts, CSVs and 50 failure plots: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/normalwidth_122417_current_ema999_20261006_v1_width_paths`. Baseline: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/normalwidth_122417_current_ema999_20261006_v1_off_no_reverse`; frozen EMA999 u12100; 91 fibers/182 directions. No retracing, no inference. Native scalar sampling shared with embedding-query code; coherent channel coverage checked.

Width is predicted width in network-input voxels, not annotated true width. Reference sampled every8base voxels; trace samples are original committed points. Reference fiber summaries count only forward cases once; trace point summaries include both directions and weight points equally. Fibers/directions/events are not independent.

Prior means contain seed and all strictly preceding points, reset per restart. Missing earlier widths invalidate the mean. This is a committed-path diagnostic, not exact speculative beam-state reconstruction. Failure baseline is frozen strictly before the final departure from the acceptance ellipse (>1); locating that departure uses future outcome. Inside-ellipse is not proof of correct branch. Reference comparisons use stored matchedXYZ; offsets stay within the same observed segment. Reference continuation after failure is shown, but no nonexistent continued trace is invented.

## Reference variation (one value per fiber)

CV% distribution: `{'n': 91, 'mean': 5.698413815231608, 'p05': 2.7923885345346164, 'median': 4.5073871365283775, 'p95': 14.80757631235549, 'max': 19.985360861440086}`
Within-fiber P95 absolute deviation from median (%): `{'n': 91, 'mean': 11.492273113367604, 'p05': 5.368499304804191, 'median': 8.967164298397496, 'p95': 28.235134476184463, 'max': 46.107616502317704}`

## Causal prior-width deviations (point-weighted)

| Point group | Valid points | Median deviation % | P95 % | >10% | >20% | >40% |
|---|---:|---:|---:|---:|---:|---:|
| inside | 71466 | 3.36 | 22.46 | 14.33% | 5.90% | 1.03% |
| outside | 1331 | 13.23 | 39.03 | 57.70% | 35.84% | 3.91% |
| near_reference | 63878 | 3.09 | 15.38 | 10.69% | 3.39% | 0.43% |
| failure_free_segment | 58551 | 3.07 | 15.18 | 10.04% | 3.59% | 1.05% |

## Failure windows (event-weighted)

| Offset from final excursion (base vx) | Paired events | Trace median deviation % | GT median deviation % | GT preferred % | 10% wrong-only / GT-only rejected | 40% wrong-only / GT-only rejected |
|---|---:|---:|---:|---:|---:|---:|
| -256 | 42/42 | 9.84 | 6.90 | 61.904761904761905 | 12/4 | 0/0 |
| -128 | 44/44 | 14.40 | 7.81 | 68.18181818181817 | 12/4 | 1/0 |
| -64 | 49/49 | 15.89 | 10.77 | 63.26530612244898 | 9/4 | 2/0 |
| 0 | 50/50 | 14.07 | 10.74 | 48.0 | 13/9 | 2/1 |
| 64 | 49/49 | 18.52 | 9.29 | 67.3469387755102 | 16/2 | 4/0 |
| 128 | 49/49 | 16.40 | 11.91 | 57.14285714285714 | 15/5 | 3/2 |
| 256 | 0/0 | 0.00 | 0.00 | None | 0/0 | 0/0 |
| failure | 50/50 | 20.86 | 10.18 | 64.0 | 14/4 | 2/0 |

Coverage: `{"PHerc0139": {"requested": 44900, "valid": 44900}, "PHerc0332": {"requested": 1903, "valid": 1903}, "PHerc1299": {"requested": 23014, "valid": 23014}, "PHerc1451": {"requested": 1645, "valid": 1645}, "PHercParis4": {"requested": 145836, "valid": 145836}}`. Missing samples never become zeros. 50 events from 31 directed cases / 23 distinct fibers.

Reference deviations stratified by presence (0.5 diagnostic threshold, not a new tracer setting): `{"presence_ge_05": {"n": 30742, "mean": 3.9111747460297517, "p05": 0.2402405531187524, "median": 2.777750252679584, "p95": 10.645155058269417, "max": 49.321978817899705}, "presence_lt_05": {"n": 5405, "mean": 10.539096696180279, "p05": 0.7463430538413496, "median": 8.606816959244924, "p95": 27.338177345154197, "max": 53.94725531410016}}`.

Seed-only deviation: `{"n": 36147, "mean": 7.260547702455439, "p05": 0.39264841212599544, "median": 4.43507143625379, "p95": 25.094912557525877, "max": 77.5367485665086}` on GT; `{"n": 72797, "mean": 9.917035980156129, "p05": 0.4266460908423797, "median": 4.877722903905446, "p95": 44.615964786227735, "max": 103.86926679413455}` on traces. Seed itself excluded from all point-deviation summaries.

## Interpretation and next candidates

- These are retrospective samples, not predicted restart improvements. See paired denominators and threshold counts above; missing coverage and short histories must not be read as successful discrimination.
- A physically constant fiber width need not give a constant predicted width. Soft penalties or confidence-qualified global-width updates are hypotheses requiring controlled reruns.
- Post-excursion offsets are omitted when the trace has already ended. No continuation was fabricated.

## Reproduce

Use `reviewed_width_paths` for historical u11000; `reviewed_ema999_width` for the pinned current EMA999 benchmark and diagnostic.

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```
