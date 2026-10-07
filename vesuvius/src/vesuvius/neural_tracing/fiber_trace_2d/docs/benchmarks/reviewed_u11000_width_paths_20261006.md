# Width variation on reference/unweighted trace paths

Artifacts, CSVs and 73 failure plots: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/normalwidth_u11000_width_paths_20261006_v1`. Baseline: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/normalwidth_u11000_off_no_reverse_20261006_v1`; frozen raw u11000; 91 fibers/182 directions. No retracing, no inference. Native scalar sampling shared with embedding-query code; coherent channel coverage checked.

Width is predicted width in network-input voxels, not annotated true width. Reference sampled every8base voxels; trace samples are original committed points. Reference fiber summaries count only forward cases once; trace point summaries include both directions and weight points equally. Fibers/directions/events are not independent.

Prior means contain seed and all strictly preceding points, reset per restart. Missing earlier widths invalidate the mean. This is a committed-path diagnostic, not exact speculative beam-state reconstruction. Failure baseline is frozen strictly before the final departure from the acceptance ellipse (>1); locating that departure uses future outcome. Inside-ellipse is not proof of correct branch. Reference comparisons use stored matchedXYZ; offsets stay within the same observed segment. Reference continuation after failure is shown, but no nonexistent continued trace is invented.

## Reference variation (one value per fiber)

CV% distribution: `{'n': 91, 'mean': 6.5330729598565656, 'p05': 3.1409110533659894, 'median': 5.456857505380403, 'p95': 15.046147994682986, 'max': 18.32309287847819}`
Within-fiber P95 absolute deviation from median (%): `{'n': 91, 'mean': 13.708932184536344, 'p05': 5.371605616284212, 'median': 10.902782500893064, 'p95': 31.208242614189196, 'max': 53.42534429161722}`

## Causal prior-width deviations (point-weighted)

| Point group | Valid points | Median deviation % | P95 % | >10% | >20% | >40% |
|---|---:|---:|---:|---:|---:|---:|
| inside | 71051 | 3.93 | 29.08 | 19.74% | 9.07% | 1.55% |
| outside | 1865 | 15.42 | 43.45 | 64.83% | 40.80% | 6.70% |
| near_reference | 62818 | 3.54 | 21.48 | 14.73% | 5.54% | 0.82% |
| failure_free_segment | 54959 | 3.61 | 25.52 | 15.85% | 6.92% | 1.02% |

## Failure windows (event-weighted)

| Offset from final excursion (base vx) | Paired events | Trace median deviation % | GT median deviation % | GT preferred % | 10% wrong-only / GT-only rejected | 40% wrong-only / GT-only rejected |
|---|---:|---:|---:|---:|---:|---:|
| -256 | 67/67 | 11.16 | 8.44 | 40.298507462686565 | 11/3 | 1/3 |
| -128 | 71/71 | 11.57 | 8.52 | 57.74647887323944 | 18/8 | 4/1 |
| -64 | 72/72 | 14.85 | 9.78 | 63.888888888888886 | 24/11 | 2/2 |
| 0 | 73/73 | 19.82 | 9.94 | 65.75342465753424 | 27/9 | 5/5 |
| 64 | 72/72 | 19.42 | 9.14 | 68.05555555555556 | 28/8 | 6/4 |
| 128 | 69/69 | 15.24 | 10.64 | 56.52173913043478 | 21/7 | 6/4 |
| 256 | 0/0 | 0.00 | 0.00 | None | 0/0 | 0/0 |
| failure | 73/73 | 18.43 | 10.52 | 61.64383561643836 | 24/9 | 9/5 |

Coverage: `{"PHerc0139": {"requested": 44907, "valid": 44907}, "PHerc0332": {"requested": 1852, "valid": 1852}, "PHerc1299": {"requested": 23001, "valid": 23001}, "PHerc1451": {"requested": 1652, "valid": 1652}, "PHercParis4": {"requested": 146144, "valid": 146144}}`. Missing samples never become zeros. 73 events from 45 directed cases / 31 distinct fibers.

Reference deviations stratified by presence (0.5 diagnostic threshold, not a new tracer setting): `{"presence_ge_05": {"n": 29292, "mean": 4.208528718053881, "p05": 0.27135013977872774, "median": 3.0203199648378254, "p95": 11.094871740608292, "max": 65.15808766969901}, "presence_lt_05": {"n": 6855, "mean": 10.207734492648546, "p05": 0.6824690820413154, "median": 7.79378642796682, "p95": 27.70607060586438, "max": 75.3934580889785}}`.

Seed-only deviation: `{"n": 36147, "mean": 7.497792113673599, "p05": 0.4514758801563058, "median": 5.260128147868093, "p95": 22.612659353138653, "max": 89.44090489527792}` on GT; `{"n": 72916, "mean": 10.767564181232938, "p05": 0.4678058911296634, "median": 5.500179610762473, "p95": 44.45697240502945, "max": 141.01740422451363}` on traces. Seed itself excluded from all point-deviation summaries.

## Interpretation and next candidates

- Width has useful but imperfect branch-discrimination information: at final excursion onset it favors GT in 48/73 events. A 20% cutoff rejects trace only in 28 events, but GT only in 9; a 40% cutoff gives 5 versus 5. These are retrospective samples, not predicted restart improvements.
- A tight hard-zero gate is risky: even failure-free trace segments have 15.85% of points beyond 10% of their running mean. Try a tunable soft penalty with a nonzero floor rather than annihilating presence.
- Qualify width/history updates by presence confidence, especially across gaps, and test short spatial aggregation to suppress isolated width fluctuations. Keep a global width estimate rather than introducing unchecked drift.
- Seed-only anchoring is not supported: GT P95 deviation is 22.61% from the seed versus 17.41% from its running mean.
- Example plots case 0/failure 0 and reverse case 3/failure 0 show low-presence gaps and width fluctuations; case 3 has GT farther from the historical width than the erroneous endpoint. A physical constant fiber width does not imply a constant predicted width.
- No +256 trace-offset samples exist after final departure because tracing restarts earlier (160-base-voxel grace). No continuation was fabricated. This analysis does not replace a controlled rerun of any proposed gate.

## Reproduce

Control action `reviewed_width_paths` in scripts/training_failure_mining_control.json.

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```
