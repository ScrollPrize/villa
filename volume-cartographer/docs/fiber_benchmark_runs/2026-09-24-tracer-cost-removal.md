# Tracer cost versus reference-based piece removal

## What was measured

Production VC3D beam forward-step losses, not BP energy and not fiberlet costs.
Re-generated only the three existing dense trace datasets, with optional diagnostic
capture; native checks require exact saved point coordinates/order, whole-trace
costs and lengths before exporting. All three checks passed. Existing BP and
reference-based removal results are reused unchanged; no BP or inference rerun.

BP pieces overlap, so length-weighted percentages below count piece support,
not unique physical fiber length. Cost totals integrate fractional boundary steps.
Maximum step cost is the full original forward-step loss for any step overlapping
the piece, not a prorated boundary contribution. Density is cost per prediction
voxel (8base voxels here); native beam steps are nominally8base voxels. These
are arbitrary tracer-loss units, not percentages/probabilities.

Classic1024 and New1024 are the matched historical1024crop beam8 runs with
original20/80suppression,25degree gate,no reverse check and historical BP weights.
NewReverse2048 is the current larger2048crop with40/80,45degree and20/40reverse
check,40base backoff, and sign-focused BP. Do not attribute differences between
1024and2048to the prediction model alone.

## Removed versus retained

Each cost cell is **mean / median / p95** across pieces. Full p05/max values
and base-voxel densities are in the JSON.

|Dataset|Group|Pieces (%)|Total piece cost|Mean cost density|Worst original step cost|
|---|---|---:|---:|---:|---:|
|Classic1024|retained|4468 (65.39%)|16.373 / 13.872 / 34.337|0.302 / 0.254 / 0.638|0.721 / 0.471 / 1.997|
|Classic1024|removed|2365 (34.61%)|27.540 / 26.160 / 53.418|0.514 / 0.484 / 0.971|1.936 / 0.978 / 7.484|
|New1024|retained|893 (74.17%)|17.801 / 12.336 / 51.359|0.327 / 0.247 / 0.901|0.692 / 0.614 / 1.228|
|New1024|removed|311 (25.83%)|30.356 / 30.036 / 65.356|0.550 / 0.550 / 1.157|1.827 / 0.971 / 6.862|
|NewReverse2048|retained|5706 (84.04%)|15.379 / 9.768 / 42.607|0.381 / 0.247 / 1.001|0.725 / 0.580 / 1.453|
|NewReverse2048|removed|1084 (15.96%)|15.940 / 11.465 / 43.636|0.522 / 0.401 / 1.141|0.926 / 0.760 / 2.389|

For current NewReverse2048, removed median density is 62.7%higher and median peak-step cost is 31.1%higher. This is separation, not a clean cutoff: distributions overlap substantially. Removal is a reference-assisted policy label, not independent GT error annotation.

## Removal reasons

- Classic1024: removed {"input_defect":991,"ref_conflict":195,"sign_cover":1179}; retained {"h":1987,"v":2406,"disabled":75}.
- New1024: removed {"sign_cover":170,"input_defect":108,"ref_conflict":33}; retained {"v":472,"h":412,"disabled":9}.
- NewReverse2048: removed {"input_defect":1018,"disconnected":66}; retained {"v":2954,"h":2648,"disabled":104}.

Retained includes final-solve disabled pieces; it does not mean active or proven correct. These labels come from `_oracle_fresh_*` named OBJ groups;
original piece IDs partition each initial consistency CSV exactly.

## Initial classification before reference removal

First is the **orientation prepass** (H/V/mixed); second is the **initial winding
solve** (H/V/defect). Neither table is post-removal. Error/defect is a solver
state, not a measured GT tracing error. Values are piece-count percentages,
with length-weighted percentages in parentheses.

### Orientation prepass

|Dataset|Pieces|H % (length %)|V % (length %)|mixed % (length %)|
|---|---:|---:|---:|---:|
|Classic1024|6833|45.27 (45.21)|54.35 (54.39)|0.38 (0.40)|
|New1024|1204|48.09 (47.23)|51.41 (52.23)|0.50 (0.54)|
|NewReverse2048|6790|46.79 (46.31)|53.21 (53.69)|0.00 (0.00)|

### Initial winding solve

|Dataset|Pieces|H % (length %)|V % (length %)|defect % (length %)|
|---|---:|---:|---:|---:|
|Classic1024|6833|39.25 (39.08)|46.70 (46.66)|14.05 (14.26)|
|New1024|1204|44.68 (43.94)|47.43 (48.04)|7.89 (8.02)|
|NewReverse2048|6790|40.24 (40.15)|44.17 (45.03)|15.60 (14.82)|

On the matched1024comparison, initial defect fraction falls from14.05%to7.89%
(14.26%to8.02%by piece-support length). That is distinct from the much smaller
orientation-mixed fraction. The current2048case is supplied for context, not a
matched Classic comparison.

## Reproduce and validation

```sh
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_trace_cost_audit_20260924.json
/home/hendrik/.venv_las/bin/python volume-cartographer/scripts/analyze_trace_cost_removal.py volume-cartographer/docs/fiber_benchmark_runs/2026-09-24-tracer-cost-inputs.json volume-cartographer/docs/fiber_benchmark_runs/2026-09-24-tracer-cost-results.json
/home/hendrik/.venv_las/bin/python -m pytest -q volume-cartographer/scripts/tests/test_trace_cost_removal.py
```

86native crop tests passed (including captured-cost sum, reverse slicing and
unchanged geometry tests); two Python boundary/coverage tests passed. Independent
review checked reverse cost ordering, piece mapping, and removal partition.
Exact-retrace records and step CSVs are in
`/ephemeral/hendrik/fiber_crop_bench/20260924_cost_audit/`.
Retained geometry and totals matched prior artifacts exactly; segment-cost sums
and piece coverage checks passed. RelWithDebInfo,128workers, single diagnostic
run each: Classic50.58s,new1024 4.83s,new2048 128.43s (not controlled speedup measurements).

[Aggregate results](2026-09-24-tracer-cost-results.json) ·
[Per-piece costs and labels](2026-09-24-tracer-cost-results.pieces.csv) ·
[Input mapping](2026-09-24-tracer-cost-inputs.json)
