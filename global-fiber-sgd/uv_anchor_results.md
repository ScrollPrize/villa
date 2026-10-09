# Shared-anchor global layout: first full comparison

## Result

Implemented and tested after the accepted normal, H/V, H-direction and V-direction
stages. CCW and CW are effectively tied in this first optimization budget. Neither
is promoted as a reliable flattening: many assignments remain ambiguous, selected
offsets exceed the soft bound, and continuation alignment remains poor.

Input:489197retained original Paris4 AFV fibers,489663pieces,6606100constraints,
466continuations. All23501earlier rejected fibers remain excluded. Source geometry
and extracted constraints unchanged. Independent preparation audit confirms the
two trials differ only in intended H/phase convention (float32 wrapped-phase
arithmetic discrepancy <=1.19e-7turns).

| H convention | Rejected pieces | Retained length | Retained events | Sign violations | Ambiguous (<90%) | Selected offsets outside +/-0.25 |
|---|---:|---:|---:|---:|---:|---:|
| CCW |3.3374%|91.4355%|82.6505%|6.4448%|54.1901%|11.9900%|
| CW |3.3686%|91.3679%|82.5233%|6.4855%|54.1724%|12.0887%|

Sign violations use retained known-sign events, tolerance1e-4winding, not all
original events or anatomical GT. Initial full-graph sign errors were41.7028%
(CCW) and41.5456%(CW), with100%retention. Rejected counts16342/16495pieces.
These are not controlled comparisons to historical U/V trials, which lacked
the direction-stage inputs and used a different coordinate/loss formulation.

| H convention | Parallel sign violations | Perpendicular sign violations | Continuations with both ends retained | Median retained endpoint mismatch (base vx) |
|---|---:|---:|---:|---:|
| CCW |3.4992%|18.7328%|40/466|20045.5|
| CW |3.5396%|18.7739%|35/466|14858.1|

## Protocol and cost

10000CUDA steps,4096event samples and4096uniform piece samples perstep;
6.20033sampled event epochs, seed0, all100x100hypothesis pairs per sampled event.
Same parameters/input apart from H convention. Both shared one managed H10080GB;
wall times are workload measurements, not isolated speed benchmarks:

- CCW preparation154.90s; solve1022.35s (17.04min).
- CW preparation155.69s; solve1023.32s (17.06min).
- Each peak PyTorch allocation5522770944bytes (~5.143GiB), within24GiBgrant.
- Shared-anchor code revision based on a99702997 plus this working diff; exact
  optimizer SHA and source hashes stored in each frozen config.json/summary.json.

Exact approved commands, model equations, normalization and state layout are in
[uv_layout.md](uv_layout.md). Configs:run_uv_anchor_ccw.json/run_uv_anchor_cw.json;
outputs:data/uv_anchor_v1/ccw and data/uv_anchor_v1/cw. Completed runs are protected
against overwrite. run_uv_anchor_full_report.json regenerates the audited tables
without optimization; hashes/partition checks in each run's audit.json.

## Findings and limitations

- The requested numeric threshold0.25turns is90degrees, not45degrees. It is a
  strong squared hinge, not clipping or a guarantee. Selected retained |offset|
  median/p95/p99/max:CCW .1390/.2851/.3639/1.0617turns;
  CW .1395/.2852/.3642/1.0606. Probability mass outsidequarter11.11%/11.19%.
  Some all-bucket separations are negative; do not claim guaranteed distinct
  hypotheses. Independent audit finds crossings in only5retained pieces perrun
  (12adjacent pairsCCW,7CW), not widespread collapse. CW also has7rejected
  crossing rows. Minimum retained separationCCW-.42055/CW-.28334turns.
  All-bucket averages hide selected-hypothesis offset excursions.
- Shared phase is deliberately unwrapped. It can change integer turns, so saved
  bucket IDs are relative hypotheses; derived q is the actual layout coordinate.
- Pilot audit identifies incoherent radial initialization across pieces of one
  original fiber (1–17windings apart in that pilot). Compatible continuation
  states exist but may have negligible probability. Coherent same-source initial
  hypotheses are a clear follow-up; not silently changed during this comparison.
- Uniform regularization and degree-biased event updates have different row
  frequencies. Larger independent regularization batches or correctly weighted
  touched-row regularization are potential improvements, not implemented here.
- Independent argmax readout is unreliable while54%of pieces have no >=90%state.
  This finite sampled budget does not establish convergence or anatomical accuracy.

## Validation

49regression tests pass7.03s, covering prior stages plus unwrapped seams, gradients,
likelihood reinforcement, rejection gating, separate sign penalty, centered hinge,
frozen masks/provenance and both direction conventions. Independent full prepared
graph, pilot and full saved-state audits pass; final saved-state audit recorded in
task_log.md and uv_anchor_full_results.md. No new dependencies/native build.

Pilot results and the expected-cost control:uv_anchor_pilot_results.md. The first
weak-offset/old-continuation-normalization pilot is archived at
data/uv_anchor_v1/pilot_ccw, not a recommended result. Revised pilots use the same
final loss settings as the full trials, but10000fibers/500steps/1024batch.
