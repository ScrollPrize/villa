# Four intact reviewed fibers — 2026-10-12

Input: `data/bbox_gt/`, four unchanged VC3D JSON curves, PHercParis4/20260411134726.
No saved links; no links were invented. Full saved polylines preserved, including
the saved cspline span in000004. AFV exact-coordinate roundtrip validated.

## Final trusted-GT run

`data/bbox_gt_solve_v1/uv_whole/` —4wholefibers,0splits,0continuations,
broken category structurally disabled. All preprocessing stages retained all4.
16perpendicular Lasagna constraints,0parallel; all16haveknownoriented signs.
All4H/Vpairs connected (1–3:5,1–4:5,2–3:5,2–4:1events).

Current baseline loss weights/LRs,151hypotheses,CCW,seed0. Fresh non-reference
initialization, full-corpus radial calibration (not calibration to just4fibers).
10080full-constraint updates,one nonexclusive managed GPU; all variables jointly
optimized except the deliberately disabled broken category. No reference surface
supervision. Unlike the large baseline, each update visits all16events exactly.

| Metric | Result |
|---|---:|
| Retained fibers/length |4/4;100%|
| Retained constraints |16/16;100%|
| Bucket sign violations |0/16;0%|
| Fitted-position sign violations |0/16;0%|
| Mean absolute bucket-vs-measured winding-distance residual |0.27830turns|
| Final objective (last80updates mean, corrected diagnostic denominator) |0.21818810|
| Solve walltime |129.49s|
| Peak allocated CUDA memory |5,075,456bytes|

No sign violations from2000updates onward. Native extraction kernel0.036s;
native process~0.19s. Timing is not an exclusive performance benchmark.
The saved raw final diagnostic reports0.00173165 because full-pass benchmark
logging divided the accumulated last80updates by10080, despite resetting the
accumulator at10000. Corrected mean is raw×126=0.21818810. This affects logging
only, not gradients or assignments; denominator fixed for future runs. Raw
artifacts retained without rewriting. Single-final-diagnostic search unaffected.

| File suffix | Family | Fitted winding coordinate at stored start | At stored end |
|---|---|---:|---:|
|000001|H|22.63025|16.04167|
|000002|H|14.20223|22.53786|
|000003|V|17.71787|17.71787|
|000004|V|16.72695|16.72695|

These are inferred coordinates with arbitrary global winding origin, NOT spiral
reference IDs. The V fibers are separated by0.99091turn. H endpoints follow
stored point order, so one runs toward decreasing U and one increasing U.
Fitted categorical probabilities exceed99.9999%; this is optimizer confidence,
not calibrated anatomical certainty or proof of a unique optimum.

## Artifacts / reproduction

- `data/bbox_gt_solve_v1/report.json`: per-fiber probabilities/positions, all16
  measured distances, bucket/position differences and signs.
- `data/bbox_gt_solve_v1/fibers.sources.json`: filename/ID mapping, hashes, modes.
- `data/bbox_gt_solve_v1/vis/four_fibers/layout.tif`:1000pxhigh,LZW,equalaxis scale.
- Same directory: `layout.svg`, `layout.pdf`, separate confidence histograms.
  Existing renderer: Hblue,Vorange,one segmentperwholefiber, windingboundaries.

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_bbox_gt.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_bbox_gt_vis.json
```

Runner reuses completed unchanged stages; changed input/template configuration
requires a new output root.12CPUtests pass(adapter,151radial transfer,GTmask,
bucketmath); CUDA combined-coordinate gradient/repeatvalidation passed.

## Superseded trial / limitations

Initial baseline-default split trial (`uv/`) stopped at~6000updates following
userclarification:9pieces,3rejected. It is NOT this result and NOT GT evidence.
Only `uv_whole/` is the requested solve. Original baseline/corpus unchanged.
No explicit adjacency annotations generated; the map remains inferred. No
parallel events means direction stages use umbilicus priors rather than
inter-fiber parallel agreement. Absolute origin remains unanchored.
